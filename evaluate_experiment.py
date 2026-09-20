cat > src/evaluate_experiment.py <<'PY'
import argparse
import math
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

import boto3
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from config import AWS_REGION, PREDICTIONS_TABLE, EXPERIMENTS_TABLE

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

def to_decimal(value, places=6):
    """
    Converte float/int/stringa numerica in Decimal, formato supportato
    da DynamoDB. La conversione da str evita errori di precisione binaria.
    """
    quantizer = Decimal("1." + ("0" * places))
    return Decimal(str(value)).quantize(
        quantizer,
        rounding=ROUND_HALF_UP
    )

def dynamodb_safe(value):
    """
    Converte ricorsivamente float presenti in dict/list di scikit-learn
    in Decimal, per rendere il documento scrivibile in DynamoDB.
    """
    if isinstance(value, float):
        return to_decimal(value)

    if isinstance(value, dict):
        return {
            str(key): dynamodb_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [dynamodb_safe(item) for item in value]

    return value

def read_predictions(table, experiment_id):
    """
    Legge tutte le predizioni del singolo esperimento tramite Query
    sulla partition key experiment_id.
    """
    items = []
    last_key = None

    while True:
        query_args = {
            "KeyConditionExpression": "experiment_id = :exp",
            "ExpressionAttributeValues": {
                ":exp": experiment_id
            }
        }

        if last_key:
            query_args["ExclusiveStartKey"] = last_key

        response = table.query(**query_args)
        items.extend(response.get("Items", []))

        last_key = response.get("LastEvaluatedKey")

        if not last_key:
            break

    return items

def percentile(values, percentile_value):
    if not values:
        return 0.0

    values = sorted(values)
    position = (len(values) - 1) * percentile_value / 100
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[int(position)]

    return values[lower] + (values[upper] - values[lower]) * (
        position - lower
    )

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions stored in DynamoDB."
    )
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="Experiment ID to evaluate."
    )
    args = parser.parse_args()

    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    predictions_table = dynamodb.Table(PREDICTIONS_TABLE)
    experiments_table = dynamodb.Table(EXPERIMENTS_TABLE)

    started = time.perf_counter()
    predictions = read_predictions(predictions_table, args.experiment_id)

    if not predictions:
        print(f"No predictions found for experiment: {args.experiment_id}")
        return

    y_true = [item["gold_label"] for item in predictions]
    y_pred = [item["predicted_label"] for item in predictions]

    latencies = [
        float(item["total_latency_ms"])
        for item in predictions
        if "total_latency_ms" in item
    ]

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0
    ))

    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0
    )

    average_latency = sum(latencies) / len(latencies) if latencies else 0.0
    min_latency = min(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0
    p95_latency = percentile(latencies, 95)

    elapsed = time.perf_counter() - started

    experiment = {
        "experiment_id": args.experiment_id,
        "dataset": "FEVER development subset",
        "model": "cross-encoder/nli-MiniLM2-L6-H768",
        "evidence_source": "Wikipedia API live retrieval",
        "claims_evaluated": len(predictions),
        "accuracy": to_decimal(accuracy),
        "macro_f1": to_decimal(macro_f1),
        "labels_order": LABELS,
        "confusion_matrix": matrix.tolist(),
        "classification_report": dynamodb_safe(report),
        "average_total_latency_ms": to_decimal(average_latency, 2),
        "min_total_latency_ms": to_decimal(min_latency, 2),
        "max_total_latency_ms": to_decimal(max_latency, 2),
        "p95_total_latency_ms": to_decimal(p95_latency, 2),
        "evaluation_elapsed_seconds": to_decimal(elapsed, 4),
        "evaluated_at": datetime.now(timezone.utc).isoformat()
    }

    experiments_table.put_item(Item=experiment)

    print(f"Experiment: {args.experiment_id}")
    print(f"Claims evaluated: {len(predictions)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print()
    print("Labels order:")
    print(LABELS)
    print()
    print("Confusion matrix:")
    for row in matrix.tolist():
        print(row)
    print()
    print(f"Average latency: {average_latency:.2f} ms")
    print(f"P95 latency: {p95_latency:.2f} ms")
    print(f"Evaluation time: {elapsed:.2f} s")
    print(f"Metrics saved to FactExperiments: {args.experiment_id}")

if __name__ == "__main__":
    main()
PY