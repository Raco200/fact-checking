"""
Batch fact-checking script v2 with configurable NEI threshold.
Reads claims from DynamoDB FactClaims table.
Model name and NEI threshold are required arguments.
"""
import argparse
import time
from datetime import datetime, timezone
from decimal import Decimal

import boto3

from config import AWS_REGION, CLAIMS_TABLE, PREDICTIONS_TABLE
from factcheck_core_v2 import FactChecker


def read_claims(table, limit):
    """
    Leggi claim da DynamoDB tramite Scan paginata.
    Escludi i record manuali sample_ perché non fanno parte di FEVER.
    """
    items = []
    last_key = None

    while len(items) < limit:
        scan_args = {
            "ProjectionExpression": "claim_id, claim, gold_label, #sp",
            "ExpressionAttributeNames": {"#sp": "split"},
            "Limit": min(100, limit - len(items))
        }

        if last_key:
            scan_args["ExclusiveStartKey"] = last_key

        response = table.scan(**scan_args)
        items.extend(response.get("Items", []))
        last_key = response.get("LastEvaluatedKey")

        if not last_key:
            break

    return [
        item for item in items
        if item["claim_id"].startswith("dev_")
    ][:limit]


def save_prediction(table, experiment_id, claim_item, result):
    item = {
        "experiment_id": experiment_id,
        "claim_id": claim_item["claim_id"],
        "claim": claim_item["claim"],
        "gold_label": claim_item["gold_label"],
        "predicted_label": result["pred_label"],
        "confidence": result["confidence"],
        "evidence_title": result.get("evidence_title") or "",
        "evidence_text": result.get("evidence_text") or "",
        "probs": result.get("probs", []),
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    table.put_item(Item=item)


def main():
    parser = argparse.ArgumentParser(
        description="Run Wikipedia retrieval + NLI over DynamoDB claims with NEI threshold."
    )
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="Unique experiment ID, e.g. deberta_base_100_v1"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of FEVER claims to process."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between claims, respecting Wikipedia."
    )
    parser.add_argument(
        "--nei-threshold",
        type=float,
        required=True,
        help="NOT ENOUGH INFO threshold (required, e.g. 0.40)."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HuggingFace model name (required, e.g. cross-encoder/nli-deberta-v3-base)."
    )
    args = parser.parse_args()

    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    claims_table = dynamodb.Table(CLAIMS_TABLE)
    predictions_table = dynamodb.Table(PREDICTIONS_TABLE)

    claims = read_claims(claims_table, args.limit)

    if not claims:
        print("No FEVER claims found.")
        return

    print(f"Loaded {len(claims)} FEVER claims.")
    print("Loading model once for the entire batch...")

    checker = FactChecker(
        nli_model_name=args.model_name,
        nei_threshold=args.nei_threshold
    )
    started = time.perf_counter()

    for index, claim_item in enumerate(claims, start=1):
        claim = claim_item["claim"]

        try:
            result = checker.fact_check(claim)
            save_prediction(
                predictions_table,
                args.experiment_id,
                claim_item,
                result
            )

            print(
                f"[{index}/{len(claims)}] "
                f"{claim_item['claim_id']} | "
                f"gold={claim_item['gold_label']} | "
                f"pred={result['pred_label']} | "
                f"confidence={float(result['confidence']):.6f}"
            )

        except Exception as error:
            print(
                f"[{index}/{len(claims)}] "
                f"{claim_item['claim_id']} FAILED: {error}"
            )

        if index < len(claims):
            time.sleep(args.delay)

    elapsed = time.perf_counter() - started
    throughput = len(claims) / elapsed if elapsed else 0.0

    print("\nBatch completed.")
    print(f"Experiment ID: {args.experiment_id}")
    print(f"Claims attempted: {len(claims)}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Throughput: {throughput:.3f} claims/second")


if __name__ == "__main__":
    main()

