"""
Fact-checking core logic v2 with configurable NOT ENOUGH INFO threshold.
Uses softmax to convert model logits to probabilities.
"""
from decimal import Decimal
import numpy as np
import requests
from sentence_transformers import CrossEncoder


class FactChecker:
    """
    Fact-checking pipeline using Wikipedia live retrieval + NLI model.
    
    Args:
        nli_model_name: HuggingFace model name for NLI (required).
        nei_threshold: Minimum confidence to assign SUPPORTS/REFUTES.
                       If max(probabilities) < nei_threshold → NOT ENOUGH INFO (required).
    """
    
    LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    
    def __init__(self, nli_model_name: str, nei_threshold: float):
        print(f"Loading NLI model: {nli_model_name}...")
        self.nli_model = CrossEncoder(nli_model_name)
        self.nei_threshold = Decimal(str(nei_threshold))
    
    def _search_wikipedia(self, query: str) -> list[dict]:
        """Search Wikipedia and return top results."""
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "format": "json",
            "srsearch": query,
            "srlimit": 3,
        }
        headers = {
            "User-Agent": "FactCheckBigDataProject/1.0 (https://github.com/yourusername/topic8-factcheck-andrea-2026)",
        }
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("query", {}).get("search", [])
    
    def _get_page_intro(self, title: str) -> str:
        """Fetch the introduction of a Wikipedia page."""
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
        }
        headers = {
            "User-Agent": "FactCheckBigDataProject/1.0 (https://github.com/yourusername/topic8-factcheck-andrea-2026)",
        }
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            return page.get("extract", "")
        return ""
    
    def _retrieve_evidence(self, claim: str) -> tuple[str, str]:
        """
        Retrieve evidence for a claim from Wikipedia.
        Returns (evidence_title, evidence_text).
        """
        results = self._search_wikipedia(claim)
        if not results:
            return "", ""
        
        top_title = results[0]["title"]
        intro = self._get_page_intro(top_title)
        if intro:
            return top_title, intro
        
        if len(results) > 1:
            second_title = results[1]["title"]
            intro = self._get_page_intro(second_title)
            if intro:
                return second_title, intro
        
        return "", ""
    
    def _predict_nli(self, premise: str, hypothesis: str) -> tuple[str, Decimal, list[Decimal]]:
        """
        Predict NLI label for (premise, hypothesis) pair.
        Applies softmax to convert logits to probabilities.
        
        Returns:
            pred_label: Predicted label (SUPPORTS, REFUTES, NOT ENOUGH INFO)
            confidence: Confidence score as Decimal
            probs: List of probabilities as Decimal [p_supports, p_refutes, p_nei]
        """
        scores = self.nli_model.predict([(premise, hypothesis)], convert_to_numpy=True, show_progress_bar=False)[0]
        
        # Apply softmax to convert logits to probabilities
        exp_scores = np.exp(scores - np.max(scores))
        probs_raw = (exp_scores / exp_scores.sum()).tolist()
        
        # cross-encoder models: 0=contradiction (REFUTES), 1=entailment (SUPPORTS), 2=neutral (NEI)
        p_refutes, p_supports, p_nei = probs_raw
        
        # Reorder to [SUPPORTS, REFUTES, NOT ENOUGH INFO]
        probs = [Decimal(str(p_supports)), Decimal(str(p_refutes)), Decimal(str(p_nei))]
        
        max_prob = max(probs)
        pred_label_id = int(np.argmax([float(p) for p in probs]))
        
        # Apply NEI threshold
        if max_prob < self.nei_threshold:
            pred_label = "NOT ENOUGH INFO"
            confidence = max_prob
        else:
            pred_label = self.LABELS[pred_label_id]
            confidence = max_prob
        
        return pred_label, confidence, probs
    
    def fact_check(self, claim: str) -> dict:
        """
        Perform fact-checking on a claim.
        
        Returns:
            dict with keys: claim, pred_label, confidence, evidence_title, evidence_text, probs
        """
        evidence_title, evidence_text = self._retrieve_evidence(claim)
        
        if not evidence_text:
            return {
                "claim": claim,
                "pred_label": "NOT ENOUGH INFO",
                "confidence": Decimal("0.0"),
                "evidence_title": "",
                "evidence_text": "",
                "probs": [Decimal("0.0"), Decimal("0.0"), Decimal("0.0")],
            }
        
        pred_label, confidence, probs = self._predict_nli(evidence_text, claim)
        
        return {
            "claim": claim,
            "pred_label": pred_label,
            "confidence": confidence,
            "evidence_title": evidence_title,
            "evidence_text": evidence_text,
            "probs": probs,
        }
EOF
