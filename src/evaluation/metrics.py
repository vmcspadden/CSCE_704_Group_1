"""§11.1 metrics: precision, recall, MCC and absolute FP/FN counts.

Accuracy is never reported on its own. False positives are always reported
as an absolute count alongside the rate, e.g.
"12 FP out of 60,000 benign files (0.020% FPR)".
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import matthews_corrcoef


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {"tp": int(((y_true == 1) & (y_pred == 1)).sum()), "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
            "tn": int(((y_true == 0) & (y_pred == 0)).sum()), "fn": int(((y_true == 1) & (y_pred == 0)).sum())}


def report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    c = confusion(y_true, y_pred)
    n_ben, n_mal = c["fp"] + c["tn"], c["tp"] + c["fn"]
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    recall = c["tp"] / n_mal if n_mal else 0.0
    fpr = c["fp"] / n_ben if n_ben else 0.0
    mcc = float(matthews_corrcoef(y_true, y_pred)) if n_ben and n_mal else 0.0
    return {
        **c, "n_benign": n_ben, "n_malicious": n_mal,
        "precision": precision, "recall_tpr": recall, "fpr": fpr, "mcc": mcc,
        "false_positives": f"{c['fp']:,} FP out of {n_ben:,} benign files scanned ({100 * fpr:.3f}% FPR)",
        "false_negatives": f"{c['fn']:,} FN out of {n_mal:,} malicious files ({100 * (1 - recall):.3f}% FNR)",
    }


def threshold_for_fpr(benign_scores: np.ndarray, allowed_fp: int) -> float:
    """Smallest threshold t (verdict = score >= t) giving at most ``allowed_fp``
    benign scores >= t."""
    s = np.sort(np.asarray(benign_scores, dtype=np.float64))[::-1]
    if len(s) == 0:
        return 0.5
    allowed_fp = max(0, int(allowed_fp))
    if allowed_fp >= len(s):
        return float(s[-1])
    # threshold strictly above the (allowed_fp+1)-th highest benign score
    return float(np.nextafter(s[allowed_fp], np.inf))
