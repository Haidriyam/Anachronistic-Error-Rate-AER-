"""
Verification tests for surprisal calculations and cosine metric boundaries.
"""

import numpy as np
import pytest
from src.metrics import aggregate_audit_scores, compute_token_surprisal, cosine_similarity


def test_cosine_similarity_orthogonal():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    assert pytest.approx(cosine_similarity(v1, v2), abs=1e-6) == 0.0


def test_cosine_similarity_identical():
    v1 = np.array([2.5, -1.2, 0.8])
    assert pytest.approx(cosine_similarity(v1, v1), abs=1e-6) == 1.0


def test_surprisal_uniform_distribution():
    # 4 classes, uniform logits -> P = 0.25 -> S = -log2(0.25) = 2.0 bits
    logits = np.array([1.0, 1.0, 1.0, 1.0])
    s = compute_token_surprisal(0, logits)
    assert pytest.approx(s, abs=1e-4) == 2.0


def test_aer_calculation():
    records = [
        {"is_anachronistic": True, "delta_projection": 0.2, "surprisal_bits": 8.0},
        {"is_anachronistic": False, "delta_projection": -0.3, "surprisal_bits": 5.0},
    ]
    aer, mean_delta, mean_s = aggregate_audit_scores(records)
    assert aer == 0.5
    assert pytest.approx(mean_delta, abs=1e-4) == -0.05
    assert pytest.approx(mean_s, abs=1e-4) == 6.5
