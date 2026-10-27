"""
tests/test_math_pipeline.py
Unit tests verifying linear algebra properties and sequential pipeline validity.
"""

import os
import pytest
import numpy as np

from src.preprocessor import DigitPreprocessor
from src.pca_svd import PCASVDAnalyzer
from src.similarity import SimilarityClassifier
from src.verification import MathematicalVerifier

@pytest.fixture
def dummy_data():
    np.random.seed(42)
    # 30 samples, 400 dimensions
    X = np.random.randn(30, 400)
    # L2 normalize rows
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    y = np.repeat(np.arange(10), 3)
    return X, y

def test_preprocessor_isomorphism():
    prep = DigitPreprocessor(target_size=(20, 20))
    mat = np.random.rand(20, 20) * 255.0
    vec = prep.vectorize(mat)
    assert vec.shape == (400,)
    # Test inverse isomorphism
    mat_recon = prep.unvectorize(vec)
    np.testing.assert_allclose(mat, mat_recon, atol=1e-12)

def test_preprocessor_l2_normalization():
    prep = DigitPreprocessor(target_size=(20, 20))
    vec = np.random.randn(400)
    norm_vec, l2_norm = prep.normalize_l2(vec)
    assert np.isclose(l2_norm, np.linalg.norm(vec, 2))
    assert np.isclose(np.linalg.norm(norm_vec, 2), 1.0, atol=1e-12)

def test_pca_svd_orthonormality(dummy_data):
    X, _ = dummy_data
    pca = PCASVDAnalyzer(n_components=10).fit(X)
    res = MathematicalVerifier.verify_orthonormality(pca.V_k)
    assert res["passed"]
    assert res["frobenius_error"] < 1e-10

def test_projection_matrix_idempotence(dummy_data):
    X, _ = dummy_data
    pca = PCASVDAnalyzer(n_components=10).fit(X)
    P = pca.get_projection_matrix()
    res = MathematicalVerifier.verify_projection_idempotence(P)
    assert res["passed"]
    assert res["frobenius_error"] < 1e-10

def test_projection_matrix_symmetry(dummy_data):
    X, _ = dummy_data
    pca = PCASVDAnalyzer(n_components=10).fit(X)
    P = pca.get_projection_matrix()
    res = MathematicalVerifier.verify_projection_symmetry(P)
    assert res["passed"]
    assert res["frobenius_error"] < 1e-10

def test_svd_energy_conservation(dummy_data):
    X, _ = dummy_data
    pca = PCASVDAnalyzer(n_components=10).fit(X)
    X_c = X - pca.mean_vector
    res = MathematicalVerifier.verify_svd_energy_conservation(X_c, pca.singular_values)
    assert res["passed"]
    assert res["relative_error"] < 1e-8

def test_cauchy_schwarz_inequality():
    u = np.random.randn(15)
    v = np.random.randn(15)
    res = MathematicalVerifier.verify_cauchy_schwarz(u, v)
    assert res["passed"]
    assert res["ratio"] <= 1.000001

def test_cosine_similarity_properties():
    clf = SimilarityClassifier()
    v = np.array([1.0, 2.0, 3.0, 4.0])
    # Self similarity must be 1.0
    assert np.isclose(clf.cosine_similarity(v, v), 1.0, atol=1e-10)
    # Orthogonal vectors must have similarity 0.0
    u = np.array([-2.0, 1.0, 0.0, 0.0])
    assert np.isclose(clf.cosine_similarity(u, v), 0.0, atol=1e-10)
    # Opposite vectors must have similarity -1.0
    assert np.isclose(clf.cosine_similarity(v, -v), -1.0, atol=1e-10)

def test_reconstruction_monotonicity(dummy_data):
    X, _ = dummy_data
    pca = PCASVDAnalyzer(n_components=20).fit(X)
    sample = X[0]
    res = MathematicalVerifier.verify_reconstruction_monotonicity(sample, pca, ks=[1, 2, 5, 10, 15, 20])
    assert res["passed"]
