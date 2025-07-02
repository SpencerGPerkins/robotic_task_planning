import numpy as np
from utils.coords import match_coords, normalize

def test_match_coords_equal():
    a = [1.00001, 2.00001, 3.00001]
    b = [1.00002, 2.00002, 3.00002]
    assert match_coords(a, b, tol=1e-3)

def test_match_coords_fail():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    assert not match_coords(a, b, tol=1e-3)

def test_normalize():
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    norm = normalize(arr)
    assert np.allclose(norm.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(norm.std(axis=0), 1, atol=1e-6)