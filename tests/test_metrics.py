from alignment_metaplasticity.metrics import harmonic_mean


def test_harmonic_mean():
    assert abs(harmonic_mean(1.0, 1.0) - 1.0) < 1e-9
    assert harmonic_mean(1.0, 0.0) == 0.0
    assert 0.79 < harmonic_mean(0.8, 0.8) < 0.81
