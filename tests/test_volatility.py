from utils.volatility import VolatilityTracker

def test_volatility_tracker():
    tracker = VolatilityTracker(window_size=5)
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    vols = [tracker.update(s) for s in scores]
    
    assert vols[-1] > 0, "Volatility should be >0 with varying scores"
    assert tracker.update(0.5) >= 0, "Volatility must always be non-negative"
