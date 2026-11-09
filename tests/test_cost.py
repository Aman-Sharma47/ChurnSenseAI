from core.cost import cost_curve


def test_cheap_offers_lower_the_best_threshold():
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    p = [0.9, 0.6, 0.3, 0.4, 0.2, 0.1, 0.05, 0.05]
    cheap = max(cost_curve(y, p, 500, 1), key=lambda r: r["net_savings"])
    pricey = max(cost_curve(y, p, 500, 400), key=lambda r: r["net_savings"])
    assert cheap["threshold"] <= pricey["threshold"]


def test_free_offers_reach_every_churner():
    y = [1, 0, 1, 0]
    p = [0.9, 0.1, 0.5, 0.2]
    best = max(cost_curve(y, p, 100, 0), key=lambda r: r["net_savings"])
    assert best["churners_reached"] == 2