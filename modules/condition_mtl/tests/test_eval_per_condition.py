from modules.condition_mtl.eval_per_condition import per_condition_metrics


def test_per_condition_ao_sr():
    attr_names = ["a", "b"]
    frames = [
        {"iou": 0.8, "attr_probs": [0.9, 0.1]},  # a present
        {"iou": 0.2, "attr_probs": [0.7, 0.0]},  # a present
        {"iou": 0.6, "attr_probs": [0.0, 0.8]},  # b present
    ]
    m = per_condition_metrics(frames, attr_names, present_threshold=0.5, success_threshold=0.5)

    assert m["a"]["n"] == 2
    assert abs(m["a"]["AO"] - 0.5) < 1e-6        # (0.8 + 0.2) / 2
    assert abs(m["a"]["SR"] - 0.5) < 1e-6        # only 0.8 >= 0.5

    assert m["b"]["n"] == 1
    assert abs(m["b"]["AO"] - 0.6) < 1e-6
    assert abs(m["b"]["SR"] - 1.0) < 1e-6

    assert m["overall"]["n"] == 3
    assert abs(m["overall"]["AO"] - (1.6 / 3)) < 1e-6
    assert abs(m["overall"]["SR"] - (2 / 3)) < 1e-6
