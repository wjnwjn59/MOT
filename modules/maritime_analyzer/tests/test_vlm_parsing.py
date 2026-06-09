from modules.maritime_analyzer.vlm_analyzer import parse_vlm_json, aggregate_passes

ATTRS = ["occlusion", "background_clutter", "specular_glare", "illumination_appearance_change"]


def test_parse_extracts_and_clamps():
    raw = 'noise {"occlusion": 0.8, "specular_glare": 1.7, "severity": 0.5} trailing'
    out = parse_vlm_json(raw, ATTRS)
    assert out["occlusion"] == 0.8
    assert out["specular_glare"] == 1.0          # clamped to [0,1]
    assert out["background_clutter"] == 0.0      # missing -> 0
    assert out["severity"] == 0.5


def test_parse_returns_none_on_garbage():
    assert parse_vlm_json("no json here", ATTRS) is None


def test_aggregate_means_and_agreement_range():
    p1 = {"occlusion": 0.8, "background_clutter": 0.0, "specular_glare": 0.0,
          "illumination_appearance_change": 0.6, "severity": 0.7}
    p2 = {"occlusion": 0.4, "background_clutter": 0.0, "specular_glare": 0.0,
          "illumination_appearance_change": 0.6, "severity": 0.5}
    agg = aggregate_passes([p1, p2, None], ATTRS)
    assert abs(agg["occlusion"] - 0.6) < 1e-6
    assert abs(agg["severity"] - 0.6) < 1e-6
    assert 0.0 <= agg["vlm_agreement"] <= 1.0


def test_aggregate_empty_is_zero():
    agg = aggregate_passes([None], ATTRS)
    assert agg["occlusion"] == 0.0 and agg["vlm_agreement"] == 0.0
