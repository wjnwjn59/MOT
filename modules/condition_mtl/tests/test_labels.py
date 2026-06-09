from modules.maritime_analyzer.taxonomy import attribute_names
from modules.condition_mtl.labels import (
    record_attr_probs, record_severity, record_hard_label,
)

NAMES = attribute_names()  # 9, taxonomy order


def test_v2_record_attr_probs():
    rec = {"schema_version": 2, "severity": 0.7, "attributes": {
        "low_resolution": {"prob": 1.0, "source": "oracle"},
        "occlusion": {"prob": 0.9, "source": "vlm"},
    }}
    probs = record_attr_probs(rec, NAMES)
    assert len(probs) == 9
    assert probs[NAMES.index("low_resolution")] == 1.0
    assert probs[NAMES.index("occlusion")] == 0.9
    assert probs[NAMES.index("specular_glare")] == 0.0  # missing -> 0
    assert record_severity(rec) == 0.7


def test_v1_backward_compat_mapping():
    rec = {  # old format, no schema_version / attributes
        "vlm_response": {
            "variance_appear": {"flag": 1},      # -> illumination_appearance_change
            "partial_visibility": {"flag": 1},   # -> out_of_frame
        },
        "cv_response": {"low_res": 1, "scale_variation": 0},
    }
    probs = record_attr_probs(rec, NAMES)
    assert probs[NAMES.index("illumination_appearance_change")] == 1.0
    assert probs[NAMES.index("out_of_frame")] == 1.0
    assert probs[NAMES.index("low_resolution")] == 1.0
    assert probs[NAMES.index("scale_variation")] == 0.0


def test_hard_label_argmax_and_normal():
    rec = {"schema_version": 2, "attributes": {
        "low_resolution": {"prob": 1.0}, "occlusion": {"prob": 0.9}}}
    assert record_hard_label(rec, NAMES) == NAMES.index("low_resolution")

    normal_rec = {"schema_version": 2, "attributes": {"occlusion": {"prob": 0.1}}}
    assert record_hard_label(normal_rec, NAMES, normal_threshold=0.5) == len(NAMES)  # normal
