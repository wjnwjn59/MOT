from modules.maritime_analyzer.validation.audit_partial_visibility import compare_partial_visibility


def test_compare_counts_and_disagreement():
    # record: old VLM said partial(flag), bbox, seq, frame_file
    records = [
        {"sequence_name": "s", "frame_file": "1.jpg", "vlm_partial_flag": 1,
         "ground_truth_bbox": [10, 10, 20, 20]},   # oracle: inside -> disagree
        {"sequence_name": "s", "frame_file": "2.jpg", "vlm_partial_flag": 1,
         "ground_truth_bbox": [0, 10, 20, 20]},     # oracle: edge -> agree
        {"sequence_name": "s", "frame_file": "3.jpg", "vlm_partial_flag": 0,
         "ground_truth_bbox": [40, 40, 10, 10]},    # oracle: inside -> agree
        {"sequence_name": "s", "frame_file": "4.jpg", "vlm_partial_flag": 0,
         "ground_truth_bbox": [0, 5, 20, 20]},      # oracle: edge (x=0) -> VLM missed -> disagree
    ]
    def size_lookup(seq, frame_file):
        return (100, 100)
    stats = compare_partial_visibility(records, size_lookup, margin=1)
    assert stats["total"] == 4
    assert stats["agree"] == 2
    assert stats["disagree"] == 2
    assert abs(stats["disagreement_rate"] - 0.5) < 1e-6
    assert stats["vlm_partial_oracle_inside"] == 1   # VLM over-predicts partial
    assert stats["oracle_partial_vlm_no"] == 1       # VLM missed an edge-cropped target
