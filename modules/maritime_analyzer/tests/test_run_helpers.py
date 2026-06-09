from modules.maritime_analyzer.run import (
    shard_sequences, plan_gpu_groups, build_worker_commands, build_record,
)


def test_shard_sequences_round_robin():
    assert shard_sequences(["a", "b", "c", "d", "e"], 2) == [["a", "c", "e"], ["b", "d"]]
    assert shard_sequences(["a", "b"], 1) == [["a", "b"]]


def test_plan_gpu_groups():
    assert plan_gpu_groups([0, 1, 2, 3], tp=2) == [[0, 1], [2, 3]]
    assert plan_gpu_groups([0, 1, 2, 3], tp=1) == [[0], [1], [2], [3]]


def test_build_worker_commands_sets_visible_devices():
    groups = [[0, 1], [2, 3]]
    cmds = build_worker_commands(num_shards=2, gpu_groups=groups, dataset="/data/MVTD/train",
                                 out_dir="data", model="Qwen/Qwen3.5-35B-A3B", tp=2, seed=42)
    assert len(cmds) == 2
    env0, argv0 = cmds[0]
    assert env0["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "--worker" in argv0 and "--shard-index" in argv0
    assert "0" in argv0 and "Qwen/Qwen3.5-35B-A3B" in argv0


def test_build_record_shape():
    oracle = {"scale_variation": 0, "low_resolution": 1, "low_contrast": 0,
              "motion_blur": 0, "out_of_frame": 0, "_features": {"highlight_ratio": 0.0}}
    vlm = {"occlusion": 0.8, "background_clutter": 0.0, "specular_glare": 0.0,
           "illumination_appearance_change": 0.6, "severity": 0.7, "vlm_agreement": 0.9}
    rec = build_record("1-Boat", 12, "00000012.jpg", [1, 2, 3, 4], [5, 6, 7, 8],
                       oracle, vlm, {"dataset_path": "/data/MVTD/train"})
    assert rec["schema_version"] == 2
    assert rec["attributes"]["low_resolution"] == {"prob": 1.0, "source": "oracle"}
    assert rec["attributes"]["occlusion"] == {"prob": 0.8, "source": "vlm"}
    assert rec["severity"] == 0.7 and rec["vlm_agreement"] == 0.9
    assert len(rec["attributes"]) == 9
