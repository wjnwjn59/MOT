from modules.maritime_analyzer import taxonomy as tx


def test_attribute_names_are_unique_and_complete():
    names = tx.attribute_names()
    assert len(names) == 9
    assert len(set(names)) == 9
    assert "scale_variation" in names
    assert "specular_glare" in names


def test_ids_are_contiguous_and_unique():
    ids = [a["id"] for a in tx.ATTRIBUTES]
    assert sorted(ids) == list(range(9))


def test_oracle_vlm_partition():
    assert tx.oracle_attributes() == [
        "scale_variation", "low_resolution", "low_contrast", "motion_blur", "out_of_frame",
    ]
    assert tx.vlm_attributes() == [
        "occlusion", "background_clutter", "specular_glare", "illumination_appearance_change",
    ]


def test_prompt_mentions_every_vlm_attribute_and_severity():
    prompt = tx.build_vlm_prompt()
    for name in tx.vlm_attributes():
        assert name in prompt
    assert "severity" in prompt
    assert "JSON" in prompt
