from __future__ import annotations
import json
from typing import Dict, List

# source: "oracle" (deterministic) or "vlm" (subjective soft probability)
ATTRIBUTES: List[Dict] = [
    {"name": "scale_variation", "id": 0, "source": "oracle"},
    {"name": "low_resolution", "id": 1, "source": "oracle"},
    {"name": "low_contrast", "id": 2, "source": "oracle"},
    {"name": "motion_blur", "id": 3, "source": "oracle"},
    {"name": "out_of_frame", "id": 4, "source": "oracle"},
    {"name": "occlusion", "id": 5, "source": "vlm",
     "desc": ("a portion of the target is hidden by another object or vessel positioned "
              "between the camera and the target. This does not include the target being "
              "cut off by the image boundary.")},
    {"name": "background_clutter", "id": 6, "source": "vlm",
     "desc": ("the region surrounding the target contains visually similar structures, such "
              "as water texture, wakes, foam, shoreline, or other vessels, that reduce the "
              "contrast between the target and its background.")},
    {"name": "specular_glare", "id": 7, "source": "vlm",
     "desc": ("direct sunlight, specular reflection, or bright highlights on the water "
              "surface overlap with or surround the target and degrade its visibility.")},
    {"name": "illumination_appearance_change", "id": 8, "source": "vlm",
     "desc": ("the appearance of the target in the current frame differs substantially from "
              "the template, for example in brightness, color, pose, orientation, or visible "
              "structure.")},
]

SEVERITY_KEY = "severity"
SCHEMA_VERSION = 2

# Every subjective (VLM) attribute must define a human-readable description used in the prompt.
assert all("desc" in a for a in ATTRIBUTES if a["source"] == "vlm"), \
    "All VLM attributes must define a 'desc'"


def attribute_names() -> List[str]:
    return [a["name"] for a in ATTRIBUTES]


def oracle_attributes() -> List[str]:
    return [a["name"] for a in ATTRIBUTES if a["source"] == "oracle"]


def vlm_attributes() -> List[str]:
    return [a["name"] for a in ATTRIBUTES if a["source"] == "vlm"]


def _vlm_full() -> List[Dict]:
    return [a for a in ATTRIBUTES if a["source"] == "vlm"]


def build_vlm_prompt() -> str:
    """Formal, ASCII-only instruction prompt for the soft multi-attribute VLM annotator."""
    example = json.dumps({k: 0.0 for k in vlm_attributes() + [SEVERITY_KEY]})
    lines = [
        "Task: assess the visual challenges affecting a single target in maritime object tracking.",
        "",
        "You are given two images. Image A is a cropped template of the target object. "
        "Image B is a later frame in which the same target is indicated by a bounding box.",
        "",
        "For each challenge listed below, estimate the probability, expressed as a real number "
        "in the closed interval [0, 1], that the challenge is present and materially affects the "
        "appearance or trackability of the boxed target in Image B. Assess each challenge "
        "independently; the probabilities need not sum to one. A value of 0 indicates that the "
        "challenge is clearly absent, a value of 1 indicates that it is clearly present and "
        "severe, and intermediate values express partial presence or uncertainty.",
        "",
        "Challenges:",
    ]
    for idx, a in enumerate(_vlm_full(), start=1):
        lines.append(f"{idx}. {a['name']}: {a['desc']}")
    lines += [
        "",
        f"In addition, report '{SEVERITY_KEY}': an overall estimate in [0, 1] of how difficult "
        "it is to localize the boxed target in Image B when all factors are considered jointly.",
        "",
        "Output format: respond with a single JSON object and nothing else. Do not include any "
        "explanation or text outside the JSON object. Use exactly the following keys, each "
        "mapped to a real number in [0, 1]:",
        example,
    ]
    return "\n".join(lines)
