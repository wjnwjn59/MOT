from __future__ import annotations
from typing import Dict, List

# source: "oracle" (deterministic) or "vlm" (subjective soft probability)
ATTRIBUTES: List[Dict] = [
    {"name": "scale_variation", "id": 0, "source": "oracle"},
    {"name": "low_resolution", "id": 1, "source": "oracle"},
    {"name": "low_contrast", "id": 2, "source": "oracle"},
    {"name": "motion_blur", "id": 3, "source": "oracle"},
    {"name": "out_of_frame", "id": 4, "source": "oracle"},
    {"name": "occlusion", "id": 5, "source": "vlm",
     "desc": "the target is partly blocked by another object/vessel (NOT cut by the frame edge)"},
    {"name": "background_clutter", "id": 6, "source": "vlm",
     "desc": "water texture, wakes, foam, or other vessels make the target hard to distinguish"},
    {"name": "specular_glare", "id": 7, "source": "vlm",
     "desc": "strong sun glare or specular reflection on the water near/over the target"},
    {"name": "illumination_appearance_change", "id": 8, "source": "vlm",
     "desc": "the target's brightness, colour, pose or appearance changed notably vs the template"},
]

SEVERITY_KEY = "severity"
SCHEMA_VERSION = 2


def attribute_names() -> List[str]:
    return [a["name"] for a in ATTRIBUTES]


def oracle_attributes() -> List[str]:
    return [a["name"] for a in ATTRIBUTES if a["source"] == "oracle"]


def vlm_attributes() -> List[str]:
    return [a["name"] for a in ATTRIBUTES if a["source"] == "vlm"]


def _vlm_full() -> List[Dict]:
    return [a for a in ATTRIBUTES if a["source"] == "vlm"]


def build_vlm_prompt() -> str:
    lines = [
        "You are an expert maritime vision annotator.",
        "Image A is a template crop of a target. Image B is a new frame with the target boxed.",
        "For EACH challenge below, output a probability in [0,1] that it is present for the boxed target in Image B.",
        "Also output 'severity' in [0,1]: overall how hard the target is to track in this frame.",
        "",
        "Challenges:",
    ]
    for a in _vlm_full():
        lines.append(f"- {a['name']}: {a['desc']}")
    lines += [
        "",
        "Return STRICT JSON only: a float for each challenge name and for 'severity'. Example:",
        '{"occlusion": 0.0, "background_clutter": 0.0, "specular_glare": 0.0, '
        '"illumination_appearance_change": 0.0, "severity": 0.0}',
    ]
    return "\n".join(lines)
