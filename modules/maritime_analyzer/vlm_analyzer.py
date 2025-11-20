from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
from PIL import Image, ImageDraw
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ---- Allowed single-label classes predicted by the VLM ----
_ALLOWED = {
    0: "Occlusion",
    1: "Illumination Change",
    3: "Motion Blur",
    4: "Variance in Appearance",
    5: "Partial Visibility",
    7: "Background Clutter",
}
_KEY_FOR_ID = {
    0: "occlusion",
    1: "illu_change",
    3: "motion_blur",
    4: "variance_appear",
    5: "partial_visibility",
    7: "background_clutter",
}

_SYSTEM_PROMPT = (
    "You are an expert maritime vision annotator. Given a template object (Image A) "
    "and a new frame with the target boxed (Image B), select the SINGLE most limiting tracking challenge. "
    "Return STRICT JSON only."
)
_SINGLE_LABEL_INSTR = (
    "ALLOWED labels (choose EXACTLY ONE as most limiting), but ALSO provide a confidence for EACH label:\n"
    "0 Occlusion\n1 Illumination Change\n3 Motion Blur\n4 Variance in Appearance\n"
    "5 Partial Visibility (cropped by frame edge)\n7 Background Clutter\n\n"
    "Rules:\n"
    "- Pick the single most limiting challenge for tracking the boxed target in Image B using Image A.\n"
    "- Do NOT judge Scale Variation, Low Resolution, or Low Contrast.\n"
    "- Motion Blur refers to blur of the target object, not general defocus.\n"
    "- Partial Visibility means the target is cut off by the image border.\n\n"
    "Return STRICT JSON:\n"
    "{"
    "  \"label\": int, "
    "  \"confidences\": {\"0\": float, \"1\": float, \"3\": float, \"4\": float, \"5\": float, \"7\": float}, "
    "  \"uncertain\": bool"
    "}\n"
    "The confidences must be in [0,1] and sum approximately to 1.0."
)

@dataclass
class VLMConfig:
    model_name: str = "unsloth/Qwen2-VL-7B-Instruct"  # local path or hub id
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    temperature: float = 0.2
    max_new_tokens: int = 128
    passes: int = 3
    verbose: bool = True
    gpu_memory_utilization: float = 0.9

class VLMAnalyzer:
    def __init__(self, config: VLMConfig = VLMConfig()):
        self.config = config
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        
        # Initialize vLLM engine
        self.llm = LLM(
            model=config.model_name,
            gpu_memory_utilization=config.gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=32768,
            max_num_seqs=1,  # Single sequence processing for this use case
            dtype="auto",
        )
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=0.9,
            max_tokens=config.max_new_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id] if hasattr(self.tokenizer, 'eos_token_id') else None
        )
        
        if self.config.verbose:
            print(f"[VLMAnalyzer] Loaded vLLM model: {config.model_name}")

    # ---------- utils ----------
    @staticmethod
    def _draw_bbox_on_image(image: Image.Image, bbox, color=(255, 215, 0), width=5) -> Image.Image:
        img = image.copy()
        draw = ImageDraw.Draw(img)
        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
        return img

    def _create_classification_prompt(self) -> str:
        """Create classification prompt for tracking challenges (Qwen2.5-VL format)"""
        question = f"""You are an expert maritime vision annotator. Given a template object (Image A) and a new frame with the target boxed (Image B), select the SINGLE most limiting tracking challenge. Return STRICT JSON only.

{_SINGLE_LABEL_INSTR}"""

        # Qwen2.5-VL prompt format
        prompt = (f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
                 f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                 f"<|vision_start|><|image_pad|><|vision_end|>"
                 f"{question}<|im_end|>\n"
                 f"<|im_start|>assistant\n")
        
        return prompt

    def _gen_vllm(self, template_img: Image.Image, frame_img_boxed: Image.Image) -> str:
        """Generate response using vLLM"""
        prompt = self._create_classification_prompt()
        
        # Prepare input for vLLM
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": [template_img, frame_img_boxed]  # Template first, then current frame
            }
        }
        
        # Generate response
        outputs = self.llm.generate([inputs], sampling_params=self.sampling_params)
        response = outputs[0].outputs[0].text.strip()
        
        return response

    # ---------- public API ----------
    def classify(self, template_crop_path: str, frame_full_path: str, frame_bbox: Tuple[float, float, float, float]) -> Dict:
        """
        Returns:
          - one-hot flags for 6 classes (exactly one = 1),
          - raw_confidence for the chosen class,
          - vlm_scores: per-class confidences (continuous),
          - vlm_top: {id, name, confidence}.
        """
        frame_img = Image.open(frame_full_path).convert('RGB')
        frame_boxed = self._draw_bbox_on_image(frame_img, frame_bbox)
        template_img = Image.open(template_crop_path).convert('RGB')

        ids = list(_KEY_FOR_ID.keys())  # [0,1,3,4,5,7]
        sum_conf = {k: 0.0 for k in ids}
        parsed_count = 0

        for _ in range(self.config.passes):
            raw = self._gen_vllm(template_img, frame_boxed)

            # Parse JSON
            try:
                s = raw.find('{'); e = raw.rfind('}')
                if s == -1 or e == -1 or s >= e:
                    continue
                data = json.loads(raw[s:e+1])
            except Exception:
                continue

            # Get confidences; if missing, synthesize from label
            confs = data.get("confidences")
            label = data.get("label")
            if not isinstance(confs, dict):
                confs = {str(k): 0.0 for k in ids}
                if isinstance(label, int) and label in ids:
                    confs[str(label)] = float(data.get("confidence", 1.0))

            # Normalize so they roughly sum to 1
            tot = sum(float(confs.get(str(k), 0.0)) for k in ids)
            if tot > 0:
                for k in ids:
                    sum_conf[k] += float(confs.get(str(k), 0.0)) / tot
                parsed_count += 1
            else:
                if isinstance(label, int) and label in ids:
                    sum_conf[label] += 1.0
                    parsed_count += 1

        den = max(1, parsed_count)
        avg_conf = {k: sum_conf[k] / den for k in ids}

        # Argmax
        best_id = max(avg_conf.items(), key=lambda kv: kv[1])[0]
        best_name = _KEY_FOR_ID[best_id]
        best_conf = float(avg_conf[best_id])

        # Build outputs
        flags = {name: 0 for name in _KEY_FOR_ID.values()}
        flags[best_name] = 1
        flags["raw_confidence"] = best_conf  # type: ignore

        vlm_scores = { _KEY_FOR_ID[k]: float(avg_conf[k]) for k in ids }

        # No longer return vlm_top
        return {**flags, "vlm_scores": vlm_scores}

