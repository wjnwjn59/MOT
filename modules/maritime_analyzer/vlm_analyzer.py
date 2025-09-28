from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
from PIL import Image, ImageDraw

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

class VLMAnalyzer:
    def __init__(self, config: VLMConfig = VLMConfig()):
        self.config = config
        self._use_unsloth = False

        # Try Unsloth first
        try:
            from unsloth import FastVisionModel  # type: ignore
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.model, self.tokenizer = FastVisionModel.from_pretrained(  # type: ignore
                config.model_name, load_in_4bit=True, torch_dtype=dtype, device_map="auto",
            )
            if hasattr(self.tokenizer, "padding_side"):
                self.tokenizer.padding_side = "left"
            if hasattr(self.tokenizer, "truncation_side"):
                self.tokenizer.truncation_side = "left"
            FastVisionModel.for_inference(self.model)  # type: ignore
            self._use_unsloth = True
            if self.config.verbose:
                print(f"[VLMAnalyzer] Loaded Unsloth model: {config.model_name}")
        except Exception as e:
            if self.config.verbose:
                print(f"[VLMAnalyzer] Unsloth load failed ({e}); falling back to Transformers...")
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            self.Qwen2_5_VLForConditionalGeneration = Qwen2_5_VLForConditionalGeneration
            self.AutoProcessor = AutoProcessor
            self.process_vision_info = process_vision_info
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
            if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
                self.processor.tokenizer.padding_side = "left"
                if hasattr(self.processor.tokenizer, "truncation_side"):
                    self.processor.tokenizer.truncation_side = "left"
            self._use_unsloth = False
            if self.config.verbose:
                print("[VLMAnalyzer] Loaded Transformers fallback: Qwen/Qwen2.5-VL-7B-Instruct")

    # ---------- utils ----------
    @staticmethod
    def _draw_bbox_on_image(image: Image.Image, bbox, color=(255, 215, 0), width=5) -> Image.Image:
        img = image.copy()
        draw = ImageDraw.Draw(img)
        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
        return img

    def _single_label_messages_unsloth_text(self) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": _SINGLE_LABEL_INSTR},
            ]},
        ]
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    def _gen_unsloth(self, template_img: Image.Image, frame_img_boxed: Image.Image) -> str:
        text = self._single_label_messages_unsloth_text()
        inputs = self.tokenizer(
            [template_img, frame_img_boxed],
            text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.config.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                use_cache=True,
            )
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0]

    def _gen_transformers(self, template_path: str, frame_path: str) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": template_path},
                {"type": "image", "image": frame_path},
                {"type": "text", "text": _SINGLE_LABEL_INSTR},
            ]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.config.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
        trimmed = out[0, inputs.input_ids.shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

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
            raw = self._gen_unsloth(template_img, frame_boxed) if self._use_unsloth \
                  else self._gen_transformers(template_crop_path, frame_full_path)

            # Parse JSON
            try:
                s = raw.find('{'); e = raw.rfind('}')
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
        flags["raw_confidence"] = best_conf

        vlm_scores = { _KEY_FOR_ID[k]: float(avg_conf[k]) for k in ids }

        # No longer return vlm_top
        return {**flags, "vlm_scores": vlm_scores}

