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

# Single-label system + user instructions
_SYSTEM_PROMPT = (
    "You are an expert maritime vision annotator. Given a template object (Image A) "
    "and a new frame with the target boxed (Image B), select the SINGLE most limiting tracking challenge. "
    "Return STRICT JSON only."
)
_SINGLE_LABEL_INSTR = (
    "ALLOWED labels (choose EXACTLY ONE):\n"
    "0 Occlusion\n1 Illumination Change\n3 Motion Blur\n4 Variance in Appearance\n"
    "5 Partial Visibility (cropped by frame edge)\n7 Background Clutter\n\n"
    "Rules:\n"
    "- Pick the single most limiting challenge for tracking the boxed target in Image B using Image A.\n"
    "- Do NOT judge Scale Variation, Low Resolution, or Low Contrast.\n"
    "- Motion Blur refers to blur of the target object, not general defocus.\n"
    "- Partial Visibility means the target is cut off by the image border.\n\n"
    "Return STRICT JSON:\n"
    "{\"label\": int, \"confidence\": float, \"uncertain\": bool}"
)

@dataclass
class VLMConfig:
    model_name: str = "unsloth/Qwen2-VL-7B-Instruct"  # your local/remote path is fine
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
            # Left pad for chat-gen stability
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

            # Fallback to Transformers Qwen2.5-VL
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
        """Overlay the target bbox on the frame before sending to the VLM."""
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
    def classify(self, template_crop_path: str, frame_full_path: str, frame_bbox: Tuple[float, float, float, float]) -> Dict[str, int]:
        """
        Returns a one-hot dict, exactly one of:
          occlusion, illu_change, motion_blur, variance_appear, partial_visibility, background_clutter
        is set to 1; all others 0. Also includes 'raw_confidence' for the chosen label.
        """
        # Read images & overlay bbox on frame
        frame_img = Image.open(frame_full_path).convert('RGB')
        frame_boxed = self._draw_bbox_on_image(frame_img, frame_bbox)
        template_img = Image.open(template_crop_path).convert('RGB')

        # Accumulate scores across passes
        score = {k: 0.0 for k in _KEY_FOR_ID.keys()}  # id -> sum of confidences / votes

        for _ in range(self.config.passes):
            if self._use_unsloth:
                raw = self._gen_unsloth(template_img, frame_boxed)
            else:
                # transformers path uses on-disk paths
                # (bbox overlay is only visual; we still provide original frame path here)
                raw = self._gen_transformers(template_crop_path, frame_full_path)

            # Parse single-label JSON
            label_id = None
            confidence = 0.0
            try:
                s = raw.find('{'); e = raw.rfind('}')
                data = json.loads(raw[s:e+1])
                if isinstance(data.get("label", None), int) and data["label"] in _KEY_FOR_ID:
                    label_id = data["label"]
                    confidence = float(data.get("confidence", 1.0))
                else:
                    # fallback heuristics if model returned the old multi-label shape
                    # prefer confidences dict if present
                    confs = data.get("confidences", {})
                    if isinstance(confs, dict):
                        # pick argmax among allowed ids present in dict
                        best_k, best_v = None, -1.0
                        for k in _KEY_FOR_ID:
                            v = float(confs.get(str(k), -1.0))
                            if v > best_v:
                                best_k, best_v = k, v
                        if best_k is not None:
                            label_id, confidence = best_k, max(0.0, best_v)
                    else:
                        # if labels list present, take the first allowed
                        labels = data.get("labels", [])
                        if labels:
                            for k in labels:
                                if k in _KEY_FOR_ID:
                                    label_id = k
                                    confidence = 1.0
                                    break
            except Exception:
                # If totally unparseable, skip this pass
                pass

            if label_id is not None:
                score[label_id] += max(0.0, confidence)
            else:
                # weak vote: if nothing parsed, don't add noise
                continue

        # Pick the single best label
        best_label = max(score.items(), key=lambda kv: kv[1])[0]

        # Build one-hot output
        out = {name: 0 for name in _KEY_FOR_ID.values()}
        out[_KEY_FOR_ID[best_label]] = 1
        out["raw_confidence"] = float(score[best_label] / max(1, self.config.passes))
        return out
