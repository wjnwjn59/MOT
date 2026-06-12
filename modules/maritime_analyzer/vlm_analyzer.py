"""Soft multi-attribute maritime-condition VLM annotator (schema v2).

Produces, per frame, a soft probability for each subjective ("vlm") attribute in
the taxonomy plus an overall severity and a self-consistency agreement score.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np
import torch
from PIL import Image, ImageDraw

from transformers import AutoTokenizer

_SOFT_SYSTEM_PROMPT = (
    "You are an expert annotator for maritime visual object tracking. Given a template image "
    "of a target object (Image A) and a later frame in which the same target is marked by a "
    "bounding box (Image B), you assess the visual challenges that affect the target. You "
    "evaluate every listed challenge independently, report calibrated probabilities in the "
    "closed interval [0, 1], and respond strictly in the requested JSON format with no "
    "additional text."
)


def parse_vlm_json(raw_text: str, attr_names: List[str]) -> Optional[Dict[str, float]]:
    """Extract the JSON object from a raw VLM string and clamp each value to [0,1]."""
    try:
        s = raw_text.find('{'); e = raw_text.rfind('}')
        if s == -1 or e == -1 or s >= e:
            return None
        data = json.loads(raw_text[s:e + 1])
    except Exception:
        return None
    out: Dict[str, float] = {}
    for k in list(attr_names) + ["severity"]:
        v = data.get(k, 0.0)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0
        out[k] = min(1.0, max(0.0, v))
    return out


def aggregate_passes(parsed_list: List[Optional[Dict[str, float]]], attr_names: List[str]) -> Dict:
    """Average parsed passes per attribute; agreement = 1 - 2*mean(std) over attributes, in [0,1]."""
    keys = list(attr_names) + ["severity"]
    valid = [p for p in parsed_list if p is not None]
    if not valid:
        return {**{k: 0.0 for k in keys}, "vlm_agreement": 0.0}
    means = {k: float(np.mean([p[k] for p in valid])) for k in keys}
    if len(valid) > 1:
        stds = [float(np.std([p[k] for p in valid])) for k in attr_names]
        agreement = float(max(0.0, 1.0 - 2.0 * float(np.mean(stds))))
    else:
        agreement = 1.0
    return {**means, "vlm_agreement": agreement}


@dataclass
class VLMConfig:
    model_name: str = "/media/vli-ws2/ade81ca2-2fce-49cb-a163-e1ee8090540b/thangdd_workspace/llm_checkpoints/Qwen_Qwen3-VL-32B-Instruct"  # local path or hub id
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    temperature: float = 0.2
    max_new_tokens: int = 128
    passes: int = 3
    verbose: bool = True
    gpu_memory_utilization: float = 0.9
    seed: int = 42  # per-request sampling seed for reproducible generation
    tensor_parallel_size: int = 1  # GPUs to shard the model across (set >1 for large models)
    enforce_eager: bool = False  # disable CUDA graphs / torch.compile (robustness over speed)
    disable_custom_all_reduce: bool = False  # fall back to NCCL; fixes some multi-GPU TP hangs


class VLMAnalyzer:
    def __init__(self, config: VLMConfig = VLMConfig()):
        from vllm import LLM, SamplingParams
        self.config = config

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

        # Initialize vLLM engine
        self.llm = LLM(
            model=config.model_name,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=32768,
            max_num_seqs=1,  # Single sequence processing for this use case
            dtype="auto",
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=config.disable_custom_all_reduce,
        )

        # Sampling parameters (used as a template; classify_soft overrides the seed per pass)
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=0.9,
            max_tokens=config.max_new_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id] if hasattr(self.tokenizer, 'eos_token_id') else None
        )

        if self.config.verbose:
            print(f"[VLMAnalyzer] Loaded vLLM model: {config.model_name}")

    @staticmethod
    def _draw_bbox_on_image(image: Image.Image, bbox, color=(255, 215, 0), width=5) -> Image.Image:
        img = image.copy()
        draw = ImageDraw.Draw(img)
        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
        return img

    def classify_soft(self, template_crop_path: str, frame_full_path: str,
                      frame_bbox) -> Dict:
        """Soft multi-attribute VLM annotation for the subjective attributes + severity."""
        from modules.maritime_analyzer.taxonomy import build_vlm_prompt, vlm_attributes
        from vllm import SamplingParams
        frame_img = Image.open(frame_full_path).convert('RGB')
        frame_boxed = self._draw_bbox_on_image(frame_img, frame_bbox)
        template_img = Image.open(template_crop_path).convert('RGB')

        attr_names = vlm_attributes()
        question = build_vlm_prompt()
        prompt = (f"<|im_start|>system\n{_SOFT_SYSTEM_PROMPT}<|im_end|>\n"
                  f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                  f"<|vision_start|><|image_pad|><|vision_end|>"
                  f"{question}<|im_end|>\n<|im_start|>assistant\n")

        parsed = []
        for i in range(self.config.passes):
            inputs = {"prompt": prompt,
                      "multi_modal_data": {"image": [template_img, frame_boxed]}}
            # Per-pass seed: reproducible across runs, yet varied across passes so the
            # self-consistency / agreement signal is preserved.
            sp = SamplingParams(
                temperature=self.config.temperature,
                top_p=0.9,
                max_tokens=self.config.max_new_tokens,
                seed=self.config.seed + i,
                stop_token_ids=self.sampling_params.stop_token_ids,
            )
            out = self.llm.generate([inputs], sampling_params=sp)
            parsed.append(parse_vlm_json(out[0].outputs[0].text.strip(), attr_names))
        return aggregate_passes(parsed, attr_names)
