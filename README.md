# Maritime Object Tracking

## To-do list:
- [x] Integrate MVTD dataset.
- [x] Evaluation kit.
- [x] Scripts for single pass running training and evaluation on any models
- [] Idea for improvement 1


## Setup

```python
export PYTHONPATH="./:$PYTHONPATH"
```

## Run

1. Create dataset for environment classification
```
CUDA_VISIBLE_DEVICES=1 python modules/maritime_analyzer/run.py --dataset /mnt/VLAI_data/MVTD/train/ --model /mnt/dataset1/pretrained_fm/unsloth_Qwen2-VL-7B-Instruct/
```
