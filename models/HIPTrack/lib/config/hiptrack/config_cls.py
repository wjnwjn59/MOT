from easydict import EasyDict as edict
import yaml
from lib.config.hiptrack.config import cfg as base_cfg

"""
Config for HIPTrack with Classification Branch
"""

# Copy base config
cfg = edict(base_cfg.copy())

# MODEL - CLASSIFICATION HEAD
cfg.MODEL.CLS_HEAD = edict()
cfg.MODEL.CLS_HEAD.NUM_CLASSES = 10  # Number of classification classes (maritime tracking challenges)
cfg.MODEL.CLS_HEAD.HIDDEN_DIM = 512  # Hidden dimension for classifier
cfg.MODEL.CLS_HEAD.POOLING = 'avg'  # Pooling method: 'avg', 'max', 'attention'
cfg.MODEL.CLS_HEAD.DROPOUT = 0.3  # Dropout rate

# MODEL - Add hidden dim if not exists
if not hasattr(cfg.MODEL, 'HIDDEN_DIM'):
    cfg.MODEL.HIDDEN_DIM = 768  # For vit_base

# TRAINING - Classification
cfg.TRAIN.CLS_WEIGHT = 1.0  # Weight for classification loss
cfg.TRAIN.CLS_LOSS_TYPE = "CE"  # Loss type: CE, FOCAL, LABEL_SMOOTH
cfg.TRAIN.CLS_FOCAL_ALPHA = 0.25  # Alpha for focal loss
cfg.TRAIN.CLS_FOCAL_GAMMA = 2.0  # Gamma for focal loss
cfg.TRAIN.CLS_LABEL_SMOOTHING = 0.1  # Label smoothing factor
cfg.TRAIN.FREEZE_CLS_EPOCH = -1  # Epoch to freeze classification branch (-1 = never)

# DATA - Classification annotations
cfg.DATA.TRAIN.CLS_ANN_DIR = "/home/thinhnp/MOT/data/train_maritime_env_clf_annts"  # Classification annotation directory
cfg.DATA.VAL.CLS_ANN_DIR = "/home/thinhnp/MOT/data/train_maritime_env_clf_annts"  # Validation annotation directory


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                # Allow new keys for classification
                base_cfg[k] = v
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)

