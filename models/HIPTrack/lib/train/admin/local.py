class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/vli/thangdd_workspace/MOT/models/HIPTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/vli/thangdd_workspace/MOT/models/HIPTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/vli/thangdd_workspace/MOT/models/HIPTrack/pretrained_networks'
        self.lasot_dir = '/home/vli/thangdd_workspace/MOT/data/lasot'
        self.got10k_dir = '/home/vli/thangdd_workspace/MOT/data/train'
        self.got10k_val_dir = '/home/vli/thangdd_workspace/MOT/data/val'
        self.lasot_lmdb_dir = '/home/vli/thangdd_workspace/MOT/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/vli/thangdd_workspace/MOT/data/got10k_lmdb'
        self.trackingnet_dir = '/home/vli/thangdd_workspace/MOT/data/trackingnet'
        self.trackingnet_lmdb_dir = '/home/vli/thangdd_workspace/MOT/data/trackingnet_lmdb'
        self.coco_dir = '/home/vli/thangdd_workspace/MOT/data/coco'
        self.coco_lmdb_dir = '/home/vli/thangdd_workspace/MOT/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/vli/thangdd_workspace/MOT/data/vid'
        self.imagenet_lmdb_dir = '/home/vli/thangdd_workspace/MOT/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
