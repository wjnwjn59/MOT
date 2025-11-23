from lib.test.tracker.hiptrack import HIPTrack
from lib.models.hiptrack.hiptrack_cls import build_hiptrack_cls
import torch


class HIPTrackCls(HIPTrack):
    """
    Tracker for HIPTrack with classification branch
    Classification branch is disabled during inference
    """
    
    def __init__(self, params, dataset_name, visualize_during_infer=False):
        # Build network first (before calling super().__init__)
        network = build_hiptrack_cls(params.cfg, training=False)
        
        # Load checkpoint
        checkpoint = torch.load(params.checkpoint, map_location='cpu')
        network.load_state_dict(checkpoint['net'], strict=False)
        
        # Disable classification branch
        network.disable_cls_branch()
        
        # Set up the network
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.network.set_eval()
        
        # Initialize other components (copied from parent)
        from lib.test.tracker.data_utils import Preprocessor
        from lib.test.utils.hann import hann2d
        
        self.params = params
        self.preprocessor = Preprocessor()
        self.state = None
        self.visualize_during_infer = visualize_during_infer
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        
        # motion constrain
        self.output_window = hann2d(
            torch.tensor([self.feat_sz, self.feat_sz]).long(), 
            centered=True
        ).cuda()
        
        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                import os
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                self._init_visdom(None, 1)
        
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
    
    # All other methods (initialize, track, etc.) are inherited from HIPTrack parent class
    # forward_track() will automatically use the disabled classification branch


def get_tracker_class():
    return HIPTrackCls

