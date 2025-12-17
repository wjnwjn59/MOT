import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.hiptrack.hiptrack_cls import build_hiptrack_cls
# forward propagation related
from lib.train.actors.hiptrack_cls import HIPTrackClsActor
# for import modules
import importlib

from ..utils.focal_loss import FocalLoss
from ..utils.cls_loss import build_cls_loss


def run(settings):
    settings.description = 'Training script for HIPTrack with Classification Branch'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config_cls" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders (modified to use Got10kCls dataset)
    loader_train, loader_val = build_dataloaders_cls(cfg, settings)

    # Create network with classification head
    net = build_hiptrack_cls(cfg, training=True)

    # wrap networks to distributed one
    net.cuda()

    # Two-stage training support (SimTrackMod strategy)
    if getattr(cfg.TRAIN, 'TWO_STAGE', False):
        print("=" * 80)
        print("TWO-STAGE TRAINING ENABLED (SimTrackMod Architecture)")
        print("=" * 80)
        print(f"Stage 1: Freeze backbone, train classification head only ({cfg.TRAIN.STAGE1_EPOCHS} epochs, LR={cfg.TRAIN.STAGE1_LR})")
        print(f"Stage 2: Fine-tune entire model ({cfg.TRAIN.STAGE2_EPOCHS} epochs, LR={cfg.TRAIN.STAGE2_LR})")
        print("=" * 80)
        
        # Determine current stage based on checkpoint or start fresh
        current_epoch = 0  # Will be updated if resuming from checkpoint
        stage1_end_epoch = cfg.TRAIN.STAGE1_EPOCHS
        total_epochs = cfg.TRAIN.STAGE1_EPOCHS + cfg.TRAIN.STAGE2_EPOCHS
        
        # Stage 1: Freeze backbone and bottleneck
        if current_epoch < stage1_end_epoch:
            print(f"\n[Stage 1] Freezing backbone and bottleneck for epochs 0-{stage1_end_epoch}")
            for k, v in net.named_parameters():
                if 'backbone' in k or 'bottleneck' in k:
                    v.requires_grad = False
                    print(f"  Frozen: {k}")
            
            # Update learning rate for stage 1
            cfg.TRAIN.LR = cfg.TRAIN.STAGE1_LR
            cfg.TRAIN.EPOCH = stage1_end_epoch
        else:
            # Stage 2: Unfreeze everything
            print(f"\n[Stage 2] Unfreezing all parameters for epochs {stage1_end_epoch}-{total_epochs}")
            for k, v in net.named_parameters():
                v.requires_grad = True
                if 'backbone' in k or 'bottleneck' in k:
                    print(f"  Unfrozen: {k}")
            
            # Update learning rate for stage 2
            cfg.TRAIN.LR = cfg.TRAIN.STAGE2_LR
            cfg.TRAIN.EPOCH = total_epochs
    else:
        # Standard training: Freeze backbone parameters (original behavior)
        for k, v in net.named_parameters():
            if 'backbone' in k:
                v.requires_grad = False

    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    
    # Loss functions and Actors
    focal_loss = FocalLoss()
    cls_loss = build_cls_loss(cfg)
    
    objective = {
        'giou': giou_loss, 
        'l1': l1_loss, 
        'focal': focal_loss, 
        'cls': cls_loss
    }
    loss_weight = {
        'giou': cfg.TRAIN.GIOU_WEIGHT, 
        'l1': cfg.TRAIN.L1_WEIGHT, 
        'focal': 1.0, 
        'cls': cfg.TRAIN.CLS_WEIGHT
    }
    
    actor = HIPTrackClsActor(
        net=net, 
        objective=objective, 
        loss_weight=loss_weight, 
        settings=settings, 
        cfg=cfg, 
        multiFrame=False
    )

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)


def build_dataloaders_cls(cfg, settings):
    """
    Build dataloaders with classification support
    Modified version of build_dataloaders that uses Got10kCls dataset
    """
    # This is a placeholder - you'll need to check how base_functions.build_dataloaders works
    # and modify it to use Got10kCls instead of Got10k
    
    from lib.train.dataset import Lasot, Got10kCls, MSCOCOSeq, ImagenetVID, TrackingNet
    from lib.train.data import sampler, opencv_loader, processing, LTRLoader
    import lib.train.data.transforms as tfm
    from lib.utils.misc import is_main_process
    
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                      output_sz=output_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      transform=transform_train,
                                                      joint_transform=transform_joint,
                                                      settings=settings,
                                                      num_prev=0)

    data_processing_val = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                    output_sz=output_sz,
                                                    center_jitter_factor=settings.center_jitter_factor,
                                                    scale_jitter_factor=settings.scale_jitter_factor,
                                                    mode='sequence',
                                                    transform=transform_val,
                                                    joint_transform=transform_joint,
                                                    settings=settings,
                                                    num_prev=0)

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = False  # For template, we don't need classification
    dataset_train = sampler.TrackingSampler(datasets=names2datasets_cls(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader, cls_ann_dir=cfg.DATA.TRAIN.CLS_ANN_DIR),
                                           p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                           samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                           max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                           num_template_frames=settings.num_template, processing=data_processing_train,
                                           frame_sample_mode=sampler_mode, train_cls=train_cls)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                            num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    # Validation samplers and loaders
    dataset_val = sampler.TrackingSampler(datasets=names2datasets_cls(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader, cls_ann_dir=cfg.DATA.VAL.CLS_ANN_DIR),
                                         p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                         samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                         max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                         num_template_frames=settings.num_template, processing=data_processing_val,
                                         frame_sample_mode=sampler_mode, train_cls=train_cls)
    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                          num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                          epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val


def names2datasets_cls(name_list, settings, image_loader, cls_ann_dir=None):
    """
    Convert dataset names to dataset objects with classification support
    """
    from lib.train.dataset import Lasot, Got10kCls, MSCOCOSeq, ImagenetVID, TrackingNet
    from lib.train.admin import env_settings
    
    env = env_settings()
    datasets = []
    
    for name in name_list:
        if name == "LASOT":
            datasets.append(Lasot(env.lasot_dir, split='train', image_loader=image_loader))
        elif name == "GOT10K_vottrain":
            datasets.append(Got10kCls(split='vottrain', image_loader=image_loader, cls_ann_dir=cls_ann_dir))
        elif name == "GOT10K_votval":
            datasets.append(Got10kCls(split='votval', image_loader=image_loader, cls_ann_dir=cls_ann_dir))
        elif name == "GOT10K_train_full":
            datasets.append(Got10kCls(split='train_full', image_loader=image_loader, cls_ann_dir=cls_ann_dir))
        elif name == "COCO17":
            datasets.append(MSCOCOSeq(env.coco_dir, version="2017", image_loader=image_loader))
        elif name == "VID":
            datasets.append(ImagenetVID(env.imagenet_dir, image_loader=image_loader))
        elif name == "TRACKINGNET":
            datasets.append(TrackingNet(env.trackingnet_dir, image_loader=image_loader))
        else:
            raise ValueError(f"Unknown dataset name: {name}")
    
    return datasets

