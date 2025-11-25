import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
import json
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class Got10kCls(BaseVideoDataset):
    """ GOT-10k dataset with classification labels for maritime tracking challenges.

    This extends the base GOT-10k dataset to include frame-level classification labels
    for various tracking challenges like occlusion, illumination change, motion blur, etc.
    
    Classification labels are loaded from JSONL files in the annotation directory.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, 
                 data_fraction=None, cls_ann_dir=None):
        """
        args:
            root - path to the got-10k training data. Note: This should point to the 'train' folder inside GOT-10k
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'val'. Note: The validation split here is a subset of the official got-10k train split,
                    not NOT the official got-10k validation split. To use the official validation split, provide that as
                    the root folder instead.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
            cls_ann_dir - Directory containing classification annotation JSONL files
        """
        root = env_settings().got10k_dir if root is None else root
        super().__init__('GOT10k', root, image_loader)

        # all folders inside the root
        self.sequence_list = self._get_sequence_list()

        # seq_id is the index of the folder inside the got10k root path
        if split is not None:
            if seq_ids is not None:
                raise ValueError('Cannot set both split_name and seq_ids.')
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'got10k_train_split.txt')
            elif split == 'val':
                file_path = os.path.join(ltr_path, 'data_specs', 'got10k_val_split.txt')
            elif split == 'train_full':
                file_path = os.path.join(ltr_path, 'data_specs', 'got10k_train_full_split.txt')
            elif split == 'vottrain':
                file_path = os.path.join(ltr_path, 'data_specs', 'got10k_vot_train_split.txt')
            elif split == 'votval':
                file_path = os.path.join(ltr_path, 'data_specs', 'got10k_vot_val_split.txt')
            else:
                raise ValueError('Unknown split name.')
            seq_ids = pandas.read_csv(file_path, header=None, dtype=str).squeeze("columns").values.tolist()
        elif seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))

        self.sequence_list = [seq for seq in self.sequence_list if seq in seq_ids]

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.sequence_meta_info = self._load_meta_info()
        self.seq_per_class = self._build_seq_per_class()

        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()
        
        # Load classification annotations
        self.cls_ann_dir = cls_ann_dir
        self.cls_annotations = {}
        self._load_cls_annotations()

        # Define class mapping (based on the maritime tracking challenges)
        # Classes: ["Occlusion", "Illumination Change", "Scale Variation", "Motion Blur", 
        #           "Variance in Appearance", "Partial Visibility", "Low Resolution", 
        #           "Background Clutter", "Low Contrast Object", "Normal"]
        self.class_names = [
            "Occlusion", "Illumination Change", "Scale Variation", "Motion Blur",
            "Variance in Appearance", "Partial Visibility", "Low Resolution",
            "Background Clutter", "Low Contrast Object", "Normal"
        ]
        self.num_classes = len(self.class_names)
        print(f"Classification classes ({self.num_classes}): {self.class_names}")

    def _load_cls_annotations(self):
        """Load classification annotations from JSONL files"""
        if not os.path.exists(self.cls_ann_dir):
            print(f"Warning: Classification annotation directory not found: {self.cls_ann_dir}")
            return
        
        for seq_name in self.sequence_list:
            # Try to find the JSONL file for this sequence
            # Format: /path/to/annts/1-Ship/1-Ship.jsonl
            seq_ann_path = os.path.join(self.cls_ann_dir, seq_name, f"{seq_name}.jsonl")
            
            if os.path.exists(seq_ann_path):
                # Load JSONL file
                seq_annotations = []
                try:
                    with open(seq_ann_path, 'r') as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                seq_annotations.append(data)
                    self.cls_annotations[seq_name] = seq_annotations
                except Exception as e:
                    print(f"Error loading classification annotations for {seq_name}: {e}")
            else:
                # No annotation file found for this sequence
                pass
        
        print(f"Loaded classification annotations for {len(self.cls_annotations)} / {len(self.sequence_list)} sequences")

    def get_class_label(self, seq_name, frame_id):
        """
        Get classification label for a specific frame in a sequence
        
        Args:
            seq_name: Sequence name (e.g., '1-Ship')
            frame_id: Frame index (0-based)
        Returns:
            class_idx: Integer class index (0-9), or -100 if not found (ignore index)
        """
        if seq_name not in self.cls_annotations:
            return -100  # Ignore index for sequences without annotations
        
        seq_annots = self.cls_annotations[seq_name]
        
        # Find annotation for this frame (frame_id in JSONL is 1-based)
        target_frame_id = frame_id + 1
        
        for annot in seq_annots:
            if annot.get('frame_id') == target_frame_id:
                # Determine the class based on the annotation
                return self._determine_class_from_annotation(annot)
        
        # If no annotation found for this frame, return ignore index
        return -100
    
    def _determine_class_from_annotation(self, annot):
        """
        Determine the classification label based on annotation data
        
        Priority:
        1. Check VLM responses first - if ANY flag is 1, use that class
        2. If no VLM response, check CV responses - if multiple are 1, pick randomly
        3. Default to Normal if nothing is flagged
        """
        cv_resp = annot.get('cv_response', {})
        vlm_resp = annot.get('vlm_response', {})
        
        # VLM response mapping (priority: check all, use first one that's flagged)
        vlm_mapping = {
            'occlusion': 0,           # Occlusion
            'motion_blur': 3,          # Motion Blur
            'illu_change': 1,          # Illumination Change
            'partial_visibility': 5,   # Partial Visibility
            'variance_appear': 4,      # Variance in Appearance
            'background_clutter': 7    # Background Clutter
        }
        
        # Check VLM responses - use first one that's flagged
        for vlm_key, class_idx in vlm_mapping.items():
            if vlm_resp.get(vlm_key, {}).get('flag', 0) == 1:
                return class_idx
        
        # No VLM response found, check CV responses
        cv_mapping = {
            'scale_variation': 2,  # Scale Variation
            'low_res': 6,          # Low Resolution
            'low_contrast': 8      # Low Contrast Object
        }
        
        # Collect all CV responses that are flagged
        flagged_cv = []
        for cv_key, class_idx in cv_mapping.items():
            if cv_resp.get(cv_key, 0) == 1:
                flagged_cv.append(class_idx)
        
        # If we have CV responses, pick one randomly
        if len(flagged_cv) > 0:
            return random.choice(flagged_cv)
        
        # Default: Normal (no challenges detected)
        return 9  # Normal

    def get_name(self):
        return 'got10k'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info

    def _read_meta(self, seq_path):
        try:
            with open(os.path.join(seq_path, 'meta_info.ini')) as f:
                meta_info = f.readlines()
            object_meta = OrderedDict({'object_class_name': meta_info[5].split(': ')[-1][:-1],
                                       'motion_class': meta_info[6].split(': ')[-1][:-1],
                                       'major_class': meta_info[7].split(': ')[-1][:-1],
                                       'root_class': meta_info[8].split(': ')[-1][:-1],
                                       'motion_adverb': meta_info[9].split(': ')[-1][:-1]})
        except:
            object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta

    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self):
        with open(os.path.join(self.root, 'list.txt')) as f:
            dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "absence.label")
        cover_file = os.path.join(seq_path, "cover.label")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
        with open(cover_file, 'r', newline='') as f:
            cover = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])

        target_visible = ~occlusion & (cover>0).byte()

        visible_ratio = cover.float() / 8
        return target_visible, visible_ratio

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible, visible_ratio = self._read_target_visible(seq_path)
        visible = visible & valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:08}.jpg'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        return obj_meta['object_class_name']

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        seq_name = self.sequence_list[seq_id]
        obj_meta = self.sequence_meta_info[seq_name]

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
        
        # Add classification labels for each frame
        cls_labels = [self.get_class_label(seq_name, f_id) for f_id in frame_ids]
        anno_frames['cls_labels'] = cls_labels

        return frame_list, anno_frames, obj_meta

