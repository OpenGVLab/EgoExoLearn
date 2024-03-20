import csv
import glob
import json
import numpy as np
import os.path as osp
import pickle
import random

import decord
import pandas as pd
import torch
from ipdb import set_trace
from decord import cpu
import cv2
import io,os
from numpy.random import default_rng
rng = default_rng()

from .data_utils import video_loader


class OurTrainDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transform, tokenizer, is_training=True, 
    ):
        ### for data loading ###
        self.cfg = cfg
        self.dataset = cfg.dataset
        self.ego_root = cfg.ego_root
        self.ego_metadata = cfg.ego_metadata
        self.exo_root = cfg.exo_root
        self.exo_metadata = cfg.exo_metadata
        ### for sampling and transforming ###
        self.transform = transform
        self.is_training = is_training
        self.tokenizer = tokenizer
        self.clip_length = cfg.clip_length

        ### metadata preparation ###
        self.param_dict = {
            'root': {
                0: self.ego_root, 
                1: self.exo_root
            },
            'fps': {
                0: -1, 
                1: -1
            },
        }    
        
        assert self.dataset in ['ourdata_ego', 'ourdata_exo', 'ourdata_egoexo']
        self.ego_samples = pd.read_csv(self.ego_metadata)
        self.exo_samples = pd.read_csv(self.exo_metadata)

        self.ego_number = len(self.ego_samples)
        ### merge two datasets ###
        self.samples = {}
        if self.dataset == 'ourdata_ego':
            self.samples = {0: self.ego_samples}
        elif self.dataset == 'ourdata_exo':
            self.samples = {1: self.exo_samples}
        elif self.dataset == 'ourdata_egoexo':
            self.samples = {
                0: self.ego_samples,
                1: self.exo_samples,
            }
        print('Done init dataset')

    def __len__(self):
        ego_len = len(self.samples[0]) if 0 in self.samples else 0
        exo_len = len(self.samples[1]) if 1 in self.samples else 0
        return ego_len + exo_len

    
    def load_metadata(self, id_offset, egoexo_flag):
        data = self.samples[egoexo_flag].iloc[id_offset]
        vid = data['video_uid']
        start_second, end_second, narration = data['start_sec'], data['end_sec'], data['narration_en']
        uid = vid if 'uid' not in data else data['uid']
        return vid, uid, start_second, end_second, narration
    
    def load_video(self, root, vid, start_second, end_second, egoexo_flag):
        frames = video_loader(root=root, vid=vid, second=start_second, end_second=end_second,
                                fps=self.param_dict['fps'][egoexo_flag], clip_length=self.clip_length, jitter=self.is_training)
        
        if self.transform is not None:
            frames = self.transform(frames)
        return frames

    def __getitem__(self, i):
        ### set an indicator, 0 for ego, 1 for exo ###
        if self.dataset == 'ourdata_egoexo':
            if i < self.ego_number:
                egoexo_flag = 0
                id_offset = i
            else:
                egoexo_flag = 1
                id_offset = i - self.ego_number
        elif self.dataset == 'ourdata_ego':
            egoexo_flag = 0
            id_offset = i
        elif self.dataset == 'ourdata_exo':
            egoexo_flag = 1
            id_offset = i

        ret_info = {}
        vid, uid, start_second, end_second, narration = self.load_metadata(id_offset, egoexo_flag)
            
        frames = self.load_video(self.param_dict['root'][egoexo_flag], vid, start_second, end_second, egoexo_flag)
                
        if self.tokenizer is not None:
            caption = self.tokenizer(narration)


        ret_info['uid'] = uid
        ret_info['vid'] = vid
        ret_info['video'] = frames
        ret_info['text'] = caption        
        ret_info['raw_caption'] = narration
        return ret_info
