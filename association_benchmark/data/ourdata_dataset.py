
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
from .data_utils import video_loader

class OurDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transform=None, is_training=False, tokenizer=None):
        ### common setups ###
        self.ego_root = cfg.ego_root
        self.exo_root = cfg.exo_root
        self.metadata = cfg.metadata
        
        self.clip_length = cfg.clip_length
        self.ctx_length = cfg.ctx_length
        
        ### maybe customized ###
        self.transform = transform
        self.is_training = is_training
        self.tokenizer = tokenizer
        with open(self.metadata, 'r') as f:
            self.samples = json.load(f)
        self.fps = -1
        
    def __len__(self):
        return len(self.samples)
    
    
    def load_singleview_video(self, option, view):
        if view == 'ego':
            frames = self.transform(video_loader(self.ego_root, option['video_uid'], 
                                    float(option['start_sec']), end_second=float(option['end_sec']),
                                    clip_length=self.clip_length, fps=self.fps,
                                    jitter=self.is_training))
        else:
            frames = self.transform(video_loader(self.exo_root, option['video_uid'], 
                                    float(option['start_sec']), end_second=float(option['end_sec']),
                                    clip_length=self.clip_length, fps=self.fps,
                                    jitter=self.is_training))
        return frames
    

    def get_raw_item_v2v(self, i):
        itemMCQ = self.samples[str(i)]
        answerIndex = itemMCQ['answer']
        videoQuery = itemMCQ['query']
        cur_type = itemMCQ['types']
        
        frameQuery = self.load_singleview_video(videoQuery, 'ego' if cur_type == 1 else 'exo')
        textQuery = videoQuery['narration_en']       
    
        frames_options = []
        narration_options = []
        sampleOptions = itemMCQ['choices']
        for option_id in range(len(sampleOptions)):
            option = sampleOptions[str(option_id)]
            frames = self.load_singleview_video(option, 'exo' if cur_type == 1 else 'ego')
            frames_options.append(frames)
            narration_options.append(option['narration_en'])
        
        return frameQuery, textQuery, frames_options, narration_options, answerIndex, itemMCQ['types']

    def __getitem__(self, i):
        frameQuery, textQuery, frames_options, narration_options, answerIndex, q_type = self.get_raw_item_v2v(i)
        second_ids = 0

        raw_textQuery = textQuery
        raw_narration_options = narration_options
    
        frames = frames_options
        return frameQuery, torch.stack(frames, dim=0), answerIndex, q_type, raw_textQuery, raw_narration_options, i
