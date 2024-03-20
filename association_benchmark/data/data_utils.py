
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
from decord import cpu
import cv2
import io,os
import argparse

client = None

def get_vr(video_path):
    ### Now, we simply use decord videoReader ###
    if client is None:
        vreader = decord.VideoReader(video_path, ctx=cpu(0))
    else:
        video_bytes = client.get(video_path, enable_stream=True)
        assert video_bytes is not None, "Get video failed from {}".format(video_path)
        video_path = video_bytes
        if isinstance(video_path, bytes):
            video_path = io.BytesIO(video_bytes)
        vreader = decord.VideoReader(video_path, ctx=cpu(0))
        
    return vreader
    
def video_loader(root, vid, second=None, end_second=None, fps=30, clip_length=32, jitter=False):
    '''
    args:
        root: root directory of the video
        vid: the unique vid of the video, e.g. hello.mp4 
        second: the start second of the clip/video
        end_second: the end second of the clip/video
        fps: specify the decoding fps of the video
        clip_length: the number of frames
        jitter: True stands for random sampling, False means center sampling
    return:
        frames: torch tensor with shape: [T, H, W, C]
    '''
    ### get vr ###
    if not vid.endswith('.mp4'):
        vid = vid + '.mp4'
    vr = get_vr(osp.join(root, vid))
    fps = vr.get_avg_fps() if fps == -1 else fps
    
    ### add a sanity check ###
    second = min(second, len(vr) / vr.get_avg_fps())
    second_offset = second
    end_second = min(end_second, len(vr) / vr.get_avg_fps())
    end_second = max(second + 1, end_second)
    
    ### calculate frame_ids ###
    frame_offset = int(np.round(second_offset * fps))
    total_duration = max(int((end_second - second) * fps), clip_length)
    frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)), num_segments=clip_length, jitter=jitter)
    
    ### load frames ###
    try:
        frames = vr.get_batch(frame_ids).asnumpy()
    except decord.DECORDError as error:
        print(error)
        frames = vr.get_batch([0] * len(frame_ids)).asnumpy()

    frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
    return torch.stack(frames, dim=0)

def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    '''
    args:
        start_frame: the beginning frame indice
        end_frame: the end frame indice
        num_segment: number of frames to be sampled
        jitter: True stands for random sampling, False means center sampling
    return:
        seq: a list for the sampled frame indices 
    '''
    assert start_frame <= end_frame
    seg_size = float(end_frame - start_frame - 1) / num_segments
    seq = []
    for i in range(num_segments):
        start = int(np.round(seg_size * i) + start_frame)
        end = int(np.round(seg_size * (i + 1)) + start_frame)
        
        ### added here to avoid out-of-boundary of frame_id, as np.random.randint ###
        start = min(start, end_frame-1)
        end = min(end, end_frame)

        if jitter:
            frame_id = np.random.randint(low=start, high=(end + 1))
        else:
            frame_id = (start + end) // 2

        seq.append(frame_id)
    return seq

