import torch.utils.data as data

import os
import os.path
import numpy as np
from numpy.random import randint
import torch

from colorama import init
from colorama import Fore, Back, Style

init(autoreset=True)


class VideoRecord(object):

    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def start_sec(self):
        return float(self._data[1])

    @property
    def end_sec(self):
        return float(self._data[2])

    @property
    def labels(self):
        return eval(self._data[3])


class TSNDataSet(data.Dataset):

    def __init__(self,
                 root_path,
                 list_file,
                 feat_path,
                 num_dataload,
                 feat_suffix=".pt",
                 num_classes=11,
                 num_segments=3,
                 new_length=1,
                 modality='RGB',
                 image_tmpl='img_{:05d}.t7',
                 transform=None,
                 force_grayscale=False,
                 random_shift=False,
                 test_mode=False):

        self.root_path = root_path
        self.feat_path = feat_path
        self.feat_suffix = feat_suffix
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.num_dataload = num_dataload
        self.anticipaton_gap_sec = 1
        self.use_context_sec = 2 # here we set input context length to 2 seconds.
        self.context_stride = 1
        self.feature_fps = 5
        self.num_classes = num_classes
        self.feature_dict = dict()
        self.data_type = "anti"

        if self.modality == 'RGBDiff' or self.modality == 'RGBDiff2' or self.modality == 'RGBDiffplus':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()  # read all the video files

    def _parse_list(self):
        self.video_list = [
            VideoRecord(x.strip().split('|')) for x in open(self.list_file)
        ]
        # repeat the list if the length is less than num_dataload (especially for target data)
        n_repeat = self.num_dataload // len(self.video_list)
        n_left = self.num_dataload % len(self.video_list)
        self.video_list = self.video_list * n_repeat + self.video_list[:n_left]

    def get_rec_item(self, index):
        record = self.video_list[index]

        if record.path not in self.feature_dict.keys():
            self.feature_dict[record.path] = torch.load(os.path.join(self.feat_path, record.path + self.feat_suffix))
        feature = self.feature_dict[record.path]
        feature_len = feature.shape[0]

        context_start_sec = record.start_sec
        context_end_sec = record.end_sec

        context_start_frame = int(context_start_sec * self.feature_fps)
        context_end_frame = int(context_end_sec * self.feature_fps)
        context_start_frame = max(0, context_start_frame)
        context_end_frame = min(feature_len - 1, context_end_frame)
        assert context_start_frame < feature_len and context_end_frame < feature_len, "video path: {} feature_len: {} context_start_frame: {} context_end_frame: {} context_start_sec: {} context_end_sec: {}".format(
            record.path, feature_len, context_start_frame, context_end_frame,
            context_start_sec, context_end_sec)
        sample_feature = feature[context_start_frame:context_end_frame +
                                 1:self.context_stride]
        if sample_feature.shape[0] != self.num_segments:
            sample_feature = sample_feature.transpose(0,
                                                      1).unsqueeze(0)  # B C T
            sample_feature = torch.nn.functional.interpolate(
                sample_feature, size=self.num_segments,
                mode='linear').squeeze(0).transpose(0, 1)
        labels = record.labels
        one_hot_labels = torch.nn.functional.one_hot(
            torch.tensor(labels),
            num_classes=self.num_classes).sum(dim=0).float()
        return sample_feature, one_hot_labels

    def get_anti_item(self, index):
        record = self.video_list[index]

        if record.path not in self.feature_dict.keys():
            self.feature_dict[record.path] = torch.load(os.path.join(self.feat_path, record.path + self.feat_suffix))
        feature = self.feature_dict[record.path]
        feature_len = feature.shape[0]
        # print("feature_shape: {}".format(feature.shape))
        context_end_sec = record.start_sec - self.anticipaton_gap_sec

        context_start_sec = context_end_sec - self.use_context_sec
        if context_start_sec < 0:
            return torch.zeros((self.num_segments, 768)), torch.zeros(
                (self.num_classes))
        context_start_frame = int(context_start_sec * self.feature_fps)
        context_end_frame = int(context_end_sec * self.feature_fps)
        assert context_start_frame < feature_len and context_end_frame < feature_len, "video path: {} feature_len: {} context_start_frame: {} context_end_frame: {} context_start_sec: {} context_end_sec: {}".format(
            record.path, feature_len, context_start_frame, context_end_frame,
            context_start_sec, context_end_sec)
        context_start_frame = max(0, context_start_frame)
        context_end_frame = min(feature_len - 1, context_end_frame)

        sample_feature = feature[context_start_frame:context_end_frame +
                                 1:self.context_stride]
        if sample_feature.shape[0] != self.num_segments:
            sample_feature = sample_feature.transpose(0,
                                                      1).unsqueeze(0)  # B C T
            sample_feature = torch.nn.functional.interpolate(
                sample_feature, size=self.num_segments,
                mode='linear').squeeze(0).transpose(0, 1)

        # print(sample_feature.shape)
        # if sample_feature.shape[0] > 20:
        #     print(
        #         "video path: {} feature_len: {} context_start_frame: {} context_end_frame: {} context_start_sec: {} context_end_sec: {}"
        #         .format(record.path, feature_len, context_start_frame,
        #                 context_end_frame, context_start_sec, context_end_sec))
        #     exit()

        labels = record.labels
        one_hot_labels = torch.nn.functional.one_hot(
            torch.tensor(labels),
            num_classes=self.num_classes).sum(dim=0).float()
        return sample_feature, one_hot_labels

    def __getitem__(self, index):
        if self.data_type == "rec":
            return self.get_rec_item(index)
        elif self.data_type == "anti":
            return self.get_anti_item(index)
        else:
            raise ValueError("data_type must be rec or anti")

    def __len__(self):
        return len(self.video_list)
