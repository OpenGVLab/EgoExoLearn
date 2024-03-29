
import torch
import numpy as np
import random


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, feat_sample_rate, label_sample_rate,
                 all_sample_rate, feat_suffix=""):
        self.list_of_examples = list()
        self.num_examples = 0
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.feat_sample_rate = feat_sample_rate
        self.label_sample_rate = label_sample_rate
        self.all_sample_rate = all_sample_rate
        self.feat_suffix = feat_suffix

    def reset(self):
        self.index = 0

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, file_list):
        list_of_videos = []
        if isinstance(file_list, str):
            file_list = [file_list]
        # print("file_list:", file_list)
        # print("self.feat_suffix:", self.feat_suffix)
        assert len(self.feat_suffix) == len(file_list)
        for i, file in enumerate(file_list):
            file_ptr = open(file, 'r')
            list_of_examples = file_ptr.read().strip().split('\n')
            file_ptr.close()
            print(self.feat_suffix[i])
            list_of_examples = [
                dict(vid=x, feat_file=f"{self.features_path}{x.split('.')[0]}{self.feat_suffix[i]}.pt") for
                x in list_of_examples]
            list_of_videos.extend(list_of_examples)
        self.list_of_examples = list_of_videos
        print('there are %d examples in total' % len(self.list_of_examples))
        self.num_examples = len(self.list_of_examples)
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size, flag):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        # for re-loading target data
        if flag == 'target' and self.index == len(self.list_of_examples):
            self.reset()

        batch_input = []
        batch_target = []
        for vid in batch:
            feat_file = vid["feat_file"]
            features = torch.load(feat_file)
            if len(features.shape) == 3:
                features = features.mean(-1)
            features = features.numpy()
            features = features.transpose(1, 0)
            file_ptr = open(self.gt_path + vid["vid"], 'r')
            content = file_ptr.read().split('\n')[:-1]
            content = content[::self.label_sample_rate]
            features = features[:, ::self.feat_sample_rate]

            classes = np.zeros(min(np.shape(features)[1], len(content)))
            features = features[:, :len(classes)]
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            batch_input.append(features[:, ::self.all_sample_rate])
            batch_target.append(classes[::self.all_sample_rate])

        length_of_sequences = list(map(len, batch_target))  # frame#
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)  # if different length, pad w/ zeros
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)  # zero-padding for shorter videos
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask
