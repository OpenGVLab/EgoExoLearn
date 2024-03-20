import torch.utils.data as data

import os
import numpy as np
import glob

class FeatureRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path_better(self):
        return self._data[0]

    @property
    def path_worse(self):
        return self._data[1]

class SkillDataSet(data.Dataset):
    def __init__(self, root_path, list_file, ftr_tmpl='{}_{}.npz', action_select='06,13,14,15,18,20', use_exo=False, exo_root_path=None):

        self.root_path = root_path
        self.list_file = list_file
        self.ftr_tmpl = ftr_tmpl
        self.action_select = action_select

        self.use_exo = use_exo
        self.exo_root_path = exo_root_path

        self._parse_list()

    def _load_features(self, vid):
        # addr = os.path.join(self.root_path, self.ftr_tmpl.format(vid,'rgb'))
        addr = os.path.join(self.root_path, vid+'.npz')

        features = np.load(addr)['arr_0'][0].astype(np.float32)
        # print(features.shape)
        # exit()
        return features # 400, 1024

    def _load_features_exo(self, addr):
        # addr = os.path.join(self.root_path, self.ftr_tmpl.format(vid,'rgb'))
        features = np.load(addr)['arr_0'][0].astype(np.float32)
        # print(features.shape)
        # exit()
        return features # 400, 1024

    def _parse_list(self):
        self.pair_list = [FeatureRecord(x.strip().split(' ')) for x in open(self.list_file)]

        if self.action_select == '':
            print('please select action')
            exit()
        if ',' not in self.action_select:
            this_action_select = [int(self.action_select)]
        else:
            this_action_select = [int(x) for x in self.action_select.split(',')]
        

        new_pair_list = []
        for pair in self.pair_list:
            action_num = int(pair._data[0].split('_')[0])
            if action_num in this_action_select:
                new_pair_list.append(pair)
        
        self.pair_list = new_pair_list
        print('action choices:', this_action_select, 'len:', len(self.pair_list))
        # exit()

    def __getitem__(self, index):
        record = self.pair_list[index]
        return self.get(record,index)

    def get(self, record,index):
        vid1 = self._load_features(record.path_better)
        vid2 = self._load_features(record.path_worse)
        # print(index, vid1.shape, vid2.shape)
        # if vid1.shape[0] != 10:
        #     print('vid1.shape[0] != 10', self.pair_list[index].path_better)
        #     exit()
        # if vid2.shape[0] != 10:
        #     print('vid2.shape[0] != 10', self.pair_list[index].path_worse)
        #     exit()
        if self.use_exo:
            this_action = record.path_better.split('_')[0]
            assert this_action in ['06','13','14','15','18','20']
            exo_folder_name = this_action if this_action in ['06','18','20'] else '131415'
            exo_paths = glob.glob('%s/%s/*.npz'%(self.exo_root_path, exo_folder_name))
            exo_paths.sort()
            exo_path = np.random.choice(exo_paths)
            vid_exo = self._load_features_exo(exo_path)
            return vid1, vid2, vid_exo
        else:
            return vid1, vid2

    def __len__(self):
        return len(self.pair_list)


class FeatureRecordSingle(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

class SkillDataSetSingle(SkillDataSet):
    def _parse_list(self):
        self.pair_list = [FeatureRecordSingle(x.strip().split(' ')) for x in open(self.list_file)]


    def get(self, record):
        vid = self._load_features(record.path)

        name = record.path.split('/')[-1]
        return name, vid
