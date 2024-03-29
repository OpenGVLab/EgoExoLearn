import torch
import pandas as pd


import os
import mmengine



# videometa
videometa_file = mmengine.load("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1.1/anno_convert/renamed_full_list.json")



def get_uuid_from_original_name(name):
    name_to_uuid_dict = {v["name"].split(".")[0]:v["video_uid"] for v in videometa_file}
    # print(name_to_uuid_dict)
    return name_to_uuid_dict[name]


def rename_anno(file,new_file):
    encoding = "gb18030"
    anno = pd.read_csv(file, encoding=encoding)
    anno["video_id"] = anno["video_id"].apply(get_uuid_from_original_name)
    os.makedirs(os.path.dirname(new_file),exist_ok=True)
    anno.to_csv(new_file,index=False,encoding=encoding)
    
    
rename_anno("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1/aa_new_clip_anno/gen_aa_anno_list/gen/ego_test_valid.csv",
            "./balanced_full_annotation/ego_test_valid.csv")


rename_anno("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1/aa_new_clip_anno/gen_aa_anno_list/gen/ego_val_valid.csv",
            "./balanced_full_annotation/ego_val_valid.csv")


rename_anno("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1/aa_new_clip_anno/gen_aa_anno_list/gen/ego_train_valid.csv",
            "./balanced_full_annotation/ego_train_valid.csv")

rename_anno("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1/aa_new_clip_anno/gen_aa_anno_list/gen/exo_test_valid.csv",
            "./balanced_full_annotation/exo_test_valid.csv")


rename_anno("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1/aa_new_clip_anno/gen_aa_anno_list/gen/exo_val_valid.csv",
            "./balanced_full_annotation/exo_val_valid.csv")


rename_anno("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1/aa_new_clip_anno/gen_aa_anno_list/gen/exo_train_valid.csv",
            "./balanced_full_annotation/exo_train_valid.csv")
