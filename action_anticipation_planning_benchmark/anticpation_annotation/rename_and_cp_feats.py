import torch
import pandas as pd


import os
import mmengine



# videometa
videometa_file = mmengine.load("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1.1/anno_convert/renamed_full_list.json")
name_to_uuid_dict = {v["name"].split(".")[0]:v["video_uid"] for v in videometa_file}


def get_uuid_from_original_name(name):
    return name_to_uuid_dict[name]

def rename_and_cp(src,tgt,suffix=".pt"):
    mmengine.mkdir_or_exist(tgt)
    filelist = list(mmengine.list_dir_or_file(src,list_file=True,suffix=suffix,list_dir=False))
    print("filelist length:",len(filelist))
    for file in filelist:
        try:
            video_uid = get_uuid_from_original_name(file.replace(suffix,""))
        except Exception as e:
            print(e)
            print(file)
            continue
        new_filename = video_uid + ".pt"
        try:
            mmengine.copyfile(os.path.join(src,file), os.path.join(tgt,new_filename))
        except Exception as e:
            print(e)
            print(os.path.join(src,file), os.path.join(tgt,new_filename))


# rename_and_cp(src="/mnt/petrelfs/chenguo/data/egolearner/v1_feature_all_ego/clip_b16_25fps_chunk5_center_frame/",
#               tgt="/mnt/petrelfs/share_data/chenguo/egobridge_final_annotations/feats/ego_clip_features_5fps/")

# rename_and_cp(src="/mnt/petrelfs/chenguo/data/egolearner/v1_feature_all_exo/clip_b16_25fps_chunk5_center_frame/",
#               tgt="/mnt/petrelfs/share_data/chenguo/egobridge_final_annotations/feats/exo_clip_features_5fps/")

rename_and_cp(src="/mnt/petrelfs/chenguo/data/egolearner/v1_feature_all_ego_gazed/clip_b16_25fps_chunk5_center_frame/",
              tgt="/mnt/petrelfs/share_data/chenguo/egobridge_final_annotations/feats/ego_gazed_clip_features_5fps/",
              suffix="_50.pt")

print("finish")