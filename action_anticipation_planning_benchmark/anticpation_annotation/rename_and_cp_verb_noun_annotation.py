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


def rename_anno(src_anno_dir,tgt_anno_dir,suffix=".pt",file_prefix=""):
    mmengine.mkdir_or_exist(tgt_anno_dir)
    new_lines=  []
    for file in mmengine.list_dir_or_file(src_anno_dir):
        anno_file = os.path.join(src_anno_dir,file)
        lines = open(anno_file,'r').read().strip().split("\n")
        for line in lines:
            old_name,a,b,c = line.split("|")
            old_name = os.path.basename(old_name)
            old_name = old_name.replace("_50.pt","")
            old_name = old_name.replace(".pt","")
            new_name = get_uuid_from_original_name(old_name)
            new_name = os.path.join(file_prefix,new_name+".pt")
            new_line = f"{new_name}|{a}|{b}|{c}"
            new_lines.append(new_line)
        lines = "\n".join(new_lines)
        with open(os.path.join(tgt_anno_dir,file),"w") as f:
            f.write(lines)

rename_anno("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1/aa_new_clip_anno/gen_aa_anno_list/ta3n_list_generation/gen/ta3n/noun",
            "./noun",
            suffix=".pt",
            file_prefix="")
            
    
rename_anno("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1/aa_new_clip_anno/gen_aa_anno_list/ta3n_list_generation/gen/ta3n/verb",
            "./verb",
            suffix=".pt",
            file_prefix="")


rename_anno("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1/aa_new_clip_anno/gen_aa_anno_list/ta3n_list_generation/gen/ta3n/noun_gazed",
            "./noun_gazed",
            suffix="_50.pt",
            file_prefix="")
            
    
rename_anno("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1/aa_new_clip_anno/gen_aa_anno_list/ta3n_list_generation/gen/ta3n/verb_gazed",
            "./verb_gazed",
            suffix="_50.pt",
            file_prefix="")