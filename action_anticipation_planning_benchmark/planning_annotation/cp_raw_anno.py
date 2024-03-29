import mmengine
import os
# videometa
videometa_file = mmengine.load("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1.1/anno_convert/renamed_full_list.json")



def get_uuid_from_original_name(name):
    name_to_uuid_dict = {v["name"].split(".")[0]:v["video_uid"] for v in videometa_file}
    # print(name_to_uuid_dict)
    return name_to_uuid_dict[name]



def p(file,suffix="",output_file=None):
    with open(file,"r") as f:
        lines = f.read().strip().split("\n")
    # print(lines)
    i=0
    new_lines = []
    for line in lines:
        terms = line.split("|")
        name = os.path.basename(terms[0]).split(".")[0]
        name = name.replace(suffix,"")
        video_uid = get_uuid_from_original_name(name)
        terms[0] = video_uid
        new_line = "|".join(terms)
        new_lines.append(new_line)

    with open(output_file,"w") as f:
        f.write("\n".join(new_lines))
        
    
# ego
p("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1/aa_new_clip_anno/gen_lta_anno_list/gen/cotrain_ego_exo_anno_list.txt",
  suffix="_320p_25fps",
  output_file="./subsets/cotrain_ego_exo_anno_list.txt")


p("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1/aa_new_clip_anno/gen_lta_anno_list/gen/cotrain_ego_gaze_exo_anno_list.txt",
  suffix="_50",
  output_file="./subsets/cotrain_ego_gaze_exo_anno_list.txt")


p("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1/aa_new_clip_anno/gen_lta_anno_list/gen/train_ego_anno_list.txt",
  suffix="_320p_25fps",
  output_file="./subsets/train_ego_anno_list.txt")

p("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1/aa_new_clip_anno/gen_lta_anno_list/gen/train_ego_gaze_anno_list.txt",
  suffix="_50",
  output_file="./subsets/train_ego_gaze_anno_list.txt")
    
p("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1/aa_new_clip_anno/gen_lta_anno_list/gen/val_ego_anno_list.txt",
  suffix="_320p_25fps",
  output_file="./subsets/val_ego_anno_list.txt")

p("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1/aa_new_clip_anno/gen_lta_anno_list/gen/val_ego_gaze_anno_list.txt",
  suffix="_50",
  output_file="./subsets/val_ego_gaze_anno_list.txt")


p("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1/aa_new_clip_anno/gen_lta_anno_list/gen/test_ego_anno_list.txt",
  suffix="_320p_25fps",
  output_file="./subsets/test_ego_anno_list.txt")

p("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1/aa_new_clip_anno/gen_lta_anno_list/gen/test_ego_gaze_anno_list.txt",
  suffix="_50",
  output_file="./subsets/test_ego_gaze_anno_list.txt")



# exo
p("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1/aa_new_clip_anno/gen_lta_anno_list/gen/train_exo_anno_list.txt",
  suffix="",
  output_file="./subsets/train_exo_anno_list.txt")

p("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1/aa_new_clip_anno/gen_lta_anno_list/gen/val_exo_anno_list.txt",
  suffix="",
  output_file="./subsets/val_exo_anno_list.txt")

p("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1/aa_new_clip_anno/gen_lta_anno_list/gen/test_exo_anno_list.txt",
  suffix="",
  output_file="./subsets/test_exo_anno_list.txt")