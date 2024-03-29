import os
import mmengine



# videometa
videometa_file = mmengine.load("/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1.1/anno_convert/renamed_full_list.json")



def get_uuid_from_original_name(name):
    name_to_uuid_dict = {v["name"].split(".")[0]:v["video_uid"] for v in videometa_file}
    # print(name_to_uuid_dict)
    return name_to_uuid_dict[name]


input_dir = "/mnt/lustre/chenguo/petrelfs/workspace/EgoLearner/v1/as/generated_data/as_gts_fps25"
output_dir = "./gts_fps25"


files = os.listdir(input_dir)

for file in files:
    target_file = get_uuid_from_original_name(file.split(".")[0])
    file = os.path.join(input_dir,file)
    target_file = os.path.join(output_dir,target_file+".txt")
    print(file, target_file)
    # copy file
    os.system(f"cp {file} {target_file}")
