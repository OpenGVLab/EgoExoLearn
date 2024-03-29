import torch
from model import MultiStageModel
from egolearner_train import Trainer
from egolearner_predict import *
from egolearner_batch_gen import BatchGenerator
import os
import argparse
import random
import time
import warnings
from egobridge_settings import get_annotations_from_settings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
# architecture
parser.add_argument('--num_stages', default=4, type=int, help='stage number')
parser.add_argument('--num_layers',
                    default=10,
                    type=int,
                    help='layer number in each stage')
parser.add_argument('--num_f_maps',
                    default=64,
                    type=int,
                    help='embedded feat. dim.')
parser.add_argument('--features_dim',
                    default=2048,
                    type=int,
                    help='input feat. dim.')
parser.add_argument('--DA_adv',
                    default='none',
                    type=str,
                    help='adversarial loss (none | rev_grad)')
parser.add_argument(
    '--DA_adv_video',
    default='none',
    type=str,
    help=
    'video-level adversarial loss (none | rev_grad | rev_grad_ssl | rev_grad_ssl_2)'
)
parser.add_argument('--pair_ssl',
                    default='all',
                    type=str,
                    help='pair-feature methods for SSL-DA (all | adjacent)')
parser.add_argument('--num_seg',
                    default=10,
                    type=int,
                    help='segment number for each video')
parser.add_argument('--place_adv',
                    default=['N', 'Y', 'Y', 'N'],
                    type=str,
                    nargs="+",
                    metavar='N',
                    help='len(place_adv) == num_stages')
parser.add_argument('--multi_adv',
                    default=['N', 'N'],
                    type=str,
                    nargs="+",
                    metavar='N',
                    help='separate weights for domain discriminators')
parser.add_argument(
    '--weighted_domain_loss',
    default='Y',
    type=str,
    help='weighted domain loss for class-wise domain discriminators')
parser.add_argument('--ps_lb',
                    default='soft',
                    type=str,
                    help='pseudo-label type (soft | hard)')
parser.add_argument(
    '--source_lb_weight',
    default='pseudo',
    type=str,
    help='label type for source data weighting (real | pseudo)')
parser.add_argument('--method_centroid',
                    default='none',
                    type=str,
                    help='method to get centroids (none | prob_hard)')
parser.add_argument('--DA_sem',
                    default='mse',
                    type=str,
                    help='metric for semantic loss (none | mse)')
parser.add_argument('--place_sem',
                    default=['N', 'Y', 'Y', 'N'],
                    type=str,
                    nargs="+",
                    metavar='N',
                    help='len(place_sem) == num_stages')
parser.add_argument('--ratio_ma',
                    default=0.7,
                    type=float,
                    help='ratio for moving average centroid method')
parser.add_argument('--DA_ent',
                    default='none',
                    type=str,
                    help='entropy-related loss (none | target | attn)')
parser.add_argument('--place_ent',
                    default=['N', 'Y', 'Y', 'N'],
                    type=str,
                    nargs="+",
                    metavar='N',
                    help='len(place_ent) == num_stages')
parser.add_argument('--use_attn',
                    type=str,
                    default='none',
                    choices=['none', 'domain_attn'],
                    help='attention mechanism')
parser.add_argument('--DA_dis',
                    type=str,
                    default='none',
                    choices=['none', 'JAN'],
                    help='discrepancy method for DA')
parser.add_argument('--place_dis',
                    default=['N', 'Y', 'Y', 'N'],
                    type=str,
                    nargs="+",
                    metavar='N',
                    help='len(place_dis) == num_stages')
parser.add_argument('--DA_ens',
                    type=str,
                    default='none',
                    choices=['none', 'MCD', 'SWD'],
                    help='ensemble method for DA')
parser.add_argument('--place_ens',
                    default=['N', 'Y', 'Y', 'N'],
                    type=str,
                    nargs="+",
                    metavar='N',
                    help='len(place_ens) == num_stages')
parser.add_argument('--SS_video',
                    type=str,
                    default='none',
                    choices=['none', 'VCOP'],
                    help='video-based self-supervised learning method')
parser.add_argument('--place_ss',
                    default=['N', 'Y', 'Y', 'N'],
                    type=str,
                    nargs="+",
                    metavar='N',
                    help='len(place_ss) == num_stages')
# config & setting
parser.add_argument('--path_data', default='data/')
parser.add_argument('--path_model', default='models/')
parser.add_argument('--path_result', default='results/')
parser.add_argument('--action', default='train')
parser.add_argument('--use_target', default='none', choices=['none', 'uSv'])
parser.add_argument(
    '--split_target',
    default='0',
    help='split for target data (0: no additional split for target)')
parser.add_argument('--ratio_source',
                    default=1,
                    type=float,
                    help='percentage of total length to use for source data')
parser.add_argument(
    '--ratio_label_source',
    default=1,
    type=float,
    help=
    'percentage of labels to use for source data (after previous processing)')
parser.add_argument('--dataset', default="egobridge")
# hyper-parameters
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('--bS', default=1, type=int, help='batch size')
parser.add_argument('--alpha',
                    default=0.15,
                    type=float,
                    help='weighting for smoothing loss')
parser.add_argument('--tau',
                    default=4,
                    type=float,
                    help='threshold to truncate smoothing loss')
parser.add_argument(
    '--beta',
    default=[-2, -2],
    type=float,
    nargs="+",
    metavar='M',
    help=
    'weighting for adversarial loss & ensemble loss ([frame-beta, video-beta])'
)
parser.add_argument('--iter_max_beta',
                    default=[1000, 1000],
                    type=float,
                    nargs="+",
                    metavar='M',
                    help='for adaptive beta ([frame-beta, video-beta])')
parser.add_argument('--gamma',
                    default=-2,
                    type=float,
                    help='weighting for semantic loss')
parser.add_argument('--iter_max_gamma',
                    default=1000,
                    type=float,
                    help='for adaptive gamma')
parser.add_argument('--mu',
                    default=1,
                    type=float,
                    help='weighting for entropy loss')
parser.add_argument('--nu',
                    default=-2,
                    type=float,
                    help='weighting for the discrepancy loss')
parser.add_argument('--eta',
                    default=1,
                    type=float,
                    help='weighting for the self-supervised loss')
parser.add_argument('--iter_max_nu',
                    default=1000,
                    type=float,
                    metavar='M',
                    help='for adaptive nu')
parser.add_argument('--dim_proj',
                    default=128,
                    type=int,
                    help='projection dimension for SWD')
# runtime
parser.add_argument('--num_epochs', default=150, type=int)
parser.add_argument('--verbose', default=False, action="store_true")
parser.add_argument('--use_best_model',
                    type=str,
                    default='none',
                    choices=['none', 'source', 'target'],
                    help='save best model')
parser.add_argument('--multi_gpu', default=False, action="store_true")
parser.add_argument('--resume_epoch', default=0, type=int)
# tensorboard
parser.add_argument('--use_tensorboard', default=False, action='store_true')
parser.add_argument('--epoch_embedding',
                    default=50,
                    type=int,
                    help='select epoch # to save embedding (-1: all epochs)')
parser.add_argument('--stage_embedding',
                    default=-1,
                    type=int,
                    help='select stage # to save embedding (-1: last stage)')
parser.add_argument(
    '--num_frame_video_embedding',
    default=50,
    type=int,
    help='number of sample frames per video to store embedding')

parser.add_argument('--feat_sample_rate', default=1,
                    type=int)  # use the full temporal resolution @ 15fps
parser.add_argument('--label_sample_rate', default=2,
                    type=int)  # use the full temporal resolution @ 30fps
parser.add_argument('--all_sample_rate', default=1, type=int)
parser.add_argument('--feature_path', type=str)
parser.add_argument("--exp_type",
                    type=str,
                    default="ego-only",
                    choices=[
                        "ego-only", "exo-only", "ego2exo", "exo2ego",
                        "ego-only-gazed", "ego-only-center",
                        'ego-exo-cotraining-ego', 'ego-exo-cotraining-exo', 
                        'ego-exo-cotraining-gazed-ego', 'ego-exo-cotraining-gazed-exo', 
                        'ego-exo-da-exo','exo-ego-da-ego',
                        'ego-exo-gazed-da-exo','exo-ego-gazed-da-ego',
                    ])
parser.add_argument(
    "--test",
    action="store_true",
)

args = parser.parse_args()

print("exp_type: ", args.exp_type)
print("is test: ", args.test)

# check whether place_adv & place_sem are valid
if len(args.place_adv) != args.num_stages:
    raise ValueError('len(place_dis) should be equal to num_stages')
if len(args.place_sem) != args.num_stages:
    raise ValueError('len(place_sem) should be equal to num_stages')
if len(args.place_ent) != args.num_stages:
    raise ValueError('len(place_ent) should be equal to num_stages')
if len(args.place_dis) != args.num_stages:
    raise ValueError('len(place_dis) should be equal to num_stages')
if len(args.place_ens) != args.num_stages:
    raise ValueError('len(place_ens) should be equal to num_stages')
if len(args.place_ss) != args.num_stages:
    raise ValueError('len(place_ss) should be equal to num_stages')

if args.use_target == 'none':
    args.DA_adv = 'none'
    args.DA_sem = 'none'
    args.DA_ent = 'none'
    args.DA_dis = 'none'
    args.DA_ens = 'none'
    args.SS_video = 'none'  # focus on cross-domain setting

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

assert args.dataset == "egobridge"
# ====== Load files ====== #
ego_splits_dir = "splits_dir_fps25"
ego_splits_dir_gaze = "splits_dir_fps25_gaze"
exo_splits_dir = "splits_dir_fps25_exo"

train_source_vid_list_file, train_source_feat_suffix, \
test_source_vid_list_file, test_source_val_feat_suffix, \
train_target_vid_list_file, train_target_feat_suffix, \
test_target_vid_list_file, test_target_feat_suffix = get_annotations_from_settings(args)


# if args.exp_type == "ego-exo-cotraining":
#     # train source -> ego-train
#     train_source_vid_list_file = [
#         f"../generated_data/{ego_splits_dir}/train_seen_annotations.bundle",
#         f"../generated_data/{exo_splits_dir}/train_seen_annotations_exo.bundle",
#     ]
#     train_source_feat_suffix = ["_320p_25fps", ""]

#     # val source -> ego-val
#     test_source_vid_list_file = [
#         f"../generated_data/{ego_splits_dir}/val_seen_annotations.bundle",
#         f"../generated_data/{ego_splits_dir}/val_unseen_agent_and_cooking_task_annotations.bundle",
#         f"../generated_data/{ego_splits_dir}/val_unseen_agent_annotations.bundle",
#         f"../generated_data/{ego_splits_dir}/val_unseen_cooking_task_annotations.bundle",
#     ]
#     test_source_val_feat_suffix = [
#         "_320p_25fps", "_320p_25fps", "_320p_25fps", "_320p_25fps"
#     ]

#     # train target -> ego-train
#     train_target_vid_list_file = [
#         f"../generated_data/{ego_splits_dir}/train_seen_annotations.bundle",
#         f"../generated_data/{exo_splits_dir}/train_seen_annotations_exo.bundle",
#     ]
#     train_target_feat_suffix = ["_320p_25fps", ""]

#     test_target_vid_list_file = [
#         f"../generated_data/{ego_splits_dir}/val_seen_annotations.bundle",
#         f"../generated_data/{ego_splits_dir}/val_unseen_agent_and_cooking_task_annotations.bundle",
#         f"../generated_data/{ego_splits_dir}/val_unseen_agent_annotations.bundle",
#         f"../generated_data/{ego_splits_dir}/val_unseen_cooking_task_annotations.bundle",
#     ]

#     test_target_feat_suffix = [
#         "_320p_25fps", "_320p_25fps", "_320p_25fps", "_320p_25fps"
#     ]
#     if args.test:
#         print("test mode: ego-only")
#         test_target_vid_list_file = [
#             f"../generated_data/{ego_splits_dir}/test_seen_annotations.bundle",
#             f"../generated_data/{ego_splits_dir}/test_unseen_agent_and_cooking_task_annotations.bundle",
#             f"../generated_data/{ego_splits_dir}/test_unseen_agent_annotations.bundle",
#             f"../generated_data/{ego_splits_dir}/test_unseen_cooking_task_annotations.bundle",
#         ]

#         test_target_feat_suffix = [
#             "_320p_25fps", "_320p_25fps", "_320p_25fps", "_320p_25fps"
#         ]
# elif args.exp_type == "ego-exo-cotraining-gaze50":
#     # train source -> ego-train
#     train_source_vid_list_file = [
#         f"../generated_data/{ego_splits_dir_gaze}/train_seen_annotations.bundle",
#         f"../generated_data/{exo_splits_dir}/train_seen_annotations_exo.bundle",
#     ]
#     train_source_feat_suffix = ["_50", ""]

#     # val source -> ego-val
#     test_source_vid_list_file = [
#         f"../generated_data/{ego_splits_dir_gaze}/val_seen_annotations.bundle",
#         f"../generated_data/{ego_splits_dir_gaze}/val_unseen_agent_and_cooking_task_annotations.bundle",
#         f"../generated_data/{ego_splits_dir_gaze}/val_unseen_agent_annotations.bundle",
#         f"../generated_data/{ego_splits_dir_gaze}/val_unseen_cooking_task_annotations.bundle",
#     ]
#     test_source_val_feat_suffix = ["_50", "_50", "_50", "_50"]

#     # train target -> ego-train
#     train_target_vid_list_file = [
#         f"../generated_data/{ego_splits_dir_gaze}/train_seen_annotations.bundle",
#         f"../generated_data/{exo_splits_dir}/train_seen_annotations_exo.bundle",
#     ]
#     train_target_feat_suffix = ["_50", ""]

#     test_target_vid_list_file = [
#         f"../generated_data/{ego_splits_dir_gaze}/val_seen_annotations.bundle",
#         f"../generated_data/{ego_splits_dir_gaze}/val_unseen_agent_and_cooking_task_annotations.bundle",
#         f"../generated_data/{ego_splits_dir_gaze}/val_unseen_agent_annotations.bundle",
#         f"../generated_data/{ego_splits_dir_gaze}/val_unseen_cooking_task_annotations.bundle",
#     ]

#     test_target_feat_suffix = ["_50", "_50", "_50", "_50"]
#     if args.test:
#         print("test mode: ego-only-gaze50")
#         test_target_vid_list_file = [
#             f"../generated_data/{ego_splits_dir_gaze}/test_seen_annotations.bundle",
#             f"../generated_data/{ego_splits_dir_gaze}/test_unseen_agent_and_cooking_task_annotations.bundle",
#             f"../generated_data/{ego_splits_dir_gaze}/test_unseen_agent_annotations.bundle",
#             f"../generated_data/{ego_splits_dir_gaze}/test_unseen_cooking_task_annotations.bundle",
#         ]

#         test_target_feat_suffix = ["_50", "_50", "_50", "_50"]
# elif args.exp_type == "ego-only":
#     # train source -> ego-train
#     train_source_vid_list_file = [
#         f"../generated_data/{ego_splits_dir}/train_seen_annotations.bundle",
#     ]
#     train_source_feat_suffix = ["_320p_25fps"]

#     # val source -> ego-val
#     test_source_vid_list_file = [
#         f"../generated_data/{ego_splits_dir}/val_seen_annotations.bundle",
#         f"../generated_data/{ego_splits_dir}/val_unseen_agent_and_cooking_task_annotations.bundle",
#         f"../generated_data/{ego_splits_dir}/val_unseen_agent_annotations.bundle",
#         f"../generated_data/{ego_splits_dir}/val_unseen_cooking_task_annotations.bundle",
#     ]
#     test_source_val_feat_suffix = [
#         "_320p_25fps", "_320p_25fps", "_320p_25fps", "_320p_25fps"
#     ]

#     # train target -> ego-train
#     train_target_vid_list_file = [
#         f"../generated_data/{ego_splits_dir}/train_seen_annotations.bundle",
#     ]
#     train_target_feat_suffix = ["_320p_25fps"]

#     test_target_vid_list_file = [
#         f"../generated_data/{ego_splits_dir}/val_seen_annotations.bundle",
#         f"../generated_data/{ego_splits_dir}/val_unseen_agent_and_cooking_task_annotations.bundle",
#         f"../generated_data/{ego_splits_dir}/val_unseen_agent_annotations.bundle",
#         f"../generated_data/{ego_splits_dir}/val_unseen_cooking_task_annotations.bundle",
#     ]

#     test_target_feat_suffix = [
#         "_320p_25fps", "_320p_25fps", "_320p_25fps", "_320p_25fps"
#     ]
#     if args.test:
#         print("test mode: ego-only")
#         test_target_vid_list_file = [
#             f"../generated_data/{ego_splits_dir}/test_seen_annotations.bundle",
#             f"../generated_data/{ego_splits_dir}/test_unseen_agent_and_cooking_task_annotations.bundle",
#             f"../generated_data/{ego_splits_dir}/test_unseen_agent_annotations.bundle",
#             f"../generated_data/{ego_splits_dir}/test_unseen_cooking_task_annotations.bundle",
#         ]

#         test_target_feat_suffix = [
#             "_320p_25fps", "_320p_25fps", "_320p_25fps", "_320p_25fps"
#         ]
# elif args.exp_type == "ego-only-gaze50":
#     # train source -> ego-train
#     train_source_vid_list_file = [
#         f"../generated_data/{ego_splits_dir_gaze}/train_seen_annotations.bundle",
#     ]
#     train_source_feat_suffix = ["_50"]

#     # val source -> ego-val
#     test_source_vid_list_file = [
#         f"../generated_data/{ego_splits_dir_gaze}/val_seen_annotations.bundle",
#         f"../generated_data/{ego_splits_dir_gaze}/val_unseen_agent_and_cooking_task_annotations.bundle",
#         f"../generated_data/{ego_splits_dir_gaze}/val_unseen_agent_annotations.bundle",
#         f"../generated_data/{ego_splits_dir_gaze}/val_unseen_cooking_task_annotations.bundle",
#     ]
#     test_source_val_feat_suffix = ["_50", "_50", "_50", "_50"]

#     # train target -> ego-train
#     train_target_vid_list_file = [
#         f"../generated_data/{ego_splits_dir_gaze}/train_seen_annotations.bundle",
#     ]
#     train_target_feat_suffix = ["_50"]

#     test_target_vid_list_file = [
#         f"../generated_data/{ego_splits_dir_gaze}/val_seen_annotations.bundle",
#         f"../generated_data/{ego_splits_dir_gaze}/val_unseen_agent_and_cooking_task_annotations.bundle",
#         f"../generated_data/{ego_splits_dir_gaze}/val_unseen_agent_annotations.bundle",
#         f"../generated_data/{ego_splits_dir_gaze}/val_unseen_cooking_task_annotations.bundle",
#     ]

#     test_target_feat_suffix = ["_50", "_50", "_50", "_50"]
#     if args.test:
#         print("test mode: ego-only")
#         test_target_vid_list_file = [
#             f"../generated_data/{ego_splits_dir_gaze}/test_seen_annotations.bundle",
#             f"../generated_data/{ego_splits_dir_gaze}/test_unseen_agent_and_cooking_task_annotations.bundle",
#             f"../generated_data/{ego_splits_dir_gaze}/test_unseen_agent_annotations.bundle",
#             f"../generated_data/{ego_splits_dir_gaze}/test_unseen_cooking_task_annotations.bundle",
#         ]

#         test_target_feat_suffix = ["_50", "_50", "_50", "_50"]
# elif args.exp_type == "ego-only-center":
#     # train source -> ego-train
#     train_source_vid_list_file = [
#         f"../generated_data/{ego_splits_dir_gaze}/train_seen_annotations.bundle",
#     ]
#     train_source_feat_suffix = ["_320p_25fps_center"]

#     # val source -> ego-val
#     test_source_vid_list_file = [
#         f"../generated_data/{ego_splits_dir_gaze}/val_seen_annotations.bundle",
#         f"../generated_data/{ego_splits_dir_gaze}/val_unseen_agent_and_cooking_task_annotations.bundle",
#         f"../generated_data/{ego_splits_dir_gaze}/val_unseen_agent_annotations.bundle",
#         f"../generated_data/{ego_splits_dir_gaze}/val_unseen_cooking_task_annotations.bundle",
#     ]
#     test_source_val_feat_suffix = [
#         "_320p_25fps_center", "_320p_25fps_center", "_320p_25fps_center",
#         "_320p_25fps_center"
#     ]

#     # train target -> ego-train
#     train_target_vid_list_file = [
#         f"../generated_data/{ego_splits_dir_gaze}/train_seen_annotations.bundle",
#     ]
#     train_target_feat_suffix = ["_320p_25fps_center"]

#     test_target_vid_list_file = [
#         f"../generated_data/{ego_splits_dir_gaze}/val_seen_annotations.bundle",
#         f"../generated_data/{ego_splits_dir_gaze}/val_unseen_agent_and_cooking_task_annotations.bundle",
#         f"../generated_data/{ego_splits_dir_gaze}/val_unseen_agent_annotations.bundle",
#         f"../generated_data/{ego_splits_dir_gaze}/val_unseen_cooking_task_annotations.bundle",
#     ]

#     test_target_feat_suffix = [
#         "_320p_25fps_center", "_320p_25fps_center", "_320p_25fps_center",
#         "_320p_25fps_center"
#     ]
#     if args.test:
#         print("test mode: ego-only")
#         test_target_vid_list_file = [
#             f"../generated_data/{ego_splits_dir_gaze}/test_seen_annotations.bundle",
#             f"../generated_data/{ego_splits_dir_gaze}/test_unseen_agent_and_cooking_task_annotations.bundle",
#             f"../generated_data/{ego_splits_dir_gaze}/test_unseen_agent_annotations.bundle",
#             f"../generated_data/{ego_splits_dir_gaze}/test_unseen_cooking_task_annotations.bundle",
#         ]

#         test_target_feat_suffix = [
#             "_320p_25fps_center", "_320p_25fps_center", "_320p_25fps_center",
#             "_320p_25fps_center"
#         ]
# elif args.exp_type == "exo-only":
#     # train source -> exo-train
#     train_source_vid_list_file = [
#         f"../generated_data/{exo_splits_dir}/train_seen_annotations_exo.bundle",
#     ]
#     train_source_feat_suffix = [""]

#     # val source -> exo-val
#     test_source_vid_list_file = [
#         f"../generated_data/{exo_splits_dir}/val_seen_annotations_exo.bundle",
#     ]
#     test_source_val_feat_suffix = [""]

#     # train target -> exo-train
#     train_target_vid_list_file = [
#         f"../generated_data/{exo_splits_dir}/train_seen_annotations_exo.bundle",
#     ]
#     train_target_feat_suffix = [""]

#     test_target_vid_list_file = [
#         f"../generated_data/{exo_splits_dir}/val_seen_annotations_exo.bundle",
#     ]
#     test_target_feat_suffix = [""]
#     if args.test:
#         print("test mode: exo-only")
#         test_target_vid_list_file = [
#             f"../generated_data/{exo_splits_dir}/test_seen_annotations_exo.bundle",
#         ]

#         test_target_feat_suffix = [""]
# elif args.exp_type == "ego2exo":
#     # train source -> ego-train
#     train_source_vid_list_file = [
#         f"../generated_data/{ego_splits_dir}/train_seen_annotations.bundle",
#     ]
#     train_source_feat_suffix = ["_320p_25fps"]

#     # val source -> exo-val
#     test_source_vid_list_file = [
#         f"../generated_data/{ego_splits_dir}/val_seen_annotations.bundle",
#         f"../generated_data/{ego_splits_dir}/val_unseen_agent_and_cooking_task_annotations.bundle",
#         f"../generated_data/{ego_splits_dir}/val_unseen_agent_annotations.bundle",
#         f"../generated_data/{ego_splits_dir}/val_unseen_cooking_task_annotations.bundle",
#     ]
#     test_source_val_feat_suffix = [
#         "_320p_25fps", "_320p_25fps", "_320p_25fps", "_320p_25fps"
#     ]

#     # train target -> ego-train
#     train_target_vid_list_file = [
#         f"../generated_data/{exo_splits_dir}/train_seen_annotations_exo.bundle",
#     ]
#     train_target_feat_suffix = [""]

#     test_target_vid_list_file = [
#         f"../generated_data/{exo_splits_dir}/val_seen_annotations_exo.bundle",
#     ]

#     test_target_feat_suffix = [""]
#     if args.test:
#         print("test mode: ego2exo")
#         test_target_vid_list_file = [
#             f"../generated_data/{exo_splits_dir}/test_seen_annotations_exo.bundle",
#         ]

#         test_target_feat_suffix = [""]
# elif args.exp_type == "exo2ego":
#     # train source -> exo-train
#     train_source_vid_list_file = [
#         f"../generated_data/{exo_splits_dir}/train_seen_annotations_exo.bundle",
#     ]
#     train_source_feat_suffix = [""]
#     # features_path = args.path_data + args.dataset + "/features/"
#     # gt_path = args.path_data + args.dataset + "/groundTruth/"

#     # val source -> exo-val
#     test_source_vid_list_file = [
#         f"../generated_data/{exo_splits_dir}/val_seen_annotations_exo.bundle",
#     ]
#     test_source_val_feat_suffix = [""]

#     # train target -> ego-train
#     train_target_vid_list_file = [
#         f"../generated_data/{ego_splits_dir}/val_seen_annotations.bundle",
#     ]
#     train_target_feat_suffix = ["_320p_25fps"]

#     test_target_vid_list_file = [
#         f"../generated_data/{ego_splits_dir}/val_seen_annotations.bundle",
#         f"../generated_data/{ego_splits_dir}/val_unseen_agent_and_cooking_task_annotations.bundle",
#         f"../generated_data/{ego_splits_dir}/val_unseen_agent_annotations.bundle",
#         f"../generated_data/{ego_splits_dir}/val_unseen_cooking_task_annotations.bundle",
#     ]

#     test_target_feat_suffix = [
#         "_320p_25fps", "_320p_25fps", "_320p_25fps", "_320p_25fps"
#     ]
#     if args.test:
#         print("test mode: exo2ego")
#         test_target_vid_list_file = [
#             f"../generated_data/{ego_splits_dir}/test_seen_annotations.bundle",
#             f"../generated_data/{ego_splits_dir}/test_unseen_agent_and_cooking_task_annotations.bundle",
#             f"../generated_data/{ego_splits_dir}/test_unseen_agent_annotations.bundle",
#             f"../generated_data/{ego_splits_dir}/test_unseen_cooking_task_annotations.bundle",
#         ]

#         test_target_feat_suffix = [
#             "_320p_25fps", "_320p_25fps", "_320p_25fps", "_320p_25fps"
#         ]
# else:
#     NotImplementedError

# all feature
features_path = args.feature_path
gt_path = os.path.join(args.path_data, "gts_fps25/")

# vid_list_file = args.path_data+args.dataset+"/splits/train.split" + args.split + ".bundle"

# vid_list_file_target = args.path_data + args.dataset+"/splits/test.split"+args.split+".bundle"
# vid_list_file_test = vid_list_file_target

# if args.split_target != '0':
#     vid_list_file_target = args.path_data + args.dataset + "/splits/test_train_" + args.split_target + ".split" + args.split + ".bundle"
#     vid_list_file_test = args.path_data + args.dataset + "/splits/test_test_" + args.split_target + ".split" + args.split + ".bundle"

# mapping_file = args.path_data + args.dataset + "/mapping.txt"  # mapping between classes & indices

split = "test" if args.test else "val"
model_dir = os.path.join(args.path_model)
results_dir = os.path.join(args.path_result, split)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

actions_dict = dict()
num_classes = 28  # 1 + 27
for i in range(num_classes):
    actions_dict[str(i)] = i
# file_ptr = open(mapping_file, 'r')
# actions = file_ptr.read().split('\n')[:-1]  # list of classes
# file_ptr.close()
# actions_dict = dict()
# for a in actions:
#     actions_dict[a.split()[1]] = int(a.split()[0])

num_classes = len(actions_dict)

# initialize model & trainer
model = MultiStageModel(args, num_classes)
trainer = Trainer(num_classes)

# ====== Main Program ====== #
start_time = time.time()
if args.action == "train":
    batch_gen_source_train = BatchGenerator(
        num_classes,
        actions_dict,
        gt_path,
        features_path,
        feat_sample_rate=args.feat_sample_rate,
        label_sample_rate=args.label_sample_rate,
        all_sample_rate=args.all_sample_rate,
        feat_suffix=train_source_feat_suffix)
    batch_gen_target_train = BatchGenerator(
        num_classes,
        actions_dict,
        gt_path,
        features_path,
        feat_sample_rate=args.feat_sample_rate,
        label_sample_rate=args.label_sample_rate,
        all_sample_rate=args.all_sample_rate,
        feat_suffix=train_target_feat_suffix)
    batch_gen_source_val = BatchGenerator(
        num_classes,
        actions_dict,
        gt_path,
        features_path,
        feat_sample_rate=args.feat_sample_rate,
        label_sample_rate=args.label_sample_rate,
        all_sample_rate=args.all_sample_rate,
        feat_suffix=test_source_val_feat_suffix)
    batch_gen_target_val = BatchGenerator(
        num_classes,
        actions_dict,
        gt_path,
        features_path,
        feat_sample_rate=args.feat_sample_rate,
        label_sample_rate=args.label_sample_rate,
        all_sample_rate=args.all_sample_rate,
        feat_suffix=test_target_feat_suffix)
    batch_gen_source_train.read_data(
        train_source_vid_list_file)  # read & shuffle the source training list
    batch_gen_target_train.read_data(
        train_target_vid_list_file)  # read & shuffle the target training list
    batch_gen_source_val.read_data(
        test_source_vid_list_file)  # read & shuffle the source validation list
    batch_gen_target_val.read_data(
        test_target_vid_list_file)  # read & shuffle the target validation list
    trainer.train(model, model_dir, results_dir, batch_gen_source_train,
                  batch_gen_target_train, batch_gen_source_val,
                  batch_gen_target_val, device, args)

if args.action == "predict":
    predict(model, model_dir, results_dir, features_path,
            test_target_vid_list_file, test_target_feat_suffix,
            args.feat_sample_rate, args.all_sample_rate, args.num_epochs,
            actions_dict, device, args)

end_time = time.time()

if args.verbose:
    print('')
    print('total running time:', end_time - start_time)
