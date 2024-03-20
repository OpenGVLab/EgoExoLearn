import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Rank-aware Attention Network for Cross-view Referenced Skill Assessment")

# ============================= Model Configs ================================

parser.add_argument('--attention', action='store_true', default=True)
parser.add_argument('--num_filters', type=int, default=3)
parser.add_argument('--diversity_loss', action='store_true', default=True)
parser.add_argument('--disparity_loss', action='store_true', default=True)
parser.add_argument('--rank_aware_loss', action='store_true', default=True)
parser.add_argument('--lambda_param', type=float, default=0.1, help='weight of the diversity loss')
parser.add_argument('--m1', type=float, default=1.0, help='margin for ranking loss')
parser.add_argument('--m2', type=float, default=0.05, help='margin for disparity loss')
parser.add_argument('--m3', type=float, default=0.15, help='margin for rank aware loss')

# =========================== Learning Configs ===============================
parser.add_argument('--epochs', default=2000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--transform', action='store_true', default=True)
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')

# ============================ Monitor Configs ===============================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--eval-freq', '-ef', default=10, type=int,
                    metavar='N', help='evaluation frequency (default: 10)')

# ============================ Runtime Configs ===============================
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', default=True,
                    help='evaluate model on validation set')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)





parser.add_argument('--train_list', type=str, default='./train_split_skill.txt')
parser.add_argument('--val_list', type=str, default='./val_split_skill.txt')





#####################################################

gaze = '100' # 33,50,75,100

triplet_loss = 'y' # 'y' yes or 'n' no
relation_network = 'n' # 'y' yes or 'n' no
# triplet loss and RN should not be used together

# action_select = '18' 
# action_select = '06' 
# action_select = '20' 
# action_select = '13,14,15' 
action_select = '06,13,14,15,18,20'

debug = False
use_feature = 'videomae' # ['videomae', 'i3d']
if use_feature == 'videomae': mae_model_size = 'b' # ['s', 'b', 'l']

#####################################################
# for evaluation

parser.add_argument('--resume_train', action='store_true', default=False)
parser.add_argument('--resume_ckpt', type=str, default='./1117_vmaeb_ckpt/checkpoints_vmaeb_aall_g%s_t%s_r%s/_model_best.pth.tar'%(gaze,triplet_loss,relation_network))

#####################################################




if use_feature == 'videomae': root_path = '/work/skill_frames/videomae_features_%s/gaze100/'%(mae_model_size) if gaze == '100' else '/work/skill_frames/videomae_features_%s/gaze%s/'%(mae_model_size, gaze)
elif use_feature == 'i3d': root_path = '/work/skill_frames/i3d/save_sh_skill_i3d/' if gaze == '100' else '/work/skill_frames/i3d/save_sh_skill_i3d_%s/'%gaze
else: print('error'); exit()

if action_select in ['06','18','20']:
    checkpoint_path = 'checkpoints_a%s_g%s_t%s_r%s/'%(action_select,gaze,triplet_loss,relation_network)
    run_path = 'runs_a%s_g%s_t%s_r%s/'%(action_select,gaze,triplet_loss,relation_network)
elif action_select in ['13,14,15']:
    checkpoint_path = 'checkpoints_a131415_g%s_t%s_r%s/'%(gaze,triplet_loss,relation_network)
    run_path = 'runs_a131415_g%s_t%s_r%s/'%(gaze,triplet_loss,relation_network)
elif action_select in ['06,13,14,15,18,20']:
    checkpoint_path = 'checkpoints_aall_g%s_t%s_r%s/'%(gaze,triplet_loss,relation_network)
    run_path = 'runs_aall_g%s_t%s_r%s/'%(gaze,triplet_loss,relation_network)
else:
    assert False, 'action_select error'
if debug:
    checkpoint_path = 'checkpoints_debug/'
    run_path = 'runs_debug/'

if use_feature == 'videomae':
    checkpoint_path = checkpoint_path.replace('checkpoints', 'checkpoints_vmae%s'%mae_model_size)
    run_path = run_path.replace('runs', 'runs_vmae%s'%mae_model_size)
                                                                                                                                                                                                                                                                                                                                                                                  
parser.add_argument('--use_gpu_num', type=str, default='0')
parser.add_argument('--action_select', type=str, default=action_select) 
parser.add_argument('--root_path', type=str, default=root_path)
parser.add_argument('--snapshot_pref', type=str, default=checkpoint_path)
parser.add_argument('--run_folder', type=str, default=run_path)

tmp = False if (triplet_loss == 'n' and relation_network == 'n') else True
parser.add_argument('--use_exo', action='store_true', default=tmp)
if use_feature == 'videomae': parser.add_argument('--exo_root_path', type=str, default='/work/skill_frames/videomae_features_%s/exo/'%(mae_model_size))
elif use_feature == 'i3d': parser.add_argument('--exo_root_path', type=str, default='/work/skill_frames/i3d/save_sh_skill_i3d_exo/')
else: print('error'); exit()

# triplet loss and RN should not be used together
tmp = True if triplet_loss == 'y' else False
parser.add_argument('--triplet_loss', action='store_true', default=tmp)
tmp = True if relation_network == 'y' else False
parser.add_argument('--relation_network', action='store_true', default=tmp)
# if use RN, num_samples should be 10*2, else 10
tmp = 20 if relation_network == 'y' else 10
parser.add_argument('--num_samples', type=int, default=tmp)

if use_feature == 'videomae':
    if mae_model_size == 'l':
        parser.add_argument('--input_size', type=int, default=1024)
    elif mae_model_size == 'b':
        parser.add_argument('--input_size', type=int, default=768)
    elif mae_model_size == 's':
        parser.add_argument('--input_size', type=int, default=1024)
elif use_feature == 'i3d':
    parser.add_argument('--input_size', type=int, default=1024)
else: print('error'); exit()



