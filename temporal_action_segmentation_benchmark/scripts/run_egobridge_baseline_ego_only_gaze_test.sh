#!/bin/bash
# ----------------------------------------------------------------------------------

# === Mode Switch On/Off === #
experiment_type=ego-only-gazed
cv_method=zs # cross-view method, zero-shot, domain-adaption, knowledge-distillation, cotrain

training=false # true | false
predict=true # true | false
eval=true # true | false


# === Paths === #
path_data=../egobridge_all_annotations/as_annotation # user-defined
path_model=./models/egobridge-$experiment_type-$cv_method/ # user-defined (will create if not existing)
path_result=./results/egobridge-$experiment_type-$cv_method/ # user-defined (will create if not existing)
path_feature=/mnt/petrelfs/share_data/chenguo/egobridge_final_annotations/feats/i3d_rgb_features/

# === Config & Setting === #
use_target=none # none | uSv
dataset=egobridge # egobridge
use_best_model=none # no/mnt/petrelfs/chenguo/data/sstda/ne | source
num_seg=1 # number of segments for each video (SSTDA: 2 | VCOP: 3 | others: 1)

# SSTDA
DA_adv=none # none | rev_grad
DA_adv_video=none # none | rev_grad_ssl
use_attn=none # none | domain_attn
DA_ent=none # none | attn

# MADA
multi_adv_2=N # Y | N

# MSTN
method_centroid=none # none | prob_hard

# JAN
DA_dis=none # none | JAN

# MCD / SWD
DA_ens=none # none | MCD | SWD

# VCOP
SS_video=none # none | VCOP

# --- hyper-parameters --- #
iter_max_0=2000
iter_max_1=1400
iter_max_nu=37500
mu=0
eta=0
lr=0.0005

# === Main Program === #
echo 'use_target: '$use_target', dataset: '$dataset

## run codes ##
# train
if ($training)
then
    python egolearner_main.py --path_data=$path_data --path_model=$path_model --path_result=$path_result \
    --action=train --dataset=$dataset --lr $lr --use_target $use_target \
    --DA_adv $DA_adv --DA_adv_video $DA_adv_video --num_seg $num_seg --beta -2 -2 --iter_max_beta $iter_max_0 $iter_max_1 \
    --DA_ent $DA_ent --mu $mu --use_attn $use_attn \
    --multi_adv N $multi_adv_2 \
    --method_centroid $method_centroid --iter_max_gamma $iter_max_0 \
    --DA_dis $DA_dis --iter_max_nu $iter_max_nu \
    --DA_ens $DA_ens \
    --SS_video $SS_video --eta $eta \
    --use_best_model $use_best_model --verbose \
    --feature_path $path_feature \
    --feat_sample_rate 1 \
    --label_sample_rate 1 \
    --all_sample_rate 5 \
    --features_dim 1024 \
    --exp_type $experiment_type \
    
fi

# predict
for ((i=1;i<=150;i++))
do
  # predict
  if ($predict)
  then
      python egolearner_main.py --path_data=$path_data --path_model=$path_model --path_result=$path_result \
      --action=predict --dataset=$dataset --lr $lr  --use_target $use_target \
      --DA_adv_video $DA_adv_video --num_seg $num_seg --use_attn $use_attn \
      --multi_adv N $multi_adv_2 --method_centroid $method_centroid --DA_ens $DA_ens --SS_video $SS_video \
      --num_epochs $i \
      --feature_path $path_feature \
      --feat_sample_rate 1 \
      --label_sample_rate 1 \
      --all_sample_rate 5 \
      --features_dim 1024 \
      --exp_type $experiment_type \
      --test

  fi

  # eval
  if ($eval)
  then
      python egolearner_eval.py --path_data=$path_data --path_result=$path_result --dataset=$dataset --exp_type $experiment_type --test 
  fi
done


#----------------------------------------------------------------------------------
exit 0
