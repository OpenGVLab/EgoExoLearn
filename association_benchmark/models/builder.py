
from mmcv.utils import Registry
from omegaconf import OmegaConf
from . import retriever
import sys
import os

def build_model(config):
    ### this is a retrieval model ###
    model = getattr(retriever, config.name)(
        pretrained=config.load_visual_pretrained,
        pretrained2d=config.load_visual_pretrained is not None,
        text_use_cls_token=config.use_cls_token,
        project_embed_dim=config.project_embed_dim,
        timesformer_gated_xattn=config.timesformer_gated_xattn,
        timesformer_freeze_space=config.timesformer_freeze_space,
        num_frames=config.clip_length,
        drop_path_rate=config.drop_path_rate,
        temperature_init=config.temperature_init,
        pretrained_visual_checkpoint=config.pretrained_visual_checkpoint, ### omit for orignal clip
    )
    return model