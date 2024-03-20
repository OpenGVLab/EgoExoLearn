from .openai_model import QuickGELU, Transformer
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from .timesformer import SpaceTimeTransformer
from .openai_clip import load as load_openai_clip
from .model_utils import remap_keys
from copy import deepcopy

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 tempearture_init=0.07,
                 **kwargs,
                 ):
        super().__init__()

        self.context_length = context_length
        self.vision_width = vision_width

        self.visual = vision_model
        self.attn_mask = self.build_attention_mask()
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.attn_mask,
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = nn.LayerNorm(transformer_width)  # used to be `models.transformer.LayerNorm``

        self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        print("=> initialize initial temperature with {}".format(tempearture_init))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / tempearture_init))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def inflate_positional_embeds(self, curr_frames):
        '''
        # positional_embed: [self.ctx_length, D]
        # 
        # '''
        if self.context_length == curr_frames:
            return self.positional_embedding, self.attn_mask
        if self.context_length > curr_frames:
            return self.positional_embedding[:, :curr_frames, :], self.build_attention_mask(curr_frames)
        if self.context_length < curr_frames:
            new_temporal_embed = F.interpolate(self.positional_embedding.unsqueeze(0).unsqueeze(0), (curr_frames, self.positional_embedding.shape[-1]), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            return torch.nn.Parameter(new_temporal_embed).to(self.positional_embedding.device), self.build_attention_mask(curr_frames)

    def build_attention_mask(self, ctx_length=None):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        if ctx_length is None:
            mask = torch.empty(self.context_length, self.context_length)
        else:
            mask = torch.empty(ctx_length, ctx_length)

        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, image, use_checkpoint=False, apply_project=True):
        x = self.visual(image, use_checkpoint=use_checkpoint)
        if isinstance(x, list):
            assert len(x) == 1
            x = x[0]
        if not apply_project:
            return x
        x = x @ self.image_projection

        return x

    def encode_text(self, text, use_checkpoint=False):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        # x = x + self.positional_embedding
        curr_ctx_len = x.shape[1]
        positional_embedding, attn_mask = self.inflate_positional_embeds(curr_ctx_len)
        x = x + positional_embedding

        x = x.permute(1, 0, 2)  # NLD -> LND
        #x = self.transformer(x, use_checkpoint=use_checkpoint)
        x = self.transformer(x, use_checkpoint=use_checkpoint, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text, use_checkpoint=False, norm_embed=False):
        image_embed = self.encode_image(image, use_checkpoint=use_checkpoint)
        text_embed = self.encode_text(text, use_checkpoint=use_checkpoint)

        if norm_embed:
            image_embed = F.normalize(image_embed, dim=-1)
            text_embed = F.normalize(text_embed, dim=-1)
        return {'image_embed': image_embed,
                'text_embed': text_embed,
                'logit_scale': self.logit_scale.exp()}


def CLIP_OPENAI_TIMESFORMER_BASE(
    num_frames=4, timesformer_gated_xattn=False, drop_path_rate=0, timesformer_freeze_space=False,
    temperature_init=0.07, project_embed_dim=256, freeze_text_encoder=False, pretrained_visual_checkpoint=None,
    **kwargs,
):
    vision_model = SpaceTimeTransformer(
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
        drop_path_rate=drop_path_rate,
    )
    clip_model, _ = load_openai_clip('ViT-B/16', 'cpu')
    # print("=> Loading CLIP (ViT-B/16) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=12)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    # print(res)
    if timesformer_freeze_space:
        print("=> Freeze the space part in TimeSformer")
        freeze_list, unfreeze_list = [], []
        for n, p in vision_model.named_parameters():
            if n not in remapped_state_dict or n == 'cls_token':
                p.requires_grad = True
                unfreeze_list.append(n)
            else:
                p.requires_grad = False
                freeze_list.append(n)
        print("Freeze the pretrained parts in TimeSformer: {}".format(freeze_list))
        print(" Learn the rest parts in TimeSformer: {}".format(unfreeze_list))
    
    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()
    model = CLIP(
        embed_dim=project_embed_dim,
        vision_width=768,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        tempearture_init=temperature_init,
        **kwargs
    )
    model.transformer.load_state_dict(clip_model.transformer.state_dict())
    model.token_embedding.load_state_dict(clip_model.token_embedding.state_dict())
    model.positional_embedding.data.copy_(clip_model.positional_embedding.data)
    model.ln_final.load_state_dict(clip_model.ln_final.state_dict())
    if project_embed_dim == clip_model.text_projection.shape[1]:
        print("=> Loading CLIP's text_projection, image_projection and logit_scale directly")
        model.image_projection.data.copy_(clip_model.visual.proj.data)
        model.text_projection.data.copy_(clip_model.text_projection.data)
        model.logit_scale.data.copy_(clip_model.logit_scale.data)
    
    
    if freeze_text_encoder:
        for module in [model.token_embedding, model.positional_embedding, model.transformer, model.ln_final, model.text_projection]:
            if isinstance(module, nn.Parameter):
                module.requires_grad = False
            else:
                for p in module.parameters():
                    p.requires_grad=False
          
    return model

