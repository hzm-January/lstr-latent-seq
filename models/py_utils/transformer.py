#  Copyright (c) 3.2022. Yinyu Nie
#  License: MIT
import torch
import torch.nn as nn
from external.fast_transformers.fast_transformers.builders import TransformerEncoderBuilder, TransformerDecoderBuilder


class Transformer(nn.Module):
    def __init__(self):
        '''
        Encode scene priors from embeddings
        :param cfg: configuration file.
        :param optim_spec: optimizer parameters.
        '''
        super(Transformer, self).__init__()
        '''Optimizer parameters used in training'''
        # self.optim_spec = optim_spec
        # self.device = device

        '''Network'''
        # Parameters
        self.z_dim = 32  # 512
        # self.inst_latent_len = cfg.config.data.backbone_latent_len  # 1024
        self.max_obj_num = 7  # 53
        d_model = 32
        n_head = 4

        # Build Networks
        # empty room token in transformer encoder
        # 字典长度：房间类型数量，词向量维度：自定义512
        self.empty_token_embedding = nn.Embedding(7000, self.z_dim)  # (7000, 32)

        # Build a transformer encoder
        self.transformer_encoder = TransformerEncoderBuilder.from_kwargs(
            n_layers=1,
            n_heads=n_head,
            query_dimensions=d_model // n_head,
            value_dimensions=d_model // n_head,
            feed_forward_dimensions=d_model,
            attention_type="full",
            activation="gelu",
        ).get()

        self.transformer_decoder = TransformerDecoderBuilder.from_kwargs(
            n_layers=1,
            n_heads=n_head,
            query_dimensions=d_model // n_head,
            value_dimensions=d_model // n_head,
            feed_forward_dimensions=d_model,
            self_attention_type="full",
            cross_attention_type="full",
            activation="gelu",
        ).get()

        self.encoders = nn.ModuleList([nn.Sequential(nn.Linear(self.z_dim, self.z_dim), nn.ReLU(), nn.Linear(self.z_dim, self.z_dim), nn.ReLU()) for _ in range(self.max_obj_num)])

        # self.mlp_bbox = nn.Sequential(
        #     nn.Linear(512, 512), nn.ReLU(),  # (512, 512)
        #     nn.Linear(512, self.inst_latent_len))  # 512
        # self.mlp_comp = nn.Sequential(
        #     nn.Linear(512, 128), nn.ReLU(),  # (512, 128)
        #     nn.Linear(128, 1))  # (128, 1)

    def forward(self, latent_z, ids, max_len=7):
        obj_feats = [self.empty_token_embedding(ids)[:, None]]  # [(bs,1,32)]

        for idx in range(self.max_obj_num):
            X = torch.cat(obj_feats, dim=1)
            # if idx > 0:
            #     X = X.detach()
            X = self.transformer_encoder(X, length_mask=None)  # 得到Fk，作为key, value
            last_feat = self.transformer_decoder(latent_z, X)  # latent_z (bs,1,32) 作为query
            obj_feats.append(self.encoders[idx](last_feat))

        obj_feats = torch.cat(obj_feats[1:], dim=1)[:, :max_len]
        # box_feat = self.mlp_bbox(obj_feats)
        # completenesss_feat = self.mlp_comp(obj_feats)
        return obj_feats.unsqueeze(0)  # (1,7,bs,32) -> (1,bs,7,32)  (1,7,512)->(1,1,7,512)

