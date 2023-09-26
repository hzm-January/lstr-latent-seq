import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Latent_Encoder(nn.Module):
    def __init__(self, cfg, optim_spec=None, device='cuda'):
        '''
        Encode scene priors from embeddings
        :param cfg: configuration file.
        :param optim_spec: optimizer parameters.
        '''
        super(Latent_Encoder, self).__init__()
        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec
        self.device = device

        '''Network'''
        feature_size = 64
        self.z_dim = cfg.config.data.z_dim

        self.feature_extractor = models.resnet18(pretrained=True)
        # if cfg.config.model.generator.arch.latent_encode.freeze_bn:
        #     FrozenBatchNorm2d.freeze(self.feature_extractor)

        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_extractor.fc = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, feature_size), nn.ReLU())

        self.mean_fc = nn.Sequential(nn.Linear(feature_size, 64), nn.ReLU(),
                                     nn.Linear(64, self.z_dim))
        self.logstd_fc = nn.Sequential(nn.Linear(feature_size, 64), nn.ReLU(),
                                       nn.Linear(64, self.z_dim))

    def forward(self, img):
        n_batch, n_view, n_channel, n_width, n_height = img.shape
        img = img.view(n_batch * n_view, n_channel, n_width, n_height)
        feature = self.feature_extractor(img)
        feature = feature.view(n_batch, n_view, -1)
        feature = torch.mean(feature, dim=1, keepdim=True)
        mean = self.mean_fc(feature)
        logstd = self.logstd_fc(feature)
        q_z = dist.Normal(mean, torch.exp(logstd))
        return q_z


class Latent_Embedding(nn.Module):
    def __init__(self, flag, device='cuda'):  # flag = true train, false test
        '''
        Encode scene priors from embeddings
        :param cfg: configuration file.
        :param optim_spec: optimizer parameters.
        '''
        super(Latent_Embedding, self).__init__()

        '''Optimizer parameters used in training'''
        self.device = device
        self.flag = 'train' if flag else 'test'
        '''Network params'''
        n_modes = 100  # cfg.config.data.n_modes
        embed_dim = 32  # 32  # 240 # 512  # cfg.config.data.z_dim
        n_samples = 7000
        self.weight_embedding = nn.ModuleDict()
        # for lane_num in range(10):
        for mode in ['train', 'test']:
            self.weight_embedding["split_" + mode] = nn.Embedding(n_samples, n_modes).requires_grad_(True)  # n_modes 100

        init_main_modes = torch.randn(n_modes, embed_dim)  # (100,32)
        # project main modes to the surface of a hyper-sphere.
        init_main_modes = F.normalize(init_main_modes, p=2, dim=-1)
        self.main_modes = nn.Parameter(init_main_modes, requires_grad=False)

    def forward(self, idx):
        '''Obtain latent codes for generation'''
        '''Normalize mode weights'''
        mode_weights = self.weight_embedding["split_" + self.flag](idx.cuda())
        mode_weights = mode_weights.softmax(dim=-1)
        # mode_weights = F.gumbel_softmax(mode_weights, dim=-1)

        '''Sample main modes'''
        latent_z = torch.mm(mode_weights, self.main_modes)  # mode_weights(bs,10)  main_modes(10,240)
        latent_z = F.normalize(latent_z, p=2, dim=-1)  # latent_z (bs,240)
        return latent_z.unsqueeze(1)  # (bs,1,240)
