import math
import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from utils import *

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

# Swish activation function
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class WeightStandardizedConv1d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class ResidualConvBlock(nn.Module):
    def __init__(
        self, inc: int, outc: int, kernel_size: int, stride=1, gn=8):
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = inc==outc
        self.ks = kernel_size
        self.conv = nn.Sequential(
            WeightStandardizedConv1d(inc, outc, self.ks, stride, get_padding(self.ks)),
            nn.GroupNorm(gn, outc),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
        if self.same_channels:
            out = (x + x1)/2
        else:
            out = x1
        return out

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gn=8, factor=2):
        super(UnetDown, self).__init__()
        self.pool = nn.MaxPool1d(factor)
        self.layer = ResidualConvBlock(in_channels, out_channels, kernel_size, gn=gn)
    
    def forward(self, x):
        x = self.layer(x)
        x = self.pool(x)
        return x

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gn=8, factor=2):
        super(UnetUp, self).__init__()
        self.pool = nn.Upsample(scale_factor=factor, mode='nearest')
        self.layer = ResidualConvBlock(in_channels, out_channels, kernel_size, gn=gn)

    def forward(self, x):
        x = self.pool(x)
        x = self.layer(x)
        return x

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn. PReLU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ConditionalUNet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, z_dim=512, n_classes=10):
        super(ConditionalUNet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.z_dim = z_dim
        
        # Embedding

        self.noise_embed = nn.Embedding(10, n_feat)

        self.d1_out = n_feat*1
        self.d2_out = n_feat*2
        self.d3_out = n_feat*3
        self.d4_out = n_feat*4

        self.latent_dim = 256

        self.u1_out = n_feat
        self.u2_out = n_feat
        self.u3_out = n_feat
        self.u4_out = in_channels

        self.sin_emb = SinusoidalPosEmb(n_feat)
        # self.timeembed1 = EmbedFC(n_feat, self.u1_out)
        # self.timeembed2 = EmbedFC(n_feat, self.u2_out)
        # self.timeembed3 = EmbedFC(n_feat, self.u3_out)
        # self.contextembed1 = EmbedFC(n_classes, self.u1_out)
        self.contextembed2 = EmbedFC(n_classes, self.u2_out)
        self.contextembed3 = EmbedFC(n_classes, self.u3_out)

        self.down1 = UnetDown(in_channels, self.d1_out, 1, gn=8, factor=2)
        self.down2 = UnetDown(self.d1_out, self.d2_out, 1, gn=8, factor=2)
        self.down3 = UnetDown(self.d2_out, self.d3_out, 1, gn=8, factor=2)

        self.up2 = UnetUp(self.d3_out, self.u2_out, 1, gn=8, factor=2)
        self.up3 = UnetUp(self.u2_out+self.d2_out, self.u3_out, 1, gn=8, factor=2)
        self.up4 = UnetUp(self.u3_out+self.d1_out, self.u4_out, 1, gn=8, factor=2)
        self.out = nn.Conv1d(self.u4_out+in_channels, in_channels, 1)

    def forward(self, x, t):
        down1 = self.down1(x) # 2000 -> 1000
        down2 = self.down2(down1) # 1000 -> 500
        down3 = self.down3(down2) # 500 -> 250

        temb = self.sin_emb(t).view(-1, self.n_feat, 1) # [b, n_feat, 1]      

        up1 = self.up2(down3) # 250 -> 500
        up2 = self.up3(torch.cat([up1+temb, down2], 1)) # 500 -> 1000
        up3 = self.up4(torch.cat([up2+temb, down1], 1)) # 1000 -> 2000
        out = self.out(torch.cat([up3, x], 1)) # 2000 -> 2000

        down = (down1, down2, down3)
        up = (up1, up2, up3)
        return out, down, up

class Encoder(nn.Module):
    def __init__(self, in_channels, dim=512):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.e1_out = dim
        self.e2_out = dim
        self.e3_out = dim

        self.down1 = UnetDown(in_channels, self.e1_out, 1, gn=8, factor=2)
        self.down2 = UnetDown(self.e1_out, self.e2_out, 1, gn=8, factor=2)
        self.down3 = UnetDown(self.e2_out, self.e3_out, 1, gn=8, factor=2)

        self.avg_pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.max_pooling = nn.AdaptiveMaxPool1d(output_size=1)
        self.act = nn.Tanh()

    def forward(self, x0):
        # Down sampling
        dn1 = self.down1(x0) # 2048 -> 1024
        dn2 = self.down2(dn1) # 1024 -> 512
        dn3 = self.down3(dn2) # 512 -> 256
        z = self.avg_pooling(dn3).view(-1, self.e3_out) # [b, features]
        down = (dn1, dn2, dn3)
        out = (down, z)
        return out

class Decoder(nn.Module):
    def __init__(self, in_channels, n_feat=256, encoder_dim=512, n_classes=13):
        super(Decoder, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.e1_out = encoder_dim
        self.e2_out = encoder_dim
        self.e3_out = encoder_dim
        self.d1_out = n_feat
        self.d2_out = n_feat*2
        self.d3_out = n_feat*3
        self.u1_out = n_feat
        self.u2_out = n_feat
        self.u3_out = n_feat
        self.u4_out = in_channels

        # self.sin_emb = SinusoidalPosEmb(n_feat)
        # self.timeembed1 = EmbedFC(n_feat, self.e3_out)
        # self.timeembed2 = EmbedFC(n_feat, self.u2_out)
        # self.timeembed3 = EmbedFC(n_feat, self.u3_out)
        # self.contextembed1 = EmbedFC(self.e3_out, self.e3_out)
        # self.contextembed2 = EmbedFC(self.e3_out, self.u2_out)
        # self.contextembed3 = EmbedFC(self.e3_out, self.u3_out)
        
        # Unet up sampling
        self.up2 = UnetUp(self.d3_out+self.e3_out, self.u2_out, 1, gn=8, factor=2)
        self.up3 = UnetUp(self.d2_out+self.u2_out, self.u3_out, 1, gn=8, factor=2)
        # self.up4 = UnetUp(self.d1_out+self.u3_out, self.u4_out, 1, 1, gn=8, factor=2)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(self.d1_out+self.u3_out+in_channels*2, in_channels, 1, 1, 0),
        )

        # self.out = nn.Conv1d(self.u4_out+in_channels, in_channels, 1)
        self.pool = nn.AvgPool1d(2)

    def forward(self, x0, encoder_out, diffusion_out):
        down, z = encoder_out
        dn1, dn2, dn3 = down
        x_hat, down2, up, t = diffusion_out
        dn11, dn22, dn33 = down2 # Contains features with degraded information

        # temb = self.sin_emb(t).view(-1, self.n_feat, 1) # [b, n_feat, 1]
        # temb1 = self.timeembed1(temb).view(-1, self.e3_out, 1) # [b, features]
        # temb2 = self.timeembed2(temb).view(-1, self.u2_out, 1) # [b, features]
        # temb3 = self.timeembed3(temb).view(-1, self.u3_out, 1) # [b, features]

        # embed context, time step
        # ct2 = self.contextembed2(z).view(-1, self.u2_out, 1) # [b, n_feat, 1]
        # ct3 = self.contextembed3(z).view(-1, self.u3_out, 1) # [b, n_feat, 1]

        # Up sampling
        # up22, up33, up44 = up
        # up2 = self.up2(dn33)
        up2 = self.up2(torch.cat([dn3, dn33.detach()], 1)) # 256 -> 512
        up3 = self.up3(torch.cat([up2, dn22.detach()], 1)) # 512 -> 1024
        out = self.up4(torch.cat([self.pool(x0), self.pool(x_hat.detach()), up3, dn11.detach()], 1)) # 1024 -> 2048
        # out = self.out(torch.cat([out, x0-x_hat], 1)) # 2048 -> 2048
        # out = self.out(out)
        return out

class DiffE(nn.Module):
    def __init__(self, encoder, decoder, fc):
        super(DiffE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.fc = fc

    def forward(self, x0, ddpm_out):
        encoder_out = self.encoder(x0)
        decoder_out = self.decoder(x0, encoder_out, ddpm_out)
        fc_out = self.fc(encoder_out[1])
        return decoder_out, fc_out
    
class DecoderNoDiff(nn.Module):
    def __init__(self, in_channels, n_feat=256, encoder_dim=512, n_classes=13):
        super(DecoderNoDiff, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.e1_out = encoder_dim
        self.e2_out = encoder_dim
        self.e3_out = encoder_dim
        self.u1_out = n_feat
        self.u2_out = n_feat
        self.u3_out = n_feat
        self.u4_out = n_feat

        self.sin_emb = SinusoidalPosEmb(n_feat)
        self.timeembed1 = EmbedFC(n_feat, self.e3_out)
        self.timeembed2 = EmbedFC(n_feat, self.u2_out)
        self.timeembed3 = EmbedFC(n_feat, self.u3_out)
        self.contextembed1 = EmbedFC(self.e3_out, self.e3_out)
        self.contextembed2 = EmbedFC(self.e3_out, self.u2_out)
        self.contextembed3 = EmbedFC(self.e3_out, self.u3_out)
        
        # Unet up sampling
        self.up2 = UnetUp(self.e3_out, self.u2_out, 1, gn=8, factor=2)
        self.up3 = UnetUp(self.e2_out+self.u2_out, self.u3_out, 1, gn=8, factor=2)
        # self.up4 = UnetUp(self.e1_out+self.u3_out, self.u4_out, 1, 1, gn=in_channels, factor=2, is_res=True)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(self.u3_out+self.e1_out+in_channels, in_channels, 1, 1, 0),
        )

        self.out = nn.Conv1d(self.u4_out, in_channels, 1)
        self.pool = nn.AvgPool1d(2)

    def forward(self, x0, x_hat, encoder_out, t):
        down, z = encoder_out
        dn1, dn2, dn3 = down
        tembd = self.sin_emb(t).view(-1, self.n_feat, 1) # [b, n_feat, 1]
        tembd1 = self.timeembed1(self.sin_emb(t)).view(-1, self.e3_out, 1) # [b, n_feat, 1]
        tembd2 = self.timeembed2(self.sin_emb(t)).view(-1, self.u2_out, 1) # [b, n_feat, 1]
        tembd3 = self.timeembed3(self.sin_emb(t)).view(-1, self.u3_out, 1) # [b, n_feat, 1]

        # Up sampling
        ddpm_loss = F.l1_loss(x0, x_hat, reduction='none')

        up2 = self.up2(dn3) # 256 -> 512
        up3 = self.up3(torch.cat([up2, dn2], 1)) # 512 -> 1024
        out = self.up4(torch.cat([self.pool(x0), self.pool(x_hat), up3, dn1], 1)) # 1024 -> 2048
        # out = self.out(torch.cat([out, x_hat], 1)) # 2048 -> 2048
        # out = self.out(out)
        return out

class LinearClassifier(nn.Module):
    def __init__(self, in_dim, latent_dim, emb_dim):
        super().__init__()
        self.linear_out = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=latent_dim),
            nn.GroupNorm(4, latent_dim),
            nn.PReLU(),
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.GroupNorm(4, latent_dim),
            nn.PReLU(),
            nn.Linear(in_features=latent_dim, out_features=emb_dim)
            )
        self.logsoft = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.linear_out(x)
        # x = self.logsoft(x)
        # x = self.softmax(x)
        return x

class DeepConvNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=256, dropoutRate=0.5):
        super(DeepConvNet, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5)),
            nn.Conv2d(25, 25, kernel_size=(Chans, 1)),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.9),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropoutRate)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5)),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.9),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropoutRate)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5)),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.9),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropoutRate)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5)),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.9),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropoutRate)
        )
        
        self.dense = nn.Sequential(
            nn.Linear(24200, nb_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x

class ShallowConvNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=1000, dropoutRate=0.5):
        super(ShallowConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, kernel_size=(1, 200))
        self.conv2 = nn.Conv2d(40, 40, kernel_size=(Chans, 1))


        self.batch_norm = nn.BatchNorm1d(256, eps=1e-05, momentum=0.9)
        self.activation1 = nn.Softplus()
        # self.pooling = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 7))
        self.pooling = nn.AvgPool1d(kernel_size=200)
        self.activation2 = nn.LogSigmoid()
        self.dropout = nn.Dropout(dropoutRate)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, nb_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = x.view(-1, 1, 64, 2000)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch_norm(x)
        x = self.activation1(x)
        x = self.pooling(x)
        x = self.activation2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    # assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    # beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    beta_t = cosine_beta_schedule(T, s = 0.008).float()
    #beta_t = sigmoid_beta_schedule(T).float()

    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    
    alphas_cumprod = torch.cumprod(alpha_t, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrta = torch.sqrt(alpha_t)
    one_minus_abar = 1 - alphabar_t
    sqrt_one_minus_abar = torch.sqrt(one_minus_abar)

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    posterior_variance = beta_t * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    posterior_log_variance_clipped =  torch.log(posterior_variance.clamp(min =1e-20))

    thing = sqrta * sqrt_one_minus_abar / beta_t

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        "posterior_variance": posterior_variance,
        "posterior_log_variance_clipped": posterior_log_variance_clipped,
        "thing": thing,
    }
def ddpm_gamma_schedules(theta_0, T):
    # beta_t = cosine_beta_schedule(T, s = 0.008).float()
    beta_t = (1e-2 - 1e-6) * torch.arange(0, T + 1, dtype=torch.float32) / T + 1e-6
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()
    sqrtabar_t = torch.sqrt(alphabar_t)
    theta_t = theta_0 * sqrtabar_t
    k_t = beta_t / (alpha_t * theta_0**2)
    kbar_t = torch.cumsum(k_t, dim=0)

    return {
        "theta_t_"+str(theta_0).replace('.', ''): theta_t,
        "kbar_t_"+str(theta_0).replace('.', ''): kbar_t,
        "sqrtabar_t_"+str(theta_0).replace('.', ''): sqrtabar_t,
    }

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1, gamma=False, g_val=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        self.gamma = gamma
        self.theta_01 = g_val
        
        # if gamma:
        #     for k, v in ddpm_gamma_schedules(self.theta_01, n_T).items():
        #         self.register_buffer(k, v)
        # else:
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, y, mode="train"):
        """
        this method is used in training, so samples t and noise randomly
        """
        if mode=="train":
            _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        else:
            _ts = torch.randint(0, 1, (x.shape[0],)).to(self.device)
        b,c,n = x.shape
        # if self.gamma:
        #     kbar_t = getattr(self, "kbar_t_"+str(self.theta_01).replace('.', ''))
        #     theta_t = getattr(self, "theta_t_"+str(self.theta_01).replace('.', ''))
        #     sqrtabar_t = getattr(self, "sqrtabar_t_"+str(self.theta_01).replace('.', ''))
            
        #     k = kbar_t[_ts]
        #     theta = theta_t[_ts]
        #     dists = [torch.distributions.Gamma(k[i], theta[i]) for i in range(b)]

        #     # sample from gamma distribution
        #     noise = torch.stack([dists[i].sample((c, n)) for i in range(b)], dim=0)

        #     kbar_t = kbar_t[_ts, None, None]
        #     theta_t = theta_t[_ts, None, None]
        #     sqrtabar_t = sqrtabar_t[_ts, None, None]
        #     # print(sqrtabar_t.shape, kbar_t.shape, theta_t.shape, noise.shape)
        #     x_t = sqrtabar_t*x + (noise - kbar_t*theta_t)
        #     target = (noise - kbar_t*theta_t)/torch.sqrt(1 - sqrtabar_t**2)
        # else:
        noise = torch.randn_like(x)  # eps ~ N(0, 1)
        target = noise
        x_t = self.sqrtab[_ts, None, None]*x + self.sqrtmab[_ts, None, None]*noise

        model_log_variance = self.posterior_log_variance_clipped
        model_variance = (0.5 * model_log_variance).exp()
        # thing = self.thing[_ts, None, None] * model_variance[_ts, None, None]

        # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.
        # print(x_t.shape, c.shape, _ts.shape)
        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(y)+self.drop_prob).to(self.device)
        times = _ts / self.n_T
        output, down, up = self.nn_model(x_t, times)
        # return MSE between added noise, and our predicted noise
        return output, down, up, noise, times

    def sample(self, n_sample, size, device, guide_w = 0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.arange(0,10).to(device) # context for us just cycles throught the mnist labels
        c_i = c_i.repeat(int(n_sample/c_i.shape[0]))

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. # makes second half of batch context free

        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store