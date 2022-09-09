from pickle import TRUE
import torch.nn as nn
import torch
from torch.nn import parameter
import torch.nn.functional as F

from pytorch3d.loss import chamfer_distance
from torch.nn.modules.container import T
from pn_kit import PointNet, SetAbstraction, MLP

from geomloss import SamplesLoss


class AE(nn.Module):
    def __init__(self, K, k, d, L):
        super(AE, self).__init__()

        self.sa = SetAbstraction(npoint=K, K=16, in_channel=0, mlp=[32, 64, 128], bn=False)
        self.pn = PointNet(in_channel=3+128, mlps=[128, 256, 512, d], relu=[True, True, True, False], bn=False)

        self.inv_pool = nn.Sequential(
            nn.Linear(d, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, k*128),
            nn.ReLU(),
        )
        self.inv_mlp = MLP(in_channel=d+128, mlps=[128, 64, 32, 3], relu=[True, True, True, False], bn=False)

        self.K = K
        self.k = k
        self.L = L
        self.quantize = STEQuantize.apply
    
    def forward(self, xyz):
        BS = xyz.shape[0]
        # ENCODE
        xyz = xyz.transpose(2, 1)
        _, xyz_feature = self.sa(xyz)
        latent = self.pn(torch.cat((xyz, xyz_feature), dim=1))

        # QUANTIZATION
        #spread = self.L - 1 + 0.2
        spread = self.L - 0.2
        latent = torch.sigmoid(latent) * spread - spread / 2
        latent_quantized_trans = self.quantize(latent)
        #print(latent_quantized)

        # DECODE
        linear_output = self.inv_pool(latent_quantized_trans)
        linear_output = linear_output.view(BS, -1, self.k)
        latent_quantized = latent_quantized_trans.unsqueeze(-1).repeat((1, 1, self.k))
        mlp_input = torch.cat((linear_output, latent_quantized), dim=1)
        new_xyz = self.inv_mlp(mlp_input)
        new_xyz = new_xyz.transpose(2, 1)

        return new_xyz, latent, latent_quantized_trans

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pc_pred, pc_target, fbpp, λ):
        '''
        Input:
            pred: reconstructed point cloud (B, N, 3)
            target: origin point cloud (B, CxN, 3)
        '''
        d, loss_normals = chamfer_distance(pc_pred, pc_target)
        r = fbpp
        loss =  d + λ * r
        return loss

class STEQuantize(torch.autograd.Function):
    """Straight-Through Estimator for Quantization.

    Forward pass implements quantization by rounding to integers,
    backward pass is set to gradients of the identity function.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.round()

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs

class ConditionalProbabilityModel(nn.Module):
    def __init__(self, L, d):
        super(ConditionalProbabilityModel, self).__init__()
        self.L = L
        self.d = d

        # We predict a probility mess for each element in bottleneck
        output_channels = self.d * L
        
        self.model_pn = PointNet(in_channel=3, mlps=[64, 128, 256], relu=[True, True, True], bn=False)
        self.model_mlp = nn.Sequential(
            nn.Conv2d(3+256, 512, 1),
            #nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1),
            #nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, d*self.L, 1),
        )

    def forward(self, sampled_xyz):
        ''' sampled_xyz: B, S, 3 | latent_quantized: BS, d '''
        # NORMALIZATION ?
        B, S, C = sampled_xyz.shape

        feature = self.model_pn(sampled_xyz.transpose(1, 2)) # feature: B, d

        # mlp_input: B, d+3, S, 1 | output: B, d*L, S, 1
        mlp_input = torch.cat((sampled_xyz, feature.repeat((1, S)).view(B, S, -1)), dim=2)
        mlp_input = mlp_input.unsqueeze(-1).transpose(1, 2)
        output = self.model_mlp(mlp_input)
        output = output.transpose(1, 2).view(B, S, self.d, self.L)

        pmf = F.softmax(output, dim=3)
        # pmf has shape (B, S, d, L)
    
        return pmf



