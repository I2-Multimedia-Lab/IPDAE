import os
import multiprocessing
import numpy as np
import pandas as pd

import torch.nn as nn
import torch
import torch.nn.functional as F

from pytorch3d.ops.knn import _KNN, knn_gather, knn_points
from tqdm import tqdm
from pyntcloud import PyntCloud
from plyfile import PlyData

import octree_np

OCTREE_BPP_DICT = {
    1024:0.07,
    512:0.125,
    256:0.25,
    128:0.5,
    64:1.0,
}

def read_point_cloud(filepath):
    plydata = PlyData.read(filepath)
    try:
        pc = np.array(np.transpose(np.stack((plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z'])))).astype(np.float32)        
    except:
        pc = np.array(np.transpose(np.stack((plydata['vertex']['X'],plydata['vertex']['Y'],plydata['vertex']['Z'])))).astype(np.float32)        
    return pc

def read_point_clouds(file_path_list):
    print('loading point clouds...')
    with multiprocessing.Pool() as p:
        pcs = np.array(list(tqdm(p.imap(read_point_cloud, file_path_list, 32), total=len(file_path_list))))
    return np.array(pcs)

def save_point_cloud(pc, filename, path='./viewing/'):
    points = pd.DataFrame(pc, columns=['x', 'y', 'z'])
    cloud = PyntCloud(points)
    cloud.to_file(os.path.join(path, filename))



# NORMLIZE
def normalize(pc, margin=0.01):
    # pc: (1, N, 3), one point cloud
    # margin: rescaling pc to [0+margin, 1-margin]
    device = pc.device

    x, y, z = pc[0, :, 0], pc[0, :, 1], pc[0, :, 2]
    center = torch.Tensor([(x.max()+x.min())/2, (y.max()+y.min())/2, (z.max()+z.min())/2]).to(device)
    longest = torch.max(torch.Tensor([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()])).to(device)

    pc = pc - center
    pc = pc * (1-margin) / longest
    pc = pc + 0.5
    
    return pc, center, longest

def denormalize(pc, cetner, longest, margin=0.01):
    pc = pc - 0.5
    pc = pc * longest / (1-margin)
    pc = pc + cetner
    return pc

def n_scale_batch(batch_pc, margin=0.01):

    device = batch_pc.device
    B, S, _ = batch_pc.shape

    x, y, z = batch_pc[:, :, 0], batch_pc[:, :, 1], batch_pc[:, :, 2]
    x_max, x_min, y_max, y_min, z_max, z_min = x.max(dim=1)[0], x.min(dim=1)[0], y.max(dim=1)[0], y.min(dim=1)[0], z.max(dim=1)[0], z.min(dim=1)[0]
    x_max, x_min, y_max, y_min, z_max, z_min = x_max.unsqueeze(-1), x_min.unsqueeze(-1), y_max.unsqueeze(-1), y_min.unsqueeze(-1), z_max.unsqueeze(-1), z_min.unsqueeze(-1)
    
    #center = torch.cat([(x_max+x_min)/2, (y_max+y_min)/2, (z_max+z_min)/2], dim=1).to(device)
    longest = torch.max(torch.cat([x_max-x_min, y_max-y_min, z_max-z_min], dim=1), dim=1)[0].to(device)

    scaling = (1-margin) / longest

    #batch_pc = batch_pc - center.view(B, 1, 3)
    
    batch_pc = batch_pc * scaling.view(B, 1, 1)
    #batch_pc = batch_pc + 0.5

    return batch_pc, scaling

def d_n_scale_batch(batch_pc, scaling):
    device = batch_pc.device
    B, S, _ = batch_pc.shape
    #batch_pc = batch_pc - 0.5
    batch_pc = batch_pc / scaling.view(B, 1, 1)
    #batch_pc = batch_pc + center.view(B, 1, 3)
    return batch_pc

# POINTNET
class PointNet(nn.Module):
    def __init__(self, in_channel, mlps, relu, bn):
        super(PointNet, self).__init__()

        mlps.insert(0, in_channel)
        self.mlp_Modules = nn.ModuleList()
        for i in range(len(mlps) - 1):
            if relu[i]:
                if bn:
                    mlp_Module = nn.Sequential(
                        nn.Conv2d(mlps[i], mlps[i+1], 1),
                        nn.BatchNorm2d(mlps[i+1]),
                        nn.ReLU(),
                        )
                else:
                    mlp_Module = nn.Sequential(
                        nn.Conv2d(mlps[i], mlps[i+1], 1),
                        nn.ReLU(),
                        )
            else:
                mlp_Module = nn.Sequential(
                    nn.Conv2d(mlps[i], mlps[i+1], 1),
                    )
            self.mlp_Modules.append(mlp_Module)


    def forward(self, points):
        """
        Input:
            points: input points position data, [B, C, N]
        Return:
            points: feature data, [B, D]
        """
        
        points = points.unsqueeze(-1) # [B, C, N, 1]
        
        for m in self.mlp_Modules:
            points = m(points)
        # [B, D, N, 1]
        
        #points_np = points.detach().cpu().numpy()
        #np.save('./npys/ae_pn_feature.npy', points_np)

        points = torch.max(points, 2)[0]    # [B, D, 1]
        points = points.squeeze(-1) # [B, D] 

        return points

class SetAbstraction(nn.Module):
    def __init__(self, npoint, K, in_channel, mlp, bn=False, finalRelu=True):
        super(SetAbstraction, self).__init__()
        self.npoint = npoint
        self.K = K
        self.bn = bn
        self.finalRelu = finalRelu

        if self.bn:
            self.bn0 = nn.BatchNorm2d(mlp[0])
            self.bn1 = nn.BatchNorm2d(mlp[1])
            self.bn2 = nn.BatchNorm2d(mlp[2])

        self.conv0 = nn.Conv2d(in_channel+3, mlp[0], 1)
        self.conv1 = nn.Conv2d(mlp[0], mlp[1], 1)
        self.conv2 = nn.Conv2d(mlp[1], mlp[2], 1)


    def forward(self, xyz):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # 转置
        xyz = xyz.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        K = self.K
    
        # 使用farthest point sample从点列中采样出S个点
        if S == N:
            new_xyz = xyz
        else:
            new_xyz = index_points(xyz, farthest_point_sample_batch(xyz, S))
        #dist, group_idx = self.knn(xyz, new_xyz)
        
        #print('group_idx:', group_idx.size())
        #print(group_idx)
        #grouped_xyz = index_points(xyz, group_idx)
        dists, idx, grouped_xyz = knn_points(new_xyz, xyz, K=self.K, return_nn=True)
        grouped_xyz -= new_xyz.view(B, S, 1, C)

        # 接下来将分组过后的点集计算特征值
        grouped_points = grouped_xyz

        grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]

        grouped_points = F.relu(self.bn0(self.conv0(grouped_points))) if self.bn else F.relu(self.conv0(grouped_points))
        grouped_points = F.relu(self.bn1(self.conv1(grouped_points))) if self.bn else F.relu(self.conv1(grouped_points))

        grouped_points = self.conv2(grouped_points)
        if self.bn:
            grouped_points = self.bn2(grouped_points)
        if self.finalRelu:
            grouped_points = F.relu(grouped_points)

        new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]

        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points

class CMLP(nn.Module):
    def __init__(self, in_channel, mlps, relu, bn):
        super(CMLP, self).__init__()
        self.bn = bn

        mlps.insert(0, in_channel)
        self.Mlp_Modules = nn.ModuleList()
        for i in range(len(mlps) - 1):
            if relu[i]:
                if bn:
                    mlp_Module = nn.Sequential(
                        nn.Conv2d(mlps[i], mlps[i+1], 1),
                        nn.BatchNorm2d(mlps[i+1]),
                        nn.ReLU(),
                        )
                else:
                    mlp_Module = nn.Sequential(
                        nn.Conv2d(mlps[i], mlps[i+1], 1),
                        nn.ReLU(),
                        )
            else:
                mlp_Module = nn.Sequential(
                    nn.Conv2d(mlps[i], mlps[i+1], 1),
                    )
            self.Mlp_Modules.append(mlp_Module)


    def forward(self, points):
        """
        Input:
            points: input points position data, [B, C, N]
        Return:
            points: feature data, [B, D']
        """
        B, C, N = points.shape
        points = points.unsqueeze(-1)
        # points B, C, N, 1
        
        points_mx_ls = []
        for m in self.Mlp_Modules:
            points = m(points)
            points_mx_ls.append(torch.max(points, 2)[0])

        # points_mx_ls [n_mlp * (B, D, 1)]
    
        points = torch.cat(points_mx_ls, dim=1).squeeze(-1)
        # [B, D*n_mlp]

        return points

class MLP(nn.Module):
    def __init__(self, in_channel, mlps, relu, bn):
        super(MLP, self).__init__()

        mlps.insert(0, in_channel)
        self.mlp_Modules = nn.ModuleList()
        for i in range(len(mlps) - 1):
            if relu[i]:
                if bn:
                    mlp_Module = nn.Sequential(
                        nn.Conv2d(mlps[i], mlps[i+1], 1),
                        nn.BatchNorm2d(mlps[i+1]),
                        nn.ReLU(),
                        )
                else:
                    mlp_Module = nn.Sequential(
                        nn.Conv2d(mlps[i], mlps[i+1], 1),
                        nn.ReLU(),
                        )
            else:
                mlp_Module = nn.Sequential(
                    nn.Conv2d(mlps[i], mlps[i+1], 1),
                    )
            self.mlp_Modules.append(mlp_Module)


    def forward(self, points):
        """
        Input:
            points: input points position data, [B, C, N]
        Return:
            points: feature data, [B, D, N]
        """
        
        points = points.unsqueeze(-1) # [B, C, N, 1]
        
        for m in self.mlp_Modules:
            points = m(points)
        # [B, D, N, 1]
        
        points = points.squeeze(-1) # [B, D, N] 

        return points


# SAMPLING
def farthest_point_sample_batch(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] or [B, S, K]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    #print('points size:', points.size(), 'idx size:', idx.size())
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    # view_shape == [B, S, K]
    view_shape[1:] = [1] * (len(view_shape) - 1)
    # view_shape == [B, 1, 1]
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    # repeat_shape == [1, S, K]
    #print('points:', points.size(), ', idx:', idx.size(), ', view_shape:', view_shape)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    # batch_indices == tensor[0, 1, ..., B-1]
    #print('batch_indices:', batch_indices.size())
    batch_indices = batch_indices.view(view_shape)
    # batch_indices size == [B, 1, 1]
    #print('after view batch_indices:', batch_indices.size())
    batch_indices = batch_indices.repeat(repeat_shape)
    # batch_indices size == [B, S, K]
    new_points = points[batch_indices, idx.long(), :]
    return new_points

def random_point_sample_batch(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        new_xyz: sampled pointcloud index, [B, npoint, 3]
    """
    device = xyz.device
    B, N, C = xyz.shape
    
    idx = torch.Tensor([True if i < npoint else False for i in range(N)]).to(device)
    idx = idx[torch.randperm(idx.size(0))].bool()

    return xyz[:, idx, :]


# NP OCTREE
def encode_sampled_np(sampled_xyz, scale, N, min_bpp):
    codebits = 0
    codes, depthes = [], []
    for i in range(sampled_xyz.shape[0]):
        pc = sampled_xyz[i]
        DEPTH = 0
        while True:
            DEPTH += 1
            code = octree_np.encode(pc, scale, DEPTH)
            bpp = round(code.shape[0]/N, 5)
            pc_rec = octree_np.getDecodeFromPc(pc, scale, DEPTH)

            if bpp > min_bpp and pc_rec.shape == pc.shape:
                break
        #print(DEPTH)
        codebits += code.shape[0]
        codes.append(code)
        depthes.append(DEPTH)
        #print(depthes)
    return codes, codebits

def encode_sampled_np_depth(sampled_xyz, scale, N, depth):
    codebits = 0
    codes, depthes = [], []
    for i in range(sampled_xyz.shape[0]):
        pc = sampled_xyz[i]
        DEPTH = depth
        while True:
            code = octree_np.encode(pc, scale, DEPTH)
            bpp = round(code.shape[0]/N, 5)
            pc_rec = octree_np.getDecodeFromPc(pc, scale, DEPTH)

            if pc_rec.shape == pc.shape:
                break
            DEPTH += 1
        #print(DEPTH)
        codebits += code.shape[0]
        codes.append(code)
        depthes.append(DEPTH)
        #print(depthes)
    return codes, codebits

def decode_sampled_np(codes, scale):
    rec_sampled_xyz = []
    for i in range(len(codes)):
        rec_sampled_xyz.append(octree_np.decode(codes[i], scale))
    return rec_sampled_xyz

def get_decode_from_pc(sampled_xyz, scale, depth):
    rec_sampled_xyz = []
    for i in range(sampled_xyz.shape[0]):
        pc = sampled_xyz[i]
        rec_sampled_xyz.append(octree_np.getDecodeFromPc(pc, scale, depth))
    return rec_sampled_xyz
    
# PMF
def estimate_bits_from_pmf(pmf, sym):
    L = pmf.shape[-1]
    pmf = pmf.reshape(-1, L)
    sym = sym.reshape(-1, 1)
    assert pmf.shape[0] == sym.shape[0]
    relevant_probabilities = torch.gather(pmf, dim=1, index=sym)
    # gather: 以sym为index从pmf中找出对应的数
    # relevant_probabilities shape: [B, 1]

    # torch.clamp(input, min=None, max=None), Clamps all elements in input into the range [min, max].
    bits = torch.sum(-torch.log2(relevant_probabilities.clamp(min=1e-3)))
    return bits

def pmf_to_cdf(pmf):
    cdf = pmf.cumsum(dim=-1)
    #print(cdf.shape)
    spatial_dimensions = pmf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    # On GPU, softmax followed by cumsum can lead to the final value being 
    # slightly bigger than 1, so we clamp.
    cdf_with_0 = cdf_with_0.clamp(max=1.)
    return cdf_with_0

def binary_array_to_byte_array(a):
    byte_stream = bytearray()
    for i in range(0, len(a), 8):
        byte_stream.append(int(''.join([str(e) for e in a[i:i+8]]), 2))
    return byte_stream

def byte_array_to_binary_array(byte_stream):
    int_values = [x for x in byte_stream]
    binary_array = []
    for i in range(0, len(int_values)):
        binary_array.append(list(f'{int_values[i]:08b}'))
    binary_array = np.array(binary_array, dtype=np.int).flatten()
    return binary_array