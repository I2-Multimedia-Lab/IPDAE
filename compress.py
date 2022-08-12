import os
import time
import subprocess
from glob import glob

import numpy as np
import torch
from tqdm import tqdm
from pytorch3d.ops.knn import _KNN, knn_gather, knn_points
import torchac

import pc_io
from plyfile import PlyData
import pn_kit
import global_cd_net_mlpd as net

torch.manual_seed(1)
np.random.seed(1)

ALPHA = 2
K = 1024
d = 16   # Bottleneck Size
L = 7   # Quantization Level
MIN_SAMPLED_BPP = 0.07

N0 = 1024

'''
INPUT_PATH = '/home/yk/Projects/DataSets/ShapeNet_pc_01_2048p/test/*.ply'
COMPRESSED_PATH = f'./data/ShapeNet_Compressed_N_ALPHA{ALPHA}_K{K}_d{d}_L{L}_Lambda{1e-6}/'
'''

INPUT_PATH = '/mnt/hdd/datasets_yk/ModelNet40_pc_01_8192p/**/test/*.ply'
COMPRESSED_PATH = f'./data/ModelNet_Compressed_N_ALPHA{ALPHA}_K{K}_d{d}_L{L}_Lambda{1e-6}/'

'''
INPUT_PATH = '/mnt/hdd/datasets_yk/ModelNet40_pc_01_8192p/chair/chair_0930.ply'
COMPRESSED_PATH = f'./data/ModelNet_Compressed_Teaser/'
'''
'''
INPUT_PATH = '/home/yk/Projects/DataSets/Stanford3d_pc/Area_1/*.ply'
COMPRESSED_PATH = f'./data/Stanford3d_Area_1_Compressed_N_ALPHA{ALPHA}_K{K}_d{d}_L{L}_Lambda{1e-6}/'
'''

'''
INPUT_PATH = '/home/yk/Projects/DataSets/KITTI_pc/sequences/00/*.ply'
COMPRESSED_PATH = f'./data/KITTI_seq_00_Compressed_N_ALPHA{ALPHA}_K{K}_d{d}_L{L}_Lambda{1e-6}/'
'''

INPUT_PATH = '/mnt/hdd/datasets_yk/msft/andrew9/ply/frame0000.ply'
COMPRESSED_PATH = f'./data/andrew9_Compressed_N_ALPHA{ALPHA}_K{K}_d{d}_L{L}_Lambda{1e-6}/'

NET_PATH, PROB_PATH = f'./model/ModelNet_N{8192}_ALPHA{2}_K{K}_d{d}_L{L}_Lambda{1e-6}_MIN_SAMPLED_BPP{MIN_SAMPLED_BPP}_N0_normalized_CD/ae_s78500.pkl', f'./model/ModelNet_N{8192}_ALPHA{2}_K{K}_d{d}_L{L}_Lambda{1e-6}_MIN_SAMPLED_BPP{MIN_SAMPLED_BPP}_N0_normalized_CD/prob_s78500.pkl'

B = 1 # Compress 1 Point Cloud Each Time !unchangable

# CREATE COMPRESSED PATH
if not os.path.exists(COMPRESSED_PATH):
    os.makedirs(COMPRESSED_PATH)

# READ INPUT FILES
files = np.array(glob(INPUT_PATH, recursive=True))

#points = pc_io.load_points(files, p_min, p_max)
files_cat = np.array([os.path.split(os.path.split(x)[0])[1] for x in files])
#points_train = points[files_cat == 'train']
#points_test = points[files_cat == 'test']
#files = files[files_cat == 'test']

filenames = np.array([os.path.split(x)[1] for x in files])

ae = torch.load(NET_PATH).cuda().eval()
prob = torch.load(PROB_PATH).cuda().eval()

def KNN_Patching(batch_x, sampled_xyz, K):
    dist, group_idx, grouped_xyz = knn_points(sampled_xyz, batch_x, K=K, return_nn=True)
    grouped_xyz -= sampled_xyz.view(B, S, 1, 3)
    x_patches = grouped_xyz.view(B*S, K, 3)
    return x_patches

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

def normalize(pc, margin=0.01):
    # pc: (1, N, 3), one point cloud
    # margin: rescaling pc to [0+margin, 1-margin]
    x, y, z = pc[0, :, 0], pc[0, :, 1], pc[0, :, 2]
    center = torch.Tensor([(x.max()+x.min())/2, (y.max()+y.min())/2, (z.max()+z.min())/2]).cuda()
    longest = torch.max(torch.Tensor([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()])).cuda()

    pc = pc - center
    pc = pc * (1-margin) / longest
    pc = pc + 0.5
    
    return pc, center, longest


start_time = time.time()
# DO THE COMPRESS
with torch.no_grad():
    for i in tqdm(range(filenames.shape[0])):
        # GET 1 POINT CLOUD
        #pc = pc_io.load_points([files[i]], p_min, p_max, processbar=False)
        plydata = PlyData.read(files[i])
        try:
            pc = np.array(np.transpose(np.stack((plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z'])))).astype(np.float32)        
        except:
            pc = np.array(np.transpose(np.stack((plydata['vertex']['X'],plydata['vertex']['Y'],plydata['vertex']['Z'])))).astype(np.float32)        
        pc = torch.Tensor(pc).cuda()
        pc = pc.unsqueeze(0)
        #print(pc.shape)
        
        # normalize our point cloud.
        # remove the pc to (0.5, 0.5)
        # scale the size to (0+margin, 1-margin)
        pc, center, longest = normalize(pc, margin=0.01)

        N = pc.shape[1]
        S = (int)(N * ALPHA // K)
        k = (int)(K // ALPHA)

        # SAMPLING
        sampled_xyz = pn_kit.index_points(pc, pn_kit.farthest_point_sample_batch(pc, S))
        # OCTREE ENCODE
        octree_codes, sampled_bits = pn_kit.encode_sampled_np(sampled_xyz.detach().cpu().numpy(), scale=1, N=N, min_bpp=MIN_SAMPLED_BPP)
        # OCTREE DECODE
        rec_sampled_xyz = pn_kit.decode_sampled_np(octree_codes, scale=1)
        rec_sampled_xyz = torch.Tensor(rec_sampled_xyz).cuda()
        assert rec_sampled_xyz.shape == sampled_xyz.shape

        # DIVIDE PATCH BY KNN
        x_patches = KNN_Patching(pc, rec_sampled_xyz, K)
        # DO THE ANALYSIS TRANSFORM FOR PATCHES
        x_patches = x_patches.transpose(1, 2)
        x_patches = x_patches * ((N / N0) ** (1/3))
        # FEED X_PATCHES ONE BY ONE
        
        #_, patch_features = ae.sa(x_patches)
        patch_features = []
        for j in range(S):
            _, patch_feature = ae.sa(x_patches[j].view(1, 3, K))
            patch_features.append(patch_feature.cpu())
        patch_features = torch.cat(patch_features)
        
        #latent = ae.pn(torch.cat((x_patches, patch_features), dim=1))
        latent = []
        for j in range(S):
            latent.append(ae.pn(torch.cat((x_patches[j].unsqueeze(0), patch_features[j].cuda().unsqueeze(0)), dim=1)).cpu())
        latent = torch.cat(latent)
        
        # QUANTIZATION
        spread = ae.L - 0.2
        latent = torch.sigmoid(latent) * spread - spread / 2
        latent_quantized = ae.quantize(latent)
        # LATENT_QUANTIZED SHAPE: (BS, d)
        
        # COMPUTE PROB
        pmf = prob(rec_sampled_xyz)

        # ARITHMETHIC ENCODE Y HAT
        cdf = pn_kit.pmf_to_cdf(pmf)
        cdf = cdf.cpu()
        #print(pmf[0,0,0])
        #print(cdf.shape)
        #print(cdf[0,0,0])
        n_latent_quantized = latent_quantized.view(B, S, -1).to(torch.int16).cpu() + L // 2
        byte_stream = torchac.encode_float_cdf(cdf, n_latent_quantized, check_input_bounds=True)

        # * WRITE AE TO FILE
        with open(COMPRESSED_PATH + filenames[i] + '.p.bin', 'wb') as fout:
            fout.write(byte_stream)
        # READ AND CHECK
        '''
        with open(COMPRESSED_PATH + filenames[i] + '.p.bin', 'rb') as fin:
            byte_stream = fin.read()
        assert (torchac.decode_float_cdf(cdf, byte_stream) - L // 2).float().view(B*S, -1).cuda().equal(latent_quantized)
        '''

        # * SAVE OCTREE CODE TO FILE
        octree_code = octree_codes[0]
        byte_stream = binary_array_to_byte_array(octree_code)
        with open(COMPRESSED_PATH + filenames[i] + '.s.bin', 'wb') as fout:
            fout.write(byte_stream)
        # READ AND CHECK
        '''
        with open(COMPRESSED_PATH + filenames[i] + '.s.bin', 'rb') as fin:
            byte_stream = fin.read()
        assert np.all(byte_array_to_binary_array(byte_stream) == octree_code)
        '''
        
        # * SAVE CENTER AND SCALE TO FILE
        arr = np.zeros((4))
        arr[:3] = center.detach().cpu().numpy().flatten()
        arr[3] = longest.detach().cpu().numpy()
        arr.astype(np.float32).tofile(COMPRESSED_PATH + filenames[i] + '.c.bin')

t = time.time() - start_time
n = t / len(filenames)
print(f"Done! Execution time: {round(t, 3)}s, i.e., {round(n, 5)}s per point cloud.")