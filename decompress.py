import os
from glob import glob
import time

import numpy as np
import torch
from tqdm import tqdm
from pytorch3d.ops.knn import _KNN, knn_gather, knn_points
import torchac

import pc_io
import pn_kit
import global_cd_net_mlpd as net

torch.manual_seed(1)
np.random.seed(1)

N0 = 1024
ALPHA = 2
DEVICE = 'cpu' # 'cpu' or 'cuda'

'''
DECOMPRESSED_PATH = f'./data/ShapeNet_Decompressed_N_ALPHA{ALPHA}_K{64}_d{16}_L{7}_Lambda{1e-6}/'
COMPRESSED_PATH = f'./data/ShapeNet_Compressed_N_ALPHA{ALPHA}_K{64}_d{16}_L{7}_Lambda{1e-6}/'
'''
'''
COMPRESSED_PATH = f'./data/MSFT_Compressed_N_ALPHA{ALPHA}_K{128}_d{16}_L{7}_Lambda{1e-6}/'
DECOMPRESSED_PATH = f'./data/MSFT_Decompressed_N_ALPHA{ALPHA}_K{128}_d{16}_L{7}_Lambda{1e-6}/'
'''
'''
COMPRESSED_PATH = f'./data/Stanford3d_Area_1_Compressed_N_ALPHA{ALPHA}_K{128}_d{16}_L{7}_Lambda{1e-6}/'
DECOMPRESSED_PATH = f'./data/Stanford3d_Area_1_Decompressed_N_ALPHA{ALPHA}_K{128}_d{16}_L{7}_Lambda{1e-6}/'
'''
'''
COMPRESSED_PATH = f'./data/KITTI_seq_00_Compressed_N_ALPHA{ALPHA}_K{512}_d{16}_L{7}_Lambda{1e-6}/'
DECOMPRESSED_PATH = f'./data/KITTI_seq_00_Decompressed_N_ALPHA{ALPHA}_K{512}_d{16}_L{7}_Lambda{1e-6}/'
'''

COMPRESSED_PATH = f'./data/andrew9_Compressed_N_ALPHA{ALPHA}_K{1024}_d{16}_L{7}_Lambda{1e-6}/'
DECOMPRESSED_PATH = f'./data/andrew9_Decompressed_N_ALPHA{ALPHA}_K{1024}_d{16}_L{7}_Lambda{1e-6}/'

NET_PATH, PROB_PATH = f'./model/ModelNet_N{8192}_ALPHA{2}_K{1024}_d{16}_L{7}_Lambda{1e-6}_MIN_SAMPLED_BPP{0.07}_N0_normalized_CD/ae_s78500.pkl', f'./model/ModelNet_N{8192}_ALPHA{2}_K{1024}_d{16}_L{7}_Lambda{1e-6}_MIN_SAMPLED_BPP{0.07}_N0_normalized_CD/prob_s78500.pkl'

B = 1 # Compress 1 Point Cloud Each Time !unchangable

# CREATE DECOMPRESSED_PATH PATH
if not os.path.exists(DECOMPRESSED_PATH):
    os.makedirs(DECOMPRESSED_PATH)

def byte_array_to_binary_array(byte_stream):
    int_values = [x for x in byte_stream]
    binary_array = []
    for i in range(0, len(int_values)):
        binary_array.append(list(f'{int_values[i]:08b}'))
    binary_array = np.array(binary_array, dtype=np.int32).flatten()
    return binary_array

def denormalize(pc, cetner, longest, margin=0.01):
    pc = pc - 0.5
    pc = pc * longest / (1-margin)
    pc = pc + cetner
    return pc

# GET FILENAME FROM COMPRESSED PATH
files = glob(COMPRESSED_PATH + '*.s.bin')
filenames = [x[len(COMPRESSED_PATH):-6] for x in files]

ae = torch.load(NET_PATH).to(DEVICE).eval()
# PROB MUST RUNNING ON THE GPU (don't know why...)
prob = torch.load(PROB_PATH).cuda().eval()

# CONVERT .bin FILES
start_time = time.time()
for i in tqdm(range(len(filenames))):
    octree_code_path = COMPRESSED_PATH + filenames[i] + '.s.bin'
    latent_code_path = COMPRESSED_PATH + filenames[i] + '.p.bin'
    center_scale_path = COMPRESSED_PATH + filenames[i] + '.c.bin'

    # DECODE THE OCTREED POINTS
    with open(COMPRESSED_PATH + filenames[i] + '.s.bin', 'rb') as fin:
        byte_stream = fin.read()
    octree_code = byte_array_to_binary_array(byte_stream)
    rec_sampled_xyz = pn_kit.decode_sampled_np([octree_code], scale=1)
    rec_sampled_xyz = torch.Tensor(rec_sampled_xyz)
    S = rec_sampled_xyz.shape[1]

    # GET pmf
    pmf = prob(rec_sampled_xyz.cuda())
    # USING THE pmf TO DECODE LATENT
    with open(latent_code_path, 'rb') as fin:
        byte_stream = fin.read()
    cdf = pn_kit.pmf_to_cdf(pmf).cpu()
    latent = (torchac.decode_float_cdf(cdf, byte_stream) - ae.L // 2).float().view(B*S, -1)

    # DECODE THE LATENT TO PATCHES
    latent = latent.to(DEVICE)
    linear_output = ae.inv_pool(latent)
    linear_output = linear_output.view(B*S, -1, ae.k)
    latent_quantized = latent.unsqueeze(-1).tile((1, 1, ae.k))
    mlp_input = torch.cat((linear_output, latent_quantized), dim=1)
    new_xyz = ae.inv_mlp(mlp_input)
    patches = new_xyz.transpose(2, 1)

    k = patches.shape[1]
    #K = k * ALPHA
    N = S * k
    patches = patches / ((N / N0) ** (1/3))

    # ADD PATCHES AND STRUCTURE POINTS
    pc = (patches.cpu().view(B, S, -1, 3) + rec_sampled_xyz.cpu().view(B, S, 1, 3)).reshape(B, -1, 3)
    
    # DENORMALIZE
    arr = np.fromfile(center_scale_path, dtype=np.float32)
    center = torch.Tensor(arr[:3]).reshape(1, 3)
    longest = torch.Tensor([arr[3]])
    pc = denormalize(pc, center, longest, margin=0.01)

    pc_io.save_point_cloud(pc[0].detach().cpu().numpy(), filenames[i] + '.bin.ply', path=DECOMPRESSED_PATH)

t = time.time() - start_time
n = t / len(filenames)
print(f"Done! Execution time: {round(t, 3)}s, i.e., {round(n, 5)}s per point cloud.")
