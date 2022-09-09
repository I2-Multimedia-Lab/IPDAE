import os
import time
import argparse

import numpy as np
import torch
import torchac

from tqdm import tqdm
from glob import glob

import pn_kit
import AE

torch.cuda.manual_seed(11)
torch.manual_seed(11)
np.random.seed(11)

parser = argparse.ArgumentParser(
    prog='decompress.py',
    description='Deompress Point Clouds Using Trained Model.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('compressed_path', help='Comressed .bin files folder.', default='./data/ModelNet40_K256_compressed/')
parser.add_argument('decompressed_path', help='Decompressed .ply files folder.', default='./data/ModelNet40_K256_decompressed/')
parser.add_argument('model_load_folder', help='Directory where to load trained models.', default='./model/K256/')

parser.add_argument('--N0', type=int, help='Scale Transformation constant.', default=1024)
parser.add_argument('--ALPHA', type=int, help='The factor of patch coverage ratio.', default=2)
parser.add_argument('--K', type=int, help='Number of points in each patch.', default=256)
parser.add_argument('--d', type=int, help='Bottleneck size.', default=16)
parser.add_argument('--L', type=int, help='Quantization Level.', default=7)

parser.add_argument('--device', help='AE Model Device (cpu or cuda)', default='cpu')

args = parser.parse_args()

N0 = args.N0
K = args.K
k = K // args.ALPHA

B = 1 # Compress 1 Point Cloud Each Time !unchangable in this implementation

# CREATE COMPRESSED FOLDER
if not os.path.exists(args.decompressed_path):
    os.makedirs(args.decompressed_path)

# GET FILENAME FROM COMPRESSED PATH
files = glob(args.compressed_path + '*.s.bin')
filenames = [x[len(args.compressed_path):-6] for x in files]

NET_PATH = os.path.join(args.model_load_folder, 'ae.pkl')
PROB_PATH = os.path.join(args.model_load_folder, 'prob.pkl')

ae = AE.AE(K=K, k=k, d=args.d, L=args.L).to(args.device)
ae.load_state_dict(torch.load(NET_PATH))
ae.eval()

# PROB MUST RUNNING ON THE GPU (don't know why...)
prob = AE.ConditionalProbabilityModel(args.L, args.d).cuda()
prob.load_state_dict(torch.load(PROB_PATH))
prob.eval()

time_saver = []

for i in tqdm(range(len(filenames))):
    octree_code_path = args.compressed_path + filenames[i] + '.s.bin'
    latent_code_path = args.compressed_path + filenames[i] + '.p.bin'
    center_scale_path = args.compressed_path + filenames[i] + '.c.bin'
    
    start_time = time.time()

    # DECODE THE OCTREED POINTS
    with open(octree_code_path, 'rb') as fin:
        byte_stream = fin.read()
    octree_code = pn_kit.byte_array_to_binary_array(byte_stream)
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
    latent = latent.to(args.device)
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
    pc = pn_kit.denormalize(pc, center, longest, margin=0.01)

    t = time.time() - start_time
    time_saver.append(t)

    pn_kit.save_point_cloud(pc[0].detach().cpu().numpy(), filenames[i] + '.bin.ply', path=args.decompressed_path)

mean_time = np.array(time_saver).mean()
print(f"Done! Execution time: {round(mean_time, 5)}s per point cloud.")
