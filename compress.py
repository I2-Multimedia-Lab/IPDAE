import os
import time
import argparse

import numpy as np
import torch
import torchac
from pytorch3d.ops.knn import _KNN, knn_gather, knn_points

from tqdm import tqdm
from glob import glob

import pn_kit
import AE

torch.cuda.manual_seed(11)
torch.manual_seed(11)
np.random.seed(11)

parser = argparse.ArgumentParser(
    prog='compress.py',
    description='Compress Point Clouds Using Trained Model.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('input_glob', help='Point clouds glob pattern for compression.', default='/mnt/hdd/datasets_yk/ModelNet40_pc_01_8192p/**/test/*.ply')
parser.add_argument('compressed_path', help='Comressed .bin files folder.', default='./data/ModelNet40_K256_compressed/')
parser.add_argument('model_load_folder', help='Directory where to load trained models.', default='./model/K256/')

parser.add_argument('--N0', type=int, help='Scale Transformation constant.', default=1024)
parser.add_argument('--ALPHA', type=int, help='The factor of patch coverage ratio.', default=2)
parser.add_argument('--K', type=int, help='Number of points in each patch.', default=256)
parser.add_argument('--d', type=int, help='Bottleneck size.', default=16)
parser.add_argument('--L', type=int, help='Quantization Level.', default=7)
parser.add_argument('--octree_bpp', type=int, help='Centroids bpp limit.', default=0.25)

args = parser.parse_args()

N0 = args.N0
K = args.K
k = K // args.ALPHA

B = 1 # Compress 1 Point Cloud Each Time !unchangable in this implementation

# CREATE COMPRESSED FOLDER
if not os.path.exists(args.compressed_path):
    os.makedirs(args.compressed_path)

# READ INPUT FILES
files = np.array(glob(args.input_glob, recursive=True))
filenames = np.array([os.path.split(x)[1] for x in files])

NET_PATH = os.path.join(args.model_load_folder, 'ae.pkl')
PROB_PATH = os.path.join(args.model_load_folder, 'prob.pkl')

ae = AE.AE(K=K, k=k, d=args.d, L=args.L).cuda()
ae.load_state_dict(torch.load(NET_PATH))
ae.eval()

prob = AE.ConditionalProbabilityModel(args.L, args.d).cuda()
prob.load_state_dict(torch.load(PROB_PATH))
prob.eval()


def KNN_Patching(batch_x, sampled_xyz, K):
    dist, group_idx, grouped_xyz = knn_points(sampled_xyz, batch_x, K=K, return_nn=True)
    grouped_xyz -= sampled_xyz.view(B, S, 1, 3)
    x_patches = grouped_xyz.view(B*S, K, 3)
    return x_patches

time_saver = []
# DO THE COMPRESS
with torch.no_grad():
    for i in tqdm(range(filenames.shape[0])):
        # GET 1 POINT CLOUD
        pc = pn_kit.read_point_cloud(files[i])
        pc = torch.Tensor(pc).cuda()
        pc = pc.unsqueeze(0)
        
        start_time = time.time()

        # normalize our point cloud.
        # remove the pc to (0.5, 0.5)
        # scale the size to (0+margin, 1-margin)
        pc, center, longest = pn_kit.normalize(pc, margin=0.01)

        N = pc.shape[1]
        S = (int)(N * args.ALPHA // K)

        # SAMPLING
        sampled_xyz = pn_kit.index_points(pc, pn_kit.farthest_point_sample_batch(pc, S))
        # OCTREE ENCODE
        octree_codes, sampled_bits = pn_kit.encode_sampled_np(sampled_xyz.detach().cpu().numpy(), scale=1, N=N, min_bpp=pn_kit.OCTREE_BPP_DICT[K])
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
        cdf = pn_kit.pmf_to_cdf(pmf).cpu()
        n_latent_quantized = latent_quantized.view(B, S, -1).to(torch.int16).cpu() + args.L // 2
        byte_stream = torchac.encode_float_cdf(cdf, n_latent_quantized, check_input_bounds=True)

        # * WRITE AE TO FILE
        with open(os.path.join(args.compressed_path, filenames[i] + '.p.bin'), 'wb') as fout:
            fout.write(byte_stream)

        # * SAVE OCTREE CODE TO FILE
        octree_code = octree_codes[0]
        byte_stream = pn_kit.binary_array_to_byte_array(octree_code)
        with open(os.path.join(args.compressed_path, filenames[i] + '.s.bin'), 'wb') as fout:
            fout.write(byte_stream)
        
        # * SAVE CENTER AND SCALE TO FILE
        arr = np.zeros((4))
        arr[:3] = center.detach().cpu().numpy().flatten()
        arr[3] = longest.detach().cpu().numpy()
        arr.astype(np.float32).tofile(os.path.join(args.compressed_path, filenames[i] + '.c.bin'))

        t = time.time() - start_time
        time_saver.append(t)

mean_time = np.array(time_saver).mean()
print(f"Done! Execution time: {round(mean_time, 5)}s per point cloud.")
