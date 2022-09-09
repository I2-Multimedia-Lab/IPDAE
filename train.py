import os
import argparse
import itertools

import numpy as np
import torch
import torch.utils.data as Data
from pytorch3d.ops.knn import _KNN, knn_gather, knn_points

from glob import glob

import pn_kit
import AE

torch.cuda.manual_seed(11)
torch.manual_seed(11)
np.random.seed(11)


parser = argparse.ArgumentParser(
    prog='train_ae.py',
    description='Train autoencoder using point cloud patches',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('train_glob', help='Point clouds glob pattern for training.', default='/mnt/hdd/datasets_yk/ModelNet40_pc_01_8192p/**/train/*.ply')
parser.add_argument('model_save_folder', help='Directory where to save trained models.', default='./model/K256/')

parser.add_argument('--N', type=int, help='Point cloud resolution.', default=8192)
parser.add_argument('--N0', type=int, help='Scale Transformation constant.', default=1024)
parser.add_argument('--ALPHA', type=int, help='The factor of patch coverage ratio.', default=2)
parser.add_argument('--K', type=int, help='Number of points in each patch.', default=256)
parser.add_argument('--d', type=int, help='Bottleneck size.', default=16)
parser.add_argument('--L', type=int, help='Quantization Level.', default=7)

parser.add_argument('--lr', type=float, help='Learning rate.', default=0.0005)
parser.add_argument('--batch_size', type=int, help='Batch size (must be 1).', default=1)
parser.add_argument('--lamda', type=float, help='Lambda for rate-distortion tradeoff.', default=1e-06)
parser.add_argument('--rate_loss_enable_step', type=int, help='Apply rate-distortion tradeoff at x steps.', default=40000)
parser.add_argument('--lr_decay', type=float, help='Decays the learning rate to x times the original.', default=0.1)
parser.add_argument('--lr_decay_steps', type=int, help='Decays the learning rate every x steps.', default=60000)
parser.add_argument('--max_steps', type=int, help='Train up to this number of steps.', default=80000)

args = parser.parse_args()

N = args.N
N0 = args.N0
K = args.K
S = N * args.ALPHA // K
k = K // args.ALPHA

# CREATE MODEL SAVE PATH
if not os.path.exists(args.model_save_folder):
    os.makedirs(args.model_save_folder)

files = np.array(glob(args.train_glob, recursive=True))
points = pn_kit.read_point_clouds(files)

print(f'Point train samples: {points.shape}, corrdinate range: [{points.min()}, {points.max()}]')

# PARSE TO DATASET
points_train_tensor = torch.Tensor(points)
torch_dataset = Data.TensorDataset(points_train_tensor, points_train_tensor)

loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size = args.batch_size,
    shuffle = True,
)

ae = AE.AE(K=K, k=k, d=args.d, L=args.L).cuda().train()
prob = AE.ConditionalProbabilityModel(args.L, args.d).cuda().train()
criterion = AE.get_loss().cuda()
optimizer = torch.optim.Adam(itertools.chain(ae.parameters(), prob.parameters()), lr=args.lr)

fbpps, bpps, losses = [], [], []
global_step = 0

for epoch in range(9999):
    for step, (batch_x, batch_x) in enumerate(loader):
        B = batch_x.shape[0]
        batch_x = batch_x.cuda()

        # octree_np somehow can't deal with 0.0 and 1.0 ...😥 so we put margin on sphere space
        batch_x, center, longest = pn_kit.normalize(batch_x, margin=0.01)
        
        optimizer.zero_grad()

        sampled_xyz = pn_kit.index_points(batch_x, pn_kit.farthest_point_sample_batch(batch_x, S))
        # USING OCTREE TO ENCODE SAMPLED POINTS
        octree_codes, sampled_bits = pn_kit.encode_sampled_np(sampled_xyz.detach().cpu().numpy(), scale=1, N=N, min_bpp=pn_kit.OCTREE_BPP_DICT[K])
        rec_sampled_xyz = pn_kit.decode_sampled_np(octree_codes, scale=1)
        rec_sampled_xyz = torch.Tensor(rec_sampled_xyz).cuda()

        dist, group_idx, grouped_xyz = knn_points(rec_sampled_xyz, batch_x, K=K, return_nn=True)
        grouped_xyz -= rec_sampled_xyz.view(B, S, 1, 3)
        x_patches_orig = grouped_xyz.view(B*S, K, 3)

        x_patches = x_patches_orig * ((N / N0) ** (1/3))
        patches_pred, bottleneck, latent_quantized = ae(x_patches)
        patches_pred = patches_pred / ((N / N0) ** (1/3))
        # patches_pred: [B*S, K, 3], latent_quantized: [B*S, d]

        
        pmf = prob(rec_sampled_xyz)
        feature_bits = pn_kit.estimate_bits_from_pmf(pmf=pmf, sym=(latent_quantized.view(B, S, args.d) + args.L // 2).long())

        bpp = (sampled_bits + feature_bits) / B / N
        fbpp = feature_bits / B / N
        pc_pred = (patches_pred.view(B, S, k, 3) + rec_sampled_xyz.view(B, S, 1, 3)).reshape(B, -1, 3)
        pc_target = batch_x
        if global_step < args.rate_loss_enable_step:
            loss = criterion(pc_pred, pc_target, fbpp, λ=0)
        else:
            loss = criterion(pc_pred, pc_target, fbpp, λ=args.lamda)
        loss.backward()
        optimizer.step()
        global_step += 1

        # PRINT
        losses.append(loss.item())
        fbpps.append(fbpp.item())
        bpps.append(bpp.item())
        if global_step % 500 == 0:
            print(f'Epoch:{epoch} | Step:{global_step} | Feature bpp:{round(np.array(fbpps).mean(), 5)} | Bpp:{round(np.array(bpps).mean(), 5)} | Loss:{round(np.array(losses).mean(), 5)}')
            losses, fbpps, bpps = [], [], []

            torch.save(ae.state_dict(), os.path.join(args.model_save_folder, 'ae.pkl'))
            torch.save(prob.state_dict(), os.path.join(args.model_save_folder, 'prob.pkl'))
        
         # LEARNING RATE DECAY
        if global_step % args.lr_decay_steps == 0:
            args.lr = args.lr * args.lr_decay
            for g in optimizer.param_groups:
                g['lr'] = args.lr
            print(f'Learning rate decay triggered at step {global_step}, LR is setting to{args.lr}.')

        if global_step > args.max_steps:
            break
    
    if global_step > args.max_steps:
        break

