from glob import glob
import os

import numpy as np
import torch
import torch.utils.data as Data
from tqdm import tqdm
import itertools
from pytorch3d.ops.knn import _KNN, knn_gather, knn_points

import argparse

import pn_kit
import AE

torch.cuda.manual_seed(11)
torch.manual_seed(11)
np.random.seed(11)

TRAIN_GLOB = '/mnt/hdd/datasets_yk/ModelNet40_pc_01_8192p/**/train/*.ply'

N0 = 1024
ALPHA = 2
K = 256
d = 16   # Bottleneck Size
L = 7   # Quantization Level
MIN_SAMPLED_BPP = 0.25
LAMBDA = 1e-06
BETA0, BETA1 = 0, 1

BATCH_SIZE = 1
EPOCH = 8


parser = argparse.ArgumentParser(
    prog='train_ae.py',
    description='Train autoencoder using point cloud patches',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('train_glob', help='Point clouds glob pattern for training.')
parser.add_argument('model_save_folder', help='Directory where to save/load trained models.')
parser.add_argument('--N', type=int, help='Point cloud resolution.', default=8192)
parser.add_argument('--ALPHA', type=int, help='The factor of patch coverage ratio.', default=2)
parser.add_argument('--K', type=int, help='Number of points in each patch.', default=128)
parser.add_argument('--d', type=int, help='Bottleneck size.', default=16)
parser.add_argument('--lr', type=float, help='Learning rate.', default=0.0005)
parser.add_argument('--batch_size', type=int, help='number of patches in a batch.', default=16)
parser.add_argument('--lamda', type=float, help='Lambda for rate-distortion tradeoff.', default=1e-06)
parser.add_argument('--rate_loss_enable_step', type=int, help='Apply rate-distortion tradeoff at x steps.', default=40000)
parser.add_argument('--lr_decay', type=float, help='Decays the learning rate to x times the original.', default=0.1)
parser.add_argument('--lr_decay_steps', type=int, help='Decays the learning rate every x steps.', default=60000)
parser.add_argument('--max_steps', type=int, help='Train up to this number of steps..', default=80000)

args = parser.parse_args()

S = N * ALPHA // K
k = K // ALPHA

# CREATE MODEL SAVE PATH
if not os.path.exists(args.model_save_folder):
    os.makedirs(args.model_save_folder)

files = np.array(glob(TRAIN_GLOB, recursive=True))
points = pn_kit.read_point_clouds(files)

print(f'Point train samples: {points.shape[0]}, corrdinate range: [{points.min()}, {points.max()}]')

# 先转换成 torch 能识别的 Dataset
points_train_tensor = torch.Tensor(points)
torch_dataset = Data.TensorDataset(points_train_tensor, points_train_tensor)

# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
)

ae = AE.AE(K=K, k=k, d=d, L=L).cuda().train()
prob = AE.ConditionalProbabilityModel(L, d).cuda().train()
criterion = AE.get_loss().cuda()
optimizer = torch.optim.Adam(itertools.chain(ae.parameters(), prob.parameters()), lr=args.lr)

fbpps, bpps, losses = [], [], []
global_step = 0

for epoch in range(EPOCH):
    for step, (batch_x, batch_x) in enumerate(loader):
        B = batch_x.shape[0]
        batch_x = batch_x.cuda()

        # octree_np somehow can't deal with 0.0 and 1.0 ...😥
        batch_x, center, longest = pn_kit.normalize(batch_x, margin=0.01)
        
        optimizer.zero_grad()

        sampled_xyz = pn_kit.index_points(batch_x, pn_kit.farthest_point_sample_batch(batch_x, S))
        # USING OCTREE TO ENCODE SAMPLED POINTS
        octree_codes, sampled_bits = pn_kit.encode_sampled_np(sampled_xyz.detach().cpu().numpy(), scale=1, N=N, min_bpp=MIN_SAMPLED_BPP)
        rec_sampled_xyz = pn_kit.decode_sampled_np(octree_codes, scale=1)
        rec_sampled_xyz = torch.Tensor(rec_sampled_xyz).cuda()
        #pc_io.save_point_cloud(sampled_xyz[0].detach().cpu().numpy(), f'sampled_xyz.ply', './')
        #pc_io.save_point_cloud(rec_sampled_xyz[0].detach().cpu().numpy(), f'rec_sampled_xyz.ply', './')

        dist, group_idx, grouped_xyz = knn_points(rec_sampled_xyz, batch_x, K=K, return_nn=True)
        grouped_xyz -= rec_sampled_xyz.view(B, S, 1, 3)
        x_patches_orig = grouped_xyz.view(B*S, K, 3)

        #x_patches = x_patches_orig * ((N / N0) ** (1/3))
        x_patches, p_scalings = pn_kit.n_scale_batch(x_patches_orig, margin=0.01)
        patches_pred, bottleneck, latent_quantized = ae(x_patches)
        #patches_pred = patches_pred / ((N / N0) ** (1/3))
        patches_pred = pn_kit.d_n_scale_batch(patches_pred, p_scalings)
        # patches_pred: [B*S, K, 3], latent_quantized: [B*S, d]

        
        pmf = prob(rec_sampled_xyz)
        feature_bits = pn_kit.estimate_bits_from_pmf(pmf=pmf, sym=(latent_quantized.view(B, S, d) + L // 2).long())

        bpp = (sampled_bits + feature_bits) / B / N
        fbpp = feature_bits / B / N
        pc_pred = (patches_pred.view(B, S, k, 3) + rec_sampled_xyz.view(B, S, 1, 3)).reshape(B, -1, 3)
        #pc_pred = pn_kit.denormalize(pc_pred, center, longest, margin=0.01)
        pc_target = batch_x
        if global_step < args.rate_loss_enable_step:
            loss = criterion(patches_pred, x_patches_orig, pc_pred, pc_target, fbpp, beta=(BETA0, BETA1), λ=0)
        else:
            loss = criterion(patches_pred, x_patches_orig, pc_pred, pc_target, fbpp, beta=(BETA0, BETA1), λ=LAMBDA)
        #loss = criterion(patches_pred, x_patches, pc_pred, pc_target, fbpp, beta=(BETA0, BETA1), λ=LAMBDA)
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
        
         # LEARNING RATE DECAY
        if global_step % args.lr_decay_steps == 0:
            args.lr = args.lr * args.lr_decay
            for g in optimizer.param_groups:
                g['lr'] = args.lr
            print(f'Learning rate decay triggered at step {global_step}, LR is setting to{args.lr}.')

        # SAVE AND VIEW
        if global_step % 500 == 0:
            loss_value = loss.item()
            # SAVE MODEL
            torch.save(ae, args.model_save_folder + f'ae_s{global_step}.pkl')
            torch.save(prob, args.model_save_folder + f'prob_s{global_step}.pkl')
            
            for i in range(B):
                pc_io.save_point_cloud(pc_pred[i].detach().cpu().numpy(), 
                f'Step{global_step}_{i}_xhat_loss_{round(loss_value, 3)}.ply', './data/viewing_train/')

                pc_io.save_point_cloud(rec_sampled_xyz[i].detach().cpu().numpy(), 
                f'Step{global_step}_{i}_xsampled_loss_{round(loss_value, 3)}.ply', './data/viewing_train/')

                pc_io.save_point_cloud(batch_x[i].detach().cpu().numpy(), 
                f'Step{global_step}_{i}_x_loss_{round(loss_value, 3)}.ply', './data/viewing_train/')
            
            #print(latent_quantized)
    
    #if global_step > 10000:
    #    break

#np.save('./data/bpp_ls.npy', bpp_ls)
