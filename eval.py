import os
import subprocess
import argparse

import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from pyntcloud import PyntCloud
from plyfile import PlyData

import torch
from pytorch3d.ops.knn import _KNN, knn_gather, knn_points
from pytorch3d.loss import chamfer_distance

import pn_kit

parser = argparse.ArgumentParser(
    prog='train_ae.py',
    description='Train autoencoder using point cloud patches',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('input_glob', help='Point clouds glob pattern for compression.', default='/mnt/hdd/datasets_yk/ModelNet40_pc_01_8192p/**/test/*.ply')
parser.add_argument('compressed_path', help='Comressed .bin files folder.', default='./data/ModelNet40_K256_compressed/')
parser.add_argument('decompressed_path', help='Decompressed .ply files folder.', default='./data/ModelNet40_K256_decompressed/')
parser.add_argument('output_file', help='Evaluation Detail saved as csv.', default='./eval/ModelNet40_K256.csv')
parser.add_argument('pc_error_path', help='Path to pc_error.', default='/mnt/hdd/datasets_yk/Project_Others/Others/geo_dist-master/build/pc_error')

args = parser.parse_args()

# CALC PSNR BETWEEN FILE AND DECOMPRESSED FILE
def pc_error(f, df):
    command = f'{args.pc_error_path} -a {f} -b {df} --knn {32}'
    #print("Executing " + command)
    output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    decoded_output = output.decode('utf-8').split('\n')
    data_lines = [x for x in decoded_output if '   ### ' in x]
    parsed_data_lines = [x[len('   ### '):] for x in data_lines]
    # Before last value : information about the metric
    # Last value : metric value
    data = [(','.join(x[:-1]), x[-1]) for x in [x.split(',') for x in parsed_data_lines]]
    return data

def calc_uc(input_pc, decomp_pc):

    def KNN_Region(pc, point, K):
        pc = torch.Tensor(pc)
        point = torch.Tensor(point)
        dist, group_idx, grouped_xyz = knn_points(point.view(1, 1, 3), pc.unsqueeze(0), K=K, return_nn=True)
        grouped_xyz -= point.view(1, 1, 1, 3)
        x_patches = grouped_xyz.view(K, 3)
        x_patches = x_patches.numpy()
        return x_patches

    def calc_self_neighboor_dist(pc):
        pc = torch.Tensor(pc)
        dist = torch.cdist(pc, pc, p=2)
        values, indices = torch.topk(dist, k=2, largest=False)
        neighboor_dist = values[:, 1]
        neighboor_dist = neighboor_dist.numpy()
        return neighboor_dist

    input_region = KNN_Region(input_pc, input_pc[0], 1024)
    decomp_region = KNN_Region(decomp_pc, decomp_pc[0], 1024)
    input_region_dist = calc_self_neighboor_dist(input_region)
    decomp_region_dist = calc_self_neighboor_dist(decomp_region)
    uc = np.var(decomp_region_dist) / np.var(input_region_dist)
    return uc

def get_n_points(f):
    return len(PyntCloud.from_file(f).points)

def get_file_size_in_bits(f):
    return os.stat(f).st_size * 8

# GET FILE NAME FROM DECOMPRESSED PATH
files = np.array(glob(args.input_glob, recursive=True))
filenames = np.array([os.path.split(x)[1] for x in files])


# .csv COLUMNS: [filename, p2pointPSNR, p2planePSNR, n_points_input, n_points_output, bpp]
ipt_files, p2pointPSNRs, p2planePSNRs, chamfer_ds, n_points_inputs, n_points_outputs, bpps, ucs = [], [], [], [], [], [], [], []

print('Evaluating...')
for i in tqdm(range(len(filenames))):
    input_f = files[i]
    comp_s_f = args.compressed_path + filenames[i] + '.s.bin'
    comp_p_f = args.compressed_path + filenames[i] + '.p.bin'
    comp_c_f = args.compressed_path + filenames[i] + '.c.bin'
    decomp_f = args.decompressed_path + filenames[i] + '.bin.ply'

    if not os.path.exists(decomp_f):
        continue

    ipt_files.append(filenames[i])
    # GET PSNR
    data = pc_error(input_f, decomp_f)
    p2pointPSNRs.append(round(float(data[-3][1]), 3))
    p2planePSNRs.append(round(float(data[-1][1]), 3))
    # GET NUMBER OF POINTS
    n_points_input = get_n_points(input_f)
    n_points_output = get_n_points(decomp_f)
    n_points_inputs.append(n_points_input)
    n_points_outputs.append(n_points_output)
    # GET BPP
    bpp = (get_file_size_in_bits(comp_s_f) + get_file_size_in_bits(comp_p_f) + get_file_size_in_bits(comp_c_f)) / n_points_input
    bpps.append(bpp)

    # CALC THE UNIFORMITY COEFFICIENT
    input_pc = pn_kit.read_point_cloud(input_f)
    decomp_pc = pn_kit.read_point_cloud(decomp_f)
    uc = calc_uc(input_pc, decomp_pc)
    ucs.append(np.round(uc, 3))

    # normed chamfer distance
    input_pc_max = input_pc.max()
    input_pc_min = input_pc.min()
    input_pc = (input_pc - input_pc_min) / (input_pc_max - input_pc_min)
    decomp_pc = (decomp_pc - input_pc_min) / (input_pc_max - input_pc_min)

    chamfer_d, loss_normals = chamfer_distance(torch.Tensor(decomp_pc).unsqueeze(0).cuda(), torch.Tensor(input_pc).unsqueeze(0).cuda())
    chamfer_ds.append(chamfer_d.item())

    
print(f'Done! The average p2pointPSNR: {round(np.array(p2pointPSNRs).mean(), 3)} | p2plane PSNR: {round(np.array(p2planePSNRs).mean(), 3)} | chamfer distance: {round(np.array(chamfer_ds).mean(), 8)} | bpp: {round(np.array(bpps).mean(), 3)} | uc: {round(np.array(ucs).mean(), 3)}')


# SAVE AS AN EXCEL .csv
df = pd.DataFrame()
df['filename'] = ipt_files
df['p2pointPSNR'] = p2pointPSNRs
df['p2planePSNR'] = p2planePSNRs
df['chamfer_distance'] = chamfer_ds
df['n_points_input'] = n_points_inputs
df['n_points_output'] = n_points_outputs
df['bpp'] = bpps
df['uniformity coefficient'] = ucs
df.to_csv(args.output_file)
