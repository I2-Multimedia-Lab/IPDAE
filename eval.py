import os
from glob import glob
import subprocess

import numpy as np
import pandas as pd
from tqdm import tqdm
from pyntcloud import PyntCloud
from plyfile import PlyData
from pytorch3d.ops.knn import _KNN, knn_gather, knn_points

import torch

import pc_io
from pytorch3d.loss import chamfer_distance

'''
INPUT_PATH = '/home/yk/Projects/DataSets/ShapeNet_pc_01_2048p/**/test/*.ply'
DECOMPRESSED_PATH = f'./data/ShapeNet_Decompressed_N_ALPHA{2}_K{64}_d{16}_L{7}_Lambda{1e-6}/'
COMPRESSED_PATH = f'./data/ShapeNet_Compressed_N_ALPHA{2}_K{64}_d{16}_L{7}_Lambda{1e-6}/'
OUTPUT_FILE = f'./eval/ShapeNet_N_ALPHA{2}_K{64}_d{16}_L{7}_Lambda{1e-6}.csv'
'''

INPUT_PATH = '/mnt/hdd/datasets_yk/ShapeNet_pc_01_2048p/**/test/*.ply'
COMPRESSED_PATH = f'./data/ShapeNet_Compressed_N_ALPHA{2}_K{256}_d{16}_L{7}_Lambda{1e-6}/'
DECOMPRESSED_PATH = f'./data/ShapeNet_Decompressed_N_ALPHA{2}_K{256}_d{16}_L{7}_Lambda{1e-6}/'
OUTPUT_FILE = f'./eval/ShapeNet_N_ALPHA{2}_K{256}_d{16}_L{7}_Lambda{1e-6}.csv'

'''
INPUT_PATH = '/home/yk/Projects/DataSets/Stanford3d_pc/Area_1/*.ply'
COMPRESSED_PATH = f'./data/Stanford3d_Area_1_Compressed_N_ALPHA{2}_K{64}_d{16}_L{7}_Lambda{1e-6}/'
DECOMPRESSED_PATH = f'./data/Stanford3d_Area_1_Decompressed_N_ALPHA{2}_K{64}_d{16}_L{7}_Lambda{1e-6}/'
OUTPUT_FILE = f'./eval/Stanford3d_Area_1_ALPHA{2}_K{64}_d{16}_L{7}_Lambda{1e-6}.csv'
'''
'''
INPUT_PATH = '/home/yk/Projects/DataSets/Stanford3d_pc/Area_1/*.ply'
COMPRESSED_PATH = f'/home/yk/Projects/Others/pcc_geo_cnn_v2/data/Stanford3d_pc_512_Area_1_compressed_c4_1.00e-04/'
DECOMPRESSED_PATH = f'/home/yk/Projects/Others/pcc_geo_cnn_v2/data/Stanford3d_pc_512_Area_1_decompressed_c4_1.00e-04/'
OUTPUT_FILE = f'/home/yk/Projects/Others/pcc_geo_cnn_v2/eval/Stanford3d_pc_512_Area_1_decompressed_c4_1.00e-04.csv/'
'''
'''
INPUT_PATH = '/home/yk/Projects/DataSets/KITTI_pc/sequences/00/*.ply'
COMPRESSED_PATH = f'./data/KITTI_seq_00_Compressed_N_ALPHA{2}_K{512}_d{16}_L{7}_Lambda{1e-6}/'
DECOMPRESSED_PATH = f'./data/KITTI_seq_00_Decompressed_N_ALPHA{2}_K{512}_d{16}_L{7}_Lambda{1e-6}/'
OUTPUT_FILE = f'./eval/KITTI_seq_00_Decompressed_N_ALPHA{2}_K{512}_d{16}_L{7}_Lambda{1e-6}.csv'
'''

INPUT_PATH = '/mnt/hdd/datasets_yk/msft/andrew9/ply/frame0000.ply'
COMPRESSED_PATH = f'./data/andrew9_Compressed_N_ALPHA{2}_K{1024}_d{16}_L{7}_Lambda{1e-6}/'
DECOMPRESSED_PATH = f'./data/andrew9_Decompressed_N_ALPHA{2}_K{1024}_d{16}_L{7}_Lambda{1e-6}/'
OUTPUT_FILE = f'./eval/andrew9_Decompressed_N_ALPHA{2}_K{1024}_d{16}_L{7}_Lambda{1e-6}.csv'

PC_ERROR_PATH = '/mnt/hdd/datasets_yk/Project_Others/Others/geo_dist-master/build/pc_error'

# CALC PSNR BETWEEN FILE AND DECOMPRESSED FILE
def pc_error(f, df):
    command = f'{PC_ERROR_PATH} -a {f} -b {df} --knn {32}'
    #print("Executing " + command)
    output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    decoded_output = output.decode('utf-8').split('\n')
    data_lines = [x for x in decoded_output if '   ### ' in x]
    parsed_data_lines = [x[len('   ### '):] for x in data_lines]
    # Before last value : information about the metric
    # Last value : metric value
    data = [(','.join(x[:-1]), x[-1]) for x in [x.split(',') for x in parsed_data_lines]]
    return data

def read_pc(f):
    plydata = PlyData.read(f)
    try:
        pc = np.array(np.transpose(np.stack((plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z'])))).astype(np.float32)        
    except:
        pc = np.array(np.transpose(np.stack((plydata['vertex']['X'],plydata['vertex']['Y'],plydata['vertex']['Z'])))).astype(np.float32)        
    return pc

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

def get_n_points(f):
    return len(PyntCloud.from_file(f).points)

def get_file_size_in_bits(f):
    return os.stat(f).st_size * 8

# GET FILE NAME FROM DECOMPRESSED PATH
files = pc_io.get_files(INPUT_PATH)
files_cat = np.array([os.path.split(os.path.split(x)[0])[1] for x in files])
#files = files[files_cat == 'test']

filenames = np.array([os.path.split(x)[1] for x in files])

# .csv COLUMNS: [filename, p2pointPSNR, p2planePSNR, n_points_input, n_points_output, bpp]
ipt_files, p2pointPSNRs, p2planePSNRs, chamfer_ds, n_points_inputs, n_points_outputs, bpps, ucs = [], [], [], [], [], [], [], []

print('Evaluating...')
for i in tqdm(range(len(filenames))):
    input_f = files[i]
    comp_s_f = COMPRESSED_PATH + filenames[i] + '.s.bin'
    comp_p_f = COMPRESSED_PATH + filenames[i] + '.p.bin'
    comp_c_f = COMPRESSED_PATH + filenames[i] + '.c.bin'
    decomp_f = DECOMPRESSED_PATH + filenames[i] + '.bin.ply'

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
    input_pc = read_pc(input_f)
    # input_region = KNN_Region(input_pc, input_pc[0], 1024)

    decomp_pc = read_pc(decomp_f)
    # decomp_region = KNN_Region(decomp_pc, decomp_pc[0], 1024)

    # input_region_dist = calc_self_neighboor_dist(input_region)
    # decomp_region_dist = calc_self_neighboor_dist(decomp_region)

    # uc = np.var(decomp_region_dist) / np.var(input_region_dist)
    # ucs.append(np.round(uc, 3))
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
# df['uniformity coefficient'] = ucs
df.to_csv(OUTPUT_FILE)


