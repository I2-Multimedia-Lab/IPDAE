import os
import argparse
from glob import glob

import numpy as np
from tqdm import tqdm

import pn_kit

parser = argparse.ArgumentParser('Stranford3d Sampler')
parser.add_argument('--source', help='source directory', default='/home/yk/Projects/DataSets/Stanford3dDataset_v1.2_Aligned_Version/Area_1/*/*.txt')
parser.add_argument('--dest', help='destination directory', default='/home/yk/Projects/DataSets/S3DIS-Area1_pc/Area_1')
args = parser.parse_args()

# CREATE DEST
if not os.path.exists(args.dest):
    os.makedirs(args.dest)

files = np.array(glob(args.source))
filenames = np.array([os.path.splitext(os.path.split(x)[1])[0] for x in files])

print('We get files\' name:')
print(filenames)

print('Saving .plys...')
for i in tqdm(range(files.shape[0])):
    file = files[i]
    filename = filenames[i]
    pc = np.loadtxt(file)[:, :3]
    pn_kit.save_point_cloud(pc, filename + '.ply', args.dest)
