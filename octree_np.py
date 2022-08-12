import os
import numpy as np
from tqdm import tqdm

########
## A simple Python implementation of octree coding
## Only point clouds with coordinates in the unit sphere (does not contain boundary values (i.e., 0 and 1)) are supported
########

def encode(pc, resolution, depth):
    '''
    input: 0 <= depth
    '''
    pc = getDecodeFromPc(pc, resolution, depth)
    bits_ls = [[]]
    for i in range(depth+1):
        bits_ls.append([])

    def include_points(startX, startY, startZ, endX, endY, endZ):
        if np.any((startX <= pc[:, 0]) * (pc[:, 0] <= endX) * (startY <= pc[:, 1]) * (pc[:, 1] <= endY) * (startZ <= pc[:, 2]) * (pc[:, 2] <= endZ)):
            return True
        else:
            return False

    def octree(startX, startY, startZ, currdepth):
        curr_cube_reso = resolution / (2 ** currdepth)
        if include_points(startX, startY, startZ, startX+curr_cube_reso, startY+curr_cube_reso, startZ+curr_cube_reso):
            bits_ls[currdepth].append(1)
            if currdepth == depth:
                return
            next_cube_reso = curr_cube_reso / 2
            octree(startX, startY, startZ, currdepth+1)
            octree(startX, startY, startZ+next_cube_reso, currdepth+1)
            octree(startX, startY+next_cube_reso, startZ, currdepth+1)
            octree(startX, startY+next_cube_reso, startZ+next_cube_reso, currdepth+1)

            octree(startX+next_cube_reso, startY, startZ, currdepth+1)
            octree(startX+next_cube_reso, startY, startZ+next_cube_reso, currdepth+1)
            octree(startX+next_cube_reso, startY+next_cube_reso, startZ, currdepth+1)
            octree(startX+next_cube_reso, startY+next_cube_reso, startZ+next_cube_reso, currdepth+1)
        else:
            bits_ls[currdepth].append(0)

    octree(0,0,0,0)
    bits = [i for ls in bits_ls for i in ls]
    del bits[0]
    bits = np.array(bits)
    return bits

def decode(bits, resolution):
    # build the bits_ls
    bits_ls = [[1]]
    bits = bits.tolist()
    n = 8
    while True:
        bits_group = bits[:n]
        del bits[:n]
        bits_ls.append(bits_group)
        n = sum(bits_group) * 8
        if len(bits) == 0:
            break
    depth = len(bits_ls) - 1
    pc = []
    #print(depth)
    def dec(startX, startY, startZ, currdepth):
        curr_cube_reso = resolution / (2 ** currdepth)
        b = bits_ls[currdepth].pop(0)
        if b == 1:
            #print(b)
            if currdepth == depth:
                pc.append([startX+curr_cube_reso/2, startY+curr_cube_reso/2, startZ+curr_cube_reso/2])
                return
            next_cube_reso = curr_cube_reso / 2
            dec(startX, startY, startZ, currdepth+1)
            dec(startX, startY, startZ+next_cube_reso, currdepth+1)
            dec(startX, startY+next_cube_reso, startZ, currdepth+1)
            dec(startX, startY+next_cube_reso, startZ+next_cube_reso, currdepth+1)

            dec(startX+next_cube_reso, startY, startZ, currdepth+1)
            dec(startX+next_cube_reso, startY, startZ+next_cube_reso, currdepth+1)
            dec(startX+next_cube_reso, startY+next_cube_reso, startZ, currdepth+1)
            dec(startX+next_cube_reso, startY+next_cube_reso, startZ+next_cube_reso, currdepth+1)
        else:
            return
    dec(0,0,0,0)
    pc = np.array(pc)
    return pc

def getDecodeFromPc(pc, resolution, depth):
    cube_reso = resolution / (2 ** depth)
    pc_octree = (pc // cube_reso * cube_reso) + (cube_reso / 2)
    pc_octree = np.unique(pc_octree, axis=0)
    #print(pc_octree)
    return pc_octree

