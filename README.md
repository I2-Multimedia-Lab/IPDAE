# IPDAE: Improved Patch-Based Deep Autoencoder for Point Cloud Geometry Compression

![](./figure/Teaser.png)

## Overview

Point cloud is a crucial representation of 3D contents, which has been widely used in many areas such as virtual reality, mixed reality, autonomous driving, etc. With the boost of the number of points in the data, how to efficiently compress point cloud becomes a challenging problem. In this paper, we propose a set of significant improvements to patch-based point cloud compression, i.e., a learnable context model for entropy coding, octree coding for sampling centroid points, and an integrated compression and training process. In addition, we propose an adversarial network to improve the uniformity of points during reconstruction. Our experiments show that the improved patch-based autoencoder outperforms the state-of-the-art in terms of rate-distortion performance, on both sparse and large-scale point clouds. More importantly, our method can maintain a short compression time while ensuring the reconstruction quality.

## Environment

Python 3.9.6 and Pytorch 1.9.0

**Other dependencies:**

pytorch3d 0.5.0 for KNN and chamfer loss:	https://github.com/facebookresearch/pytorch3d

geo_dist for point to plane evaluation:	https://github.com/mauriceqch/geo_dist

## Data Preparation

### Directly download converted .ply files from Google Drive (Recommand)

We have uploaded the .ply files which already converted from ModelNet40, ShapeNet and S3DIS raw format, you can get access to these data by the following link:

[8192-points ModelNet40 training and test set](https://drive.google.com/file/d/1Isa8seckZ9oNzstlE7VZcd6wVVx8LdMF/view?usp=sharing)
[2048-points ShapeNet test set](https://drive.google.com/file/d/1OzaU01kolBpfRRD0zKESYh67Hh2s2dbD/view?usp=sharing)
[S3DIS-Area1 point clouds](https://drive.google.com/file/d/1etg29uMdV932CYmWijDD7OOupjXRKZJM/view?usp=sharing)



### Convert raw datasets manually

The following steps will show you a general way to prepare point clouds in our experiment.

**ModelNet40**

1. Download the ModelNet40 data [http://modelnet.cs.princeton.edu](http://modelnet.cs.princeton.edu)

2. Convert CAD models(.off) to point clouds(.ply) by using `sample_modelnet.py`:

   ```
   python ./sample_modelnet.py ./data/ModelNet40 ./data/ModelNet40_pc_8192 --n_point 8192
   ```

**ShapeNet**

1. Download the ShapeNet data [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)

2. Sampling point clouds by using `sample_shapenet.py`:

   ```
   python ./sample_shapenet.py ./data/shapenetcore_partanno_segmentation_benchmark_v0_normal ./data/ShapeNet_pc_2048 --n_point 2048
   ```
**S3DIS**

1. Download the S3DIS data [http://buildingparser.stanford.edu/dataset.html](http://buildingparser.stanford.edu/dataset.html)

2. Sampling point clouds by using `sample_stanford3d.py`:

   ```
   python ./sample_stanford3d.py ./data/Stanford3dDataset_v1.2_Aligned_Version/Area_1/*/*.txt ./data/Stanford3d_pc/Area_1
   ```
## Training

We provided our trained models at: [Link]()

Otherwise you can use `train.py` to train our model on ModelNet40 training set:

```
python ./train.py './data/ModelNet40_pc_01_8192p/**/train/*.ply' './model/K256' --K 256
```


## Compression and Decompression

We use `compress.py` and `decompress.py` to perform compress on point clouds:

```
python ./compress.py  './data/ModelNet40_pc_01_8192p/**/test/*.ply' './data/ModelNet40_K256_compressed' './model/K256' --K 256
```

```
python ./decompress.py  './data/ModelNet40_K256_compressed' './data/ModelNet40_K256_decompressed' './model/K256' --K 256
```

## Evaluation

The Evaluation process uses the same software `geo_dist` as in [Quach's code](https://github.com/mauriceqch/pcc_geo_cnn). We use `eval.py` to calculate bitrate、PSNR and UC.

```
python ./eval.py './data/ModelNet40_pc_01_8192p/**/test/*.ply' './data/ModelNet40_K256_compressed' './data/ModelNet40_K256_decompressed' './eval/ModelNet40_K256.csv'  '../geo_dist/build/pc_error'
```
