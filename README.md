# SG-NeRF

This is the official implement of ECCV 2024 paper "SG-NeRF: Neural Surface Reconstruction with Scene Graph Optimization".

## Data Convention

The [data](https://drive.google.com/file/d/1vd-pxbIZ_Y7tpHupaitDR7rbJjUHWwcg/view?usp=sharing) is organized as follows:

```
<case_name>
|-- cameras_sphere.npz	    	# camera parameters
|-- database.db			# colmap database
|-- train			# input images from each view 
	|-- 000.png
	|-- 001.png
	...
```

Here the `cameras_sphere.npz` follows the data format in [IDR](https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md), where `world_mat_xx` denotes the world to image projection matrix, and `scale_mat_xx` denotes the normalization matrix.

The `database.db` follows the database format in [COLMAP](https://github.com/colmap/colmap/blob/main/doc/database.rst).

## Installation

```shell
pip install -r requirements.txt
```

## Running

- Training on SG-NeRF dataset

  ```shell
  python exp_runner.py --case <case_name> --conf ./confs/sgnerf.conf --gpu <cuda_index>
  ```

- Training on DTU dataset

  ```shell
  python exp_runner.py --case <case_name> --conf ./confs/dtu.conf --gpu <cuda_index>
  ```

- Training on custom configuration

  ```shell
  python exp_runner.py --case <case_name> --conf <config_file> --gpu <cuda_index>
  ```

## Acknowledgement

Some code snippets are borrowed from [NeuS](https://github.com/Totoro97/NeuS) and [COLMAP](https://github.com/colmap/colmap). Thanks for these great projects.
