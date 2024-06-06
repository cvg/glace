# GLACE: Global Local Accelerated Coordinate Encoding

----------------------------------------------------------------------------------------

This repository contains the code associated to the GLACE paper:
> **GLACE: Global Local Accelerated Coordinate Encoding**
> 
> Fangjinhua Wang, Xudong Jiang, Silvano Galliani, Christoph Vogel, Marc Pollefeys
> 
> CVPR 2024

For further information please visit:

- [Project page](https://xjiangan.github.io/glace)
- [Arxiv]

Table of contents:

- [Installation](#installation)
- [Dataset Setup](#datasets)
- [Usage](#usage)
    - [ACE Training](#ace-training)
    - [ACE Evaluation](#ace-evaluation)
    - [Training Scripts](#complete-training-and-evaluation-scripts)
    - [Pretrained GLACE Networks](#pretrained-ace-networks)
    - [Note on the Encoder Training](#encoder-training)
- [References](#publications)

## Installation

In your python environment install the required dependencies:
```shell
pip install -r requirements.txt
```
It was tested on Linux python 3.10, pytorch 2.2.2 with cuda 11.8

The GLACE network predicts dense 3D scene coordinates associated to the pixels of the input images.
In order to estimate the 6DoF camera poses, it relies on the RANSAC implementation of the DSAC* paper (Brachmann and
Rother, TPAMI 2021), which is written in C++.
As such, you need to build and install the C++/Python bindings of those functions.
You can do this with:

```shell
cd dsacstar
python setup.py install
```

Having done the steps above, you are ready to experiment with GLACE!

## Datasets

The GLACE method has been evaluated using multiple published datasets:

- [Microsoft 7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
- [Stanford 12-Scenes](https://graphics.stanford.edu/projects/reloc/)
- [Cambridge Landmarks](https://www.repository.cam.ac.uk/handle/1810/251342/)
- [Aachen Day-Night](https://www.visuallocalization.net/)

We provide scripts in the `datasets` folder to automatically download and extract the data in a format that can be
readily used by the GLACE scripts.
The format is the same used by the DSAC* codebase, see [here](https://github.com/vislearn/dsacstar#data-structure) for
details.

> **Important: make sure you have checked the license terms of each dataset before using it.**

### {7, 12}-Scenes:

You can use the `datasets/setup_{7,12}scenes.py` scripts to download the data.
As mentioned in the paper, we experimented with two variants of each of these datasets: one using the original
D-SLAM ground truth camera poses, and one using _Pseudo Ground Truth (PGT)_ camera poses obtained after running SfM on
the scenes
(see
the [ICCV 2021 paper](https://openaccess.thecvf.com/content/ICCV2021/html/Brachmann_On_the_Limits_of_Pseudo_Ground_Truth_in_Visual_Camera_ICCV_2021_paper.html)
,
and [associated code](https://github.com/tsattler/visloc_pseudo_gt_limitations/) for details).

To download and prepare the datasets using the D-SLAM poses:

```shell
cd datasets
# Downloads the data to datasets/7scenes_{chess, fire, ...}
./setup_7scenes.py
# Downloads the data to datasets/12scenes_{apt1_kitchen, ...}
./setup_12scenes.py
``` 

To download and prepare the datasets using the PGT poses:

```shell
cd datasets
# Downloads the data to datasets/pgt_7scenes_{chess, fire, ...}
./setup_7scenes.py --poses pgt
# Downloads the data to datasets/pgt_12scenes_{apt1_kitchen, ...}
./setup_12scenes.py --poses pgt
``` 

### Cambridge Landmarks / Aachen Day-Night:

We used a single variant of these datasets. Simply run:

```shell
cd datasets
# Downloads the data to datasets/Cambridge_{GreatCourt, KingsCollege, ...}
./setup_cambridge.py
# Downloads the data to datasets/aachen
./setup_aachen.py
```

Note: The Aachen Day-Night dataset has no public test ground truth. The dataset script will create dummy ground truth in the form of identity camera poses. The actual pose evaluation has to be performed via the dataset website [Visual Localization Benchmark](https://www.visuallocalization.net/).

## Usage


### Global feature extraction

We use [R2Former](https://github.com/bytedance/R2Former) for global feature. Please download the pre-trained checkpoint [CVPR23_DeitS_Rerank.pth](https://drive.google.com/file/d/1RU4wnupKXpmM0FiPeglqeNizBw4w6j38).  
Run the following to extract the global features for all the images in the dataset. 

```shell
cd datasets
python extract_features.py <scene path> --checkpoint <path to the R2Former checkpoint>
```

### GLACE Training

The GLACE scene-specific coordinate regression head for a scene can be trained using the `train_ace.py` script.
Basic usage:

```shell

torchrun --standalone --nnodes <num nodes> --nproc-per-node <num gpus per node> \
  ./train_ace.py <scene path> <output map name>
# Example:
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  ./train_ace.py datasets/7scenes_chess output/7scenes_chess.pt
```

The output map file contains just the weights of the scene-specific head network -- encoded as half-precision floating
point -- for a size of ~9MB when using default options, as mentioned in the paper. The testing script will use these
weights, together with the scene-agnostic pretrained encoder (`ace_encoder_pretrained.pt`), to estimate 6DoF
poses for the query images.

**Additional parameters** that can be passed to the training script to alter its behavior:

- `--training_buffer_size`: Changes the size of the training buffer containing decorrelated image features (see paper),
  that is created at the beginning of the training process. The default size is 16M.
- `--samples_per_image`: How many features to sample from each image during the buffer generation phase. This affects
  the amount of time necessary to fill the training buffer, but also affects the amount of decorrelation in the features
  present in the buffer. The default is 1024 samples per image.
- `--max_iterations`: How many training iterations are performed during the training. This directly affects the
  training time. Default is 30000.
- `--num_head_blocks`: The depth of the head network. Specifically, the number of extra 3-layer residual blocks to add
  to the default head depth. Default value is 1, which results in a head network composed of 9 layers, for a total of
  9MB weights.
- `--mlp_ratio`: The ratio of the hidden size of the residual block to the hidden size of the head. Default is 1.
- `--num_decoder_clusters`: The number of clusters to use in the position decoder. Default is 1.

There are other options available, they can be discovered by running the script with the `--help` flag.

### GLACE Evaluation

The pose estimation for a testing scene can be performed using the `test_ace.py` script.
Basic usage:

```shell
./test_ace.py <scene path> <output map name>
# Example:
./test_ace.py datasets/7scenes_chess output/7scenes_chess.pt
```

The script loads (a) the scene-specific GLACE head network and (b) the pre-trained scene-agnostic encoder and, for each
testing frame:

- Computes its per-pixel 3D scene coordinates, resulting in a set of 2D-3D correspondences.
- The correspondences are then passed to a RANSAC algorithm that is able to estimate a 6DoF camera pose.
- The camera poses are compared with the ground truth, and various cumulative metrics are then computed and printed
  at the end of the script.

The metrics include: %-age of frames within certain translation/angle thresholds of the ground truth,
median translation, median rotation error.

The script also creates a file containing per-frame results so that they can be parsed by other tools or analyzed
separately.
The output file is located alongside the head network and is named: `poses_<map name>_<session>.txt`.

Each line in the output file contains the results for an individual query frame, in this format:

```
file_name rot_quaternion_w rot_quaternion_x rot_quaternion_y rot_quaternion_z translation_x translation_y translation_z rot_err_deg tr_err_m inlier_count
```

There are some parameters that can be passed to the script to customize the RANSAC behavior:

- `--session`: Custom suffix to append to the name of the file containing the estimated camera poses.
- `--hypotheses`: How many pose hypotheses to generate and evaluate (i.e. the number of RANSAC iterations). Default is
    64.
- `--threshold`: Inlier threshold (in pixels) to consider a 2D-3D correspondence as valid.
- `--render_visualization`: Set to `True` to enable generating frames showing the evaluation process. Will slow down the
  testing significantly if enabled. Default `False`.
- `--render_target_path`: Base folder where the frames will be saved. The script automatically appends the current map
  name to the folder. Default is `renderings`.

There are other options available, they can be discovered by running the script with the `--help` flag.

### Complete training and evaluation scripts

We provide several scripts to run training and evaluation on the various datasets we tested our method with.
These allow replicating the results we showcased in the paper.
They are located under the `scripts` folder: `scripts/train_*.sh`.

### Pretrained GLACE Networks

We also make available the set of pretrained GLACE Heads we used for the experiments in the paper.

Each network can be passed directly to the `test_ace.py` script, together with the path to its dataset scene, to run
camera relocalization on the images of the testing split and compute the accuracy metrics, like this:

```shell
./test_ace.py datasets/7scenes_chess <Downloads>/7Scenes/7scenes_chess.pt
```

**The weights are available
at [this location](https://hkustconnect-my.sharepoint.com/:u:/g/personal/xjiangan_connect_ust_hk/ESdgFNFTuBtAqFkohVsu-wUBwMXCgEukJH0H1CCSLkxGPg?e=BL0Xx0).**


## Publications

If you use GLACE or parts of its code in your own work, please cite:

```
@inproceedings{GLACE2024CVPR,
      title     = {GLACE: Global Local Accelerated Coordinate Encoding},
      author    = {Fangjinhua Wang and Xudong Jiang and Silvano Galliani and Christoph Vogel and Marc Pollefeys},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month     = {June},
      year      = {2024}
  }
```

This code uses R2former for global feature extraction. Please consider citing:

```
@article{Zhu2023R2FU,
  title={\$R^\{2\}\$ Former: Unified Retrieval and Reranking Transformer for Place Recognition},
  author={Sijie Zhu and Linjie Yang and Chen Chen and Mubarak Shah and Xiaohui Shen and Heng Wang},
  journal={2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023},
  pages={19370-19380},
}
```

This code builds on previous camera relocalization pipelines, namely DSAC, DSAC++, DSAC*, and ACE. Please consider citing:

```
@inproceedings{brachmann2023ace,
    title={Accelerated Coordinate Encoding: Learning to Relocalize in Minutes using RGB and Poses},
    author={Brachmann, Eric and Cavallari, Tommaso and Prisacariu, Victor Adrian},
    booktitle={CVPR},
    year={2023},
}

@inproceedings{brachmann2017dsac,
  title={{DSAC}-{Differentiable RANSAC} for Camera Localization},
  author={Brachmann, Eric and Krull, Alexander and Nowozin, Sebastian and Shotton, Jamie and Michel, Frank and Gumhold, Stefan and Rother, Carsten},
  booktitle={CVPR},
  year={2017}
}

@inproceedings{brachmann2018lessmore,
  title={Learning less is more - {6D} camera localization via {3D} surface regression},
  author={Brachmann, Eric and Rother, Carsten},
  booktitle={CVPR},
  year={2018}
}

@article{brachmann2021dsacstar,
  title={Visual Camera Re-Localization from {RGB} and {RGB-D} Images Using {DSAC}},
  author={Brachmann, Eric and Rother, Carsten},
  journal={TPAMI},
  year={2021}
}
```

## License

Copyright © Niantic, Inc. 2023. Patent Pending.
All rights reserved.
Please see the [license file](LICENSE) for terms.
Modified files: `ace_network.py`, `ace_trainer.py`, `ace_vis_utils.py`, `ace_visualizer.py`, `dataset.py`, `test_ace.py`, `train_ace.py`, scripts in `scripts/` folder.

Datasets in the `datasets` folder are provided with their own licenses. Please check their license terms before using.
Global feature extraction script `datasets/extract_features.py` is based on R2Former, which is licensed under the [Apache License 2.0](https://github.com/bytedance/R2Former/blob/91d314f25de64098cdc8a479d9f022fdc2287f49/LICENSE).

