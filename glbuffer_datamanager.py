# Copyright © Niantic, Inc. 2023. All rights reserved.
# Modified by Xudong Jiang (ETH Zurich)
"""
Global Local Feature Buffer Datamanager.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
from torch.amp import autocast
from torch.nn import Parameter
from tqdm import tqdm

from scrstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from scrstudio.data.datasets.camloc_dataset import CamLocDataset, CamLocDatasetConfig
from scrstudio.data.samplers import BatchRandomSamplerConfig, DistributedSampler, GlobalFeatSamplerConfig, RepeatSampler
from scrstudio.encoders.base_encoder import Encoder, EncoderConfig
from scrstudio.utils.rich_utils import CONSOLE


@dataclass
class GLBufferDataManagerConfig(DataManagerConfig):
    """A basic data manager"""

    _target: Type = field(default_factory=lambda: GLBufferDataManager)

    global_feat: Optional[GlobalFeatSamplerConfig] = None

    training_buffer_size: int = 8000000

    samples_per_image: int = 1024

    mixed_precision: bool = True

    train_dataset: CamLocDatasetConfig = field(default_factory=lambda: CamLocDatasetConfig())

    eval_dataset: CamLocDatasetConfig = field(default_factory=lambda: CamLocDatasetConfig())

    sampler: BatchRandomSamplerConfig = field(default_factory=lambda: BatchRandomSamplerConfig())

    encoder:EncoderConfig = field(default_factory=lambda: EncoderConfig())

    num_data_loader_workers: int = 0
    num_eval_loader_workers: int = 2

    dry_run: bool = False





class GLBufferDataManager(DataManager):

    config: GLBufferDataManagerConfig
    train_dataset: CamLocDataset
    eval_dataset: CamLocDataset

    def __init__(
        self,
        config: GLBufferDataManagerConfig,
        device: Union[torch.device, str] = "cuda",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        mixed_precision: bool = True,
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.mixed_precision = mixed_precision
        self.feat_dtype=(torch.float32, torch.float16)[self.config.mixed_precision]
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        if self.config.train_dataset.data is None and  self.config.data is not None:
            self.config.train_dataset.data = self.config.data
            self.config.eval_dataset.data = self.config.data
        self.init= self.config.train_dataset.depth is not None
        super().__init__()
        
        self.encoder: Encoder = self.config.encoder.setup(data_path=self.config.train_dataset.data)

        self.train_dataset:CamLocDataset = self.config.train_dataset.setup(preprocess=self.encoder.preprocess)
        self.eval_dataset:CamLocDataset = self.config.eval_dataset.setup(preprocess=self.encoder.preprocess)
        

        base_seed = torch.initial_seed()
        CONSOLE.print(f"Base seed in rank {self.local_rank}: {base_seed}")

        # Generator used to sample random features (runs on the GPU).
        self.sampling_generator = torch.Generator(device=self.device)
        self.sampling_generator.manual_seed(base_seed + 4095)

        # Generator used to permute the feature indices during each training epoch.
        self.training_generator = torch.Generator(device=self.device)
        self.training_generator.manual_seed(base_seed + 8191)

        # Generator for global feature noise
        self.gn_generator = torch.Generator(device=self.device)
        self.gn_generator.manual_seed(base_seed + 24601)

            
        self.encoder = self.encoder.to(self.device)
        self.global_feat_dim=self.train_dataset.global_feat_dim
        self.all_feat_dim=self.encoder.out_channels+self.global_feat_dim if self.config.global_feat else self.encoder.out_channels
        
        
        if self.config.global_feat is not None:
            self.global_feat_sampler= self.config.global_feat.setup(global_feat=torch.tensor(self.train_dataset.global_feats, dtype=self.feat_dtype,device=self.device),
                                                            generator=self.gn_generator,data=self.config.train_dataset.data)
        else:
            self.global_feat_sampler=None

        if self.train_dataset and self.test_mode == "val":
            self.setup_train()
        if self.eval_dataset and self.test_mode != "inference":
            self.setup_eval()
        
        self.eval_loader = torch.utils.data.DataLoader(self.eval_dataset,
                                                       sampler = DistributedSampler(len(self.eval_dataset),world_size=self.world_size,local_rank=self.local_rank,generator=None,shuffle=False),
                                                       shuffle=False, num_workers=1 if self.config.num_eval_loader_workers > 0 else 0,
                                      persistent_workers=self.config.num_eval_loader_workers > 0,
                                      timeout=60 if self.config.num_eval_loader_workers > 0 else 0)
        self.sampler_iter=iter(self.config.sampler.setup( dataset_size=self.training_buffer_size, generator=self.training_generator))


    def create_training_buffer(self):
        # Disable benchmarking, since we have variable tensor sizes.
        torch.backends.cudnn.benchmark = False

        num_images = len(self.train_dataset)
        num_repeats = int(math.ceil(self.config.training_buffer_size / (num_images * self.config.samples_per_image)))
        samples_per_image = self.config.training_buffer_size // (num_repeats * num_images)
        training_buffer_size = num_repeats * num_images * samples_per_image
        self.training_buffer_size = training_buffer_size
        CONSOLE.print(f"Training buffer size: {training_buffer_size}, samples per image: {samples_per_image}, num_repeats: {num_repeats},num_images: {num_images}")
        
        training_dataloader = torch.utils.data.DataLoader(self.train_dataset,  batch_size=1, sampler=RepeatSampler(num_images,num_repeats),
                            num_workers=self.config.num_data_loader_workers, pin_memory=True, persistent_workers=False,
                            timeout=60 if self.config.num_data_loader_workers > 0 else 0 )

        # Create GPU training buffers.
        self.local_buffer = {
            'features': torch.empty((self.training_buffer_size, self.encoder.out_channels), dtype=self.feat_dtype, device=self.device),
            'target_px': torch.empty((self.training_buffer_size, 2), dtype=torch.float32, device=self.device),
            'img_idx': torch.empty((self.training_buffer_size,), dtype=torch.int64, device=self.device),
            'rep_idx': torch.empty((self.training_buffer_size,), dtype=torch.int64, device=self.device),
        }

        self.global_buffer ={
            "gt_poses_inv":torch.empty((num_images,num_repeats,3,4),dtype=torch.float32,device=self.device),
            "intrinsics":torch.empty((num_images,num_repeats,3,3),dtype=torch.float32,device=self.device),
            "intrinsics_inv":torch.empty((num_images,num_repeats,3,3),dtype=torch.float32,device=self.device),
        }

        if self.init:
            self.local_buffer['gt_coords']=torch.empty((self.training_buffer_size, 3), dtype=torch.float32,
                                          device=self.device)
            
        self.encoder.eval()
        train_image_iter=iter(training_dataloader)
        with torch.no_grad():
            for i in tqdm(range(num_images),disable=None):
                for j in range(num_repeats):
                    features_to_select = samples_per_image
                    batch=next(train_image_iter)
                    for key in ('image','mask','pose','pose_inv','intrinsics','intrinsics_inv'):
                        batch[key]=batch[key].to(self.device, non_blocking=True)

                    with autocast("cuda",enabled=self.config.mixed_precision):
                        encoder_output= self.encoder.keypoint_features({k:batch[k] for k in ('image','mask')},
                                                                n=features_to_select,generator=self.sampling_generator)
                    keypoints = encoder_output["keypoints"]
                    descriptors = encoder_output["descriptors"]

                    batch_data = {
                        'features': descriptors,
                        'target_px': keypoints,
                        'img_idx': i,
                        'rep_idx': j,
                    }

                    self.global_buffer['gt_poses_inv'][i,j]=batch['pose_inv'][:, :3]
                    self.global_buffer['intrinsics'][i,j]=batch['intrinsics']
                    self.global_buffer['intrinsics_inv'][i,j]=batch['intrinsics_inv']


                    if self.init:
                        if "depth" not in batch:
                            xyz_world=torch.zeros(keypoints.shape[0],3,device=self.device)
                        else:
                            depth = batch["depth"].to(self.device, non_blocking=True)
                            xy_cam = (keypoints - batch["intrinsics"][0,[0,1],2]) / batch["intrinsics"][0,[0,1],[0,1]]
                            xy_normalized = keypoints / torch.tensor(batch['image'].shape[3:1:-1],device=self.device, dtype=torch.float32) * 2 - 1
                            z = torch.nn.functional.grid_sample(depth[None], xy_normalized.reshape(1,1,-1,2),align_corners=False,mode='nearest').reshape(-1,1)
                            xyz_cam=torch.cat([xy_cam*z,z,torch.ones(z.shape[0],1,device=self.device)],dim=1) # Nx4
                            xyz_world = torch.mm(xyz_cam,batch['pose'][0, :3].t()) # Nx3
                            xyz_world[(z==0).flatten()]=0

                        batch_data['gt_coords']=xyz_world

                    buffer_idx = i * num_repeats * samples_per_image + j * samples_per_image
                    for k in batch_data:
                        self.local_buffer[k][buffer_idx:buffer_idx + samples_per_image] = batch_data[k]

                if self.config.dry_run:
                    break
                
        buffer_memory = sum([v.element_size() * v.nelement() for k, v in self.local_buffer.items()])
        buffer_memory /= 1024 * 1024 * 1024
        CONSOLE.print(f"Training buffer memory: {buffer_memory:.2f} GB")



    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.create_training_buffer()

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        

    def next_train(self, step: int) -> Tuple[Dict, Dict]:
        """Returns the next batch of data from the train dataloader."""
        random_batch_indices = next(self.sampler_iter)
        img_idx=self.local_buffer['img_idx'][random_batch_indices]
        if self.global_feat_sampler:
            features_batch=torch.empty((self.config.sampler.batch_size, self.all_feat_dim ),
                                        dtype=self.local_buffer['features'].dtype,device=self.device)
            features_batch[:,:self.global_feat_dim] = self.global_feat_sampler.sample(img_idx)
            features_batch[:,self.global_feat_dim:]=self.local_buffer['features'][random_batch_indices]
        else:
            features_batch=self.local_buffer['features'][random_batch_indices]

        input_dict={
            'features':features_batch,
        }
        gt_dict={
            'target_px':self.local_buffer['target_px'][random_batch_indices].contiguous(),
        }

        rep_idx=self.local_buffer['rep_idx'][random_batch_indices]
        for k in ('gt_poses_inv','intrinsics','intrinsics_inv'):
            gt_dict[k]=self.global_buffer[k][img_idx,rep_idx].contiguous()

        if self.init:
            gt_dict['gt_coords']=self.local_buffer['gt_coords'][random_batch_indices].contiguous()


        return input_dict, gt_dict


    def get_train_batch_size(self) -> int:
        return self.config.sampler.batch_size

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}
