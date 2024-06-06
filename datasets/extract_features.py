# Copyright 2023 Byte Dance

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Reference: https://github.com/bytedance/R2Former

import argparse
import os
from functools import partial
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class ImageFolderDataset(Dataset):
    def __init__(self, root, transform):
        self.root, self.transform = root, transform
        self.imgs = sorted(glob(os.path.join(root, '*.jpg')) + glob(os.path.join(root, '*.png')))

    def __getitem__(self, idx):
        return self.transform(Image.open(self.imgs[idx]).convert("RGB"))

    def __len__(self):
        return len(self.imgs)

class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)
        self.single = True

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        return F.normalize( (x + x_dist) / 2, p=2, dim=1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract global features')
    parser.add_argument('scene_path', type=str, help='input directory')
    parser.add_argument('--checkpoint', type=str, default='CVPR23_DeitS_Rerank.pth', help='path to the checkpoint')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--img_resize', type=int, nargs=2, default=[480, 640], help='image resize')
    args = parser.parse_args()


    model=DistilledVisionTransformer(
            img_size=args.img_resize, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=256)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = {k.replace('module.backbone.', ''): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('module.backbone')}
    model.load_state_dict(state_dict)

    device = torch.device("cuda")
    model.to(device)
    model=model.eval()

    # model = torch.compile(model)

    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(args.img_resize,antialias=False)
    ])

    for split in ['train', 'test']:
        if not os.path.exists(os.path.join(args.scene_path, split)):
            continue

        image_path = os.path.join(args.scene_path, split,"rgb")

        dataset = ImageFolderDataset(image_path, transform=base_transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)

        all_features = np.zeros((len(dataset), 256), dtype="float32")

        with torch.inference_mode():
            for i, x in enumerate(tqdm(dataloader)):
                x= model(x.to(device))
                all_features[i*args.batch_size:(i+1)*args.batch_size] = x.cpu().numpy()

        np.save(os.path.join(args.scene_path, split, "features.npy"), all_features)