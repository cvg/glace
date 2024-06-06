# Based on the ESAC code.
# https://github.com/vislearn/esac
# 
# BSD 3-Clause License

# Copyright (c) 2019, Visual Learning Lab
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import math
from pathlib import Path


from torch.utils.data import ConcatDataset
from dataset import CamLocDataset

class RoomDataset(ConcatDataset):

    def __init__(self, env_list, scene_id = -1, training=True, grid_cell_size=5, **kwargs ):

        with open(env_list, 'r') as f:
            environment = f.readlines()
        grid_size = math.ceil(math.sqrt(len(environment)))
        self.grid_cell_size = grid_cell_size
        self.means = np.zeros((len(environment), 3),dtype=np.float32)
        dataset_root=Path(env_list).parent

        dss=[]
        for i, line in enumerate(environment):
            if scene_id >= 0 and i != scene_id:
                continue

            line = line.split()
            scene =  dataset_root/line[0]
            ds=CamLocDataset((scene / 'train') if training else (scene / 'test' ), **kwargs)

            old_mean=np.array([float(line[1]), float(line[2]), float(line[3])])

            row = math.ceil((i+1) / grid_size)-1
            col = i % grid_size
            new_mean = np.array([-row, -col, 0]) * self.grid_cell_size
            
            # shift ground truth pose
            ds.pose_values[:, :3, 3] += (new_mean - old_mean)
            dss.append(ds)

        super(RoomDataset, self).__init__(dss)
        self.global_feat_dim=dss[0].global_feat_dim
        self.global_feats=np.concatenate([ds.global_feats for ds in dss], axis=0)
        self.pose_values=np.concatenate([ds.pose_values for ds in dss], axis=0)



    def __getitem__(self, index):
        ret = super(RoomDataset, self).__getitem__(index)
        ret = list(ret)
        ret[-1] = index
        return ret