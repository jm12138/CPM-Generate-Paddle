import os
import torch
import paddle
import argparse
import numpy as np
from tqdm import tqdm

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--original_model_dir", type=str, required=True, help="the detection model dir.")
parser.add_argument("--save_model_dir", type=str, required=True, help="the detection model dir.")
args = parser.parse_args()

# 加载原始模型
m0 = torch.load(os.path.join(args.original_model_dir, 'mp_rank_00_model_states.pt'), map_location='cpu')
m1 = torch.load(os.path.join(args.original_model_dir, 'mp_rank_01_model_states.pt'), map_location='cpu')

# 创建保存目录
if not os.path.exists(args.save_model_dir):
    os.mkdir(args.save_model_dir)

# 模型参数转换
state_dict = {}
for x, y in tqdm(zip(m0['module'].items(),  m1['module'].items())):
    name_0, param_0 = x
    name_1, param_1 = y
    param_0 = param_0.numpy()
    param_1 = param_1.numpy()

    if not (param_0==param_1).all():
        if 'attention.query_key_value.weight' in name_0:
            w1 = np.concatenate([param_0[:1280, :], param_1[:1280, :]], 0)
            w2 = np.concatenate([param_0[1280:1280*2, :], param_1[1280:1280*2, :]], 0)
            w3 = np.concatenate([param_0[1280*2:, :], param_1[1280*2:, :]], 0)
            param = np.concatenate([w1, w2, w3], 0).transpose()

        elif 'attention.query_key_value.bias' in name_0:
            w1 = np.concatenate([param_0[:1280], param_1[:1280]], 0)
            w2 = np.concatenate([param_0[1280:1280*2], param_1[1280:1280*2]], 0)
            w3 = np.concatenate([param_0[1280*2:], param_1[1280*2:]], 0)
            param = np.concatenate([w1, w2, w3], 0)

        elif 'attention.dense.weight' in name_0:
            param = np.concatenate([param_0, param_1], 1).transpose()

        elif 'mlp.dense_h_to_4h.weight' in name_0:
            param = np.concatenate([param_0, param_1], 0).transpose()

        elif 'mlp.dense_h_to_4h.bias' in name_0:
            param = np.concatenate([param_0, param_1], 0)

        elif 'mlp.dense_4h_to_h.weight' in name_0:
            param = np.concatenate([param_0, param_1], 1).transpose()

        elif 'word_embeddings' in name_0:
            param = np.concatenate([param_0, param_1], 0)
            
    else:
        param =  param_0

    state_dict[name_0] = paddle.to_tensor(param)

# 模型参数保存
paddle.save(state_dict, os.path.join(args.save_model_dir, 'CPM-LM.pdparams'))