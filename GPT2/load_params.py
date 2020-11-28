import os
import paddle
import numpy as np
def LoadParams(model_dir):
    names = os.listdir(model_dir)
    names.sort()
    state_dict = {}
    for name in names:
        if 'npy' in name:
            param = np.load(os.path.join(model_dir, name))
            param = paddle.to_tensor(param, dtype='float32')
            state_dict[name[:-4]] = param
            print(name[:-4], param.shape)
    return state_dict