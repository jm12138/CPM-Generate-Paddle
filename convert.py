import os
import torch
import numpy as np

m0 = torch.load('./model-v1/80000/mp_rank_00_model_states.pt', map_location='cpu')
m1 = torch.load('./model-v1/80000/mp_rank_01_model_states.pt', map_location='cpu')

if not os.path.exists('numpy'):
    os.mkdir('numpy')
for x, y in zip(m0['module'].items(),  m1['module'].items()):
    n0, p0 = x
    n1, p1 = y
    if not (p0.numpy()==p1.numpy()).all():
        if 'attention.query_key_value.weight' in n0:
            w1 = np.concatenate([p0.numpy()[:1280, :], p1.numpy()[:1280, :]], 0).transpose()
            w2 = np.concatenate([p0.numpy()[1280:1280*2, :], p1.numpy()[1280:1280*2, :]], 0).transpose()
            w3 = np.concatenate([p0.numpy()[1280*2:, :], p1.numpy()[1280*2:, :]], 0).transpose()
            p = np.concatenate([w1, w2, w3], 1)
        elif 'attention.query_key_value.bias' in n0:
            w1 = np.concatenate([p0.numpy()[:1280], p1.numpy()[:1280]], 0)
            w2 = np.concatenate([p0.numpy()[1280:1280*2], p1.numpy()[1280:1280*2]], 0)
            w3 = np.concatenate([p0.numpy()[1280*2:], p1.numpy()[1280*2:]], 0)
            p = np.concatenate([w1, w2, w3], 0)
        elif 'attention.dense.weight' in n0:
            p = np.concatenate([p0.numpy(), p1.numpy()], 1)
            p = np.transpose(p)
        elif 'mlp.dense_h_to_4h.weight' in n0:
            p = np.concatenate([p0.numpy(), p1.numpy()], 0)
            p = np.transpose(p)
        elif 'mlp.dense_h_to_4h.bias' in n0:
            p = np.concatenate([p0.numpy(), p1.numpy()], 0)
        elif 'mlp.dense_4h_to_h.weight' in n0:
            p = np.concatenate([p0.numpy(), p1.numpy()], 1)
            p = np.transpose(p)
        elif 'word_embeddings' in n0:
            p = np.concatenate([p0.numpy(), p1.numpy()], 0)
        else: 
            print('other')
            print(n0, p0.numpy().shape)
            print(n1, p1.numpy().shape)
    else:
        p =  p0.numpy()
    print(n0, p.shape)
    np.save('numpy/'+n0, p)