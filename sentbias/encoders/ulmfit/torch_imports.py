import os
import torch

import warnings
warnings.filterwarnings('ignore', message='Implicit dimension choice', category=UserWarning)

def children(m): return m if isinstance(m, (list, tuple)) else list(m.children())
def save_model(m, p): torch.save(m.state_dict(), p)
def load_model(m, p): m.load_state_dict(torch.load(p, map_location=lambda storage, loc: storage))

def load_pre(pre, f, fn):
    m = f()
    path = os.path.dirname(__file__)
    if pre: load_model(m, f'{path}/weights/{fn}.pth')
    return m
