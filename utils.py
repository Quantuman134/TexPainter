import os
import torch
import random
import numpy as np
from pytorch3d import io

def net_config(port=7890):
    proxy = f'http://127.0.0.1:{port}'

    os.environ['http_proxy'] = proxy 
    os.environ['HTTP_PROXY'] = proxy
    os.environ['https_proxy'] = proxy
    os.environ['HTTPS_PROXY'] = proxy

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def mesh_import(mesh_dir, device='cpu'):
    mesh_obj = io.load_objs_as_meshes([mesh_dir], device=device)

    verts_packed = mesh_obj.verts_packed()

    # mesh normalization
    verts_max = verts_packed.max(dim=0).values
    verts_min = verts_packed.min(dim=0).values
    max_length = (verts_max - verts_min).max().item()
    center = (verts_max + verts_min)/2

    verts_list = mesh_obj.verts_list()
    verts_list[:] = [(verts_obj - center)/max_length for verts_obj in verts_list] #[-0.5, 0.5]
    mesh_obj._verts_packed = (verts_packed - center)/max_length

    verts, faces, aux = io.load_obj(mesh_dir, device=device)
    verts = (verts - center)/max_length

    mesh_data = {'mesh_obj': mesh_obj,'verts': verts, 'faces': faces, 'aux': aux}

    return mesh_data