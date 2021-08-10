import math
import random
import torch

def add_gaussian_noise(t, std=0.5):
    return torch.normal(t, std,)

def get_random_rotmat(dim, axis=-1, degrees=(-180, 180), device=None):
    # get a random rotation matrix (torch.FloatTensor)
    if axis < 0:
        # rotate all axis
        axis = list(range(dim))
    elif not isinstance(axis, list,):
        axis = [ axis ]

    degree = math.pi * random.uniform(*degrees) / 180.0
    sin, cos = math.sin(degree), math.cos(degree)
    if dim == 2:
        matrix = torch.FloatTensor([[cos, sin], [-sin, cos]], device=device)
    else:
        matrix = None
        for ax in axis:
            if ax == 0:
                mat = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
            elif ax == 1:
                mat = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
            else:
                mat = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
            mat = torch.FloatTensor(mat, device=device)
            if matrix is None:
                matrix = mat
            else:
                matrix = torch.matmul(matrix, mat)
    return matrix

def compute_centered_lrf(center, a, b):
    """
    center->a : x-axis
    a-center-b-plane cross x-axis : z-axis
    x-axis cross z-axis : y-axis
    """
    a = a - center
    b = b - center
    x = _normalize(a)
    z = _normalize(torch.cross(a, b, dim=-1))
    y = _normalize(torch.cross(x, z))
    lrf = torch.stack([x,y,z], dim=-1)
    return lrf

def _normalize(v):
    return v / torch.sqrt(torch.sum(v ** 2, dim=-1, keepdim=True))


if __name__ == '__main__':
    matrix = get_random_rotmat(3)
    print(torch.matmul(matrix.T, matrix))
