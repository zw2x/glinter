import pickle
import sys
import numpy as np
import torch

def read_ranked_pairs(path):
    pairs = []
    with open(path, 'rt') as fh:
        for line in fh:
            recpos, ligpos, score = line.strip().split()
            recpos = int(recpos)
            ligpos = int(ligpos)
            score = float(score)
            pairs.append((recpos, ligpos, score))
    return pairs

def read_score_mat(path):
    with open(path, 'rb') as fh:
        score = pickle.load(fh)
    return score

def read_alnidx(path):
    with open(path, 'rb') as fh:
        data = pickle.load(fh)['model']
        recidx = data['recidx'].cpu().squeeze(0).numpy()
        ligidx = data['ligidx'].cpu().squeeze(0).numpy()
    return recidx, ligidx

if __name__ == '__main__':
    score = read_score_mat(sys.argv[1])
    topk, thr = [10, 25, 50, 1/10, 1/5], 8
    indices = np.argsort(score.reshape(-1))[::-1]
    with open(sys.argv[2], 'rb') as fh:
        dist = pickle.load(fh).numpy()
    if len(sys.argv) > 3:
        recidx, ligidx = read_alnidx(sys.argv[3])
        dist = dist[recidx, :][:,ligidx]
    flat_dist = dist.reshape(-1)
    assert indices.shape == flat_dist.shape
    for k in topk:
        minlen = min(dist.shape)
        den = k if k > 1 else int(k * minlen)
        den = min(indices.shape[0], den) # min(all-contacts, denominator)
        print(float(sum(flat_dist[indices[:den]] <= thr) / den))
    print(sum(flat_dist <= thr) / len(flat_dist))
