import torch

def bin_dist(
    dist, clash_thr=0, dist_thrs=[8], num_cls=2, check=False,
):
    if check:
        dist_thrs = sorted(dist_thrs)
        assert len(dist_thrs) > 0 and len(dist_thrs) == num_cls - 1
        assert clash_thr < dist_thrs[0]
 
    _dist = torch.zeros(dist.size(), dtype=torch.long)
    _dist[dist < dist_thrs[0]] = 0
    masks = []
    for i in range(num_cls - 1):
        _dist[dist >= dist_thrs[i]] = i + 1
    _dist[dist < clash_thr] = num_cls
    return _dist
