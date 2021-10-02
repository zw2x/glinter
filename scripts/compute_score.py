import pickle
import os
import numpy as np

def read_residue_positions(pos_file):
    with open(pos_file, 'rt') as fh:
        pos = [ int(_) for _ in fh.readline().strip().split() ]
    pos = np.array(pos, dtype=np.long)
    return pos

def show(d, name1, name2, pos1, pos2):
    with open(f'{d}/{name1}:{name2}.out.pkl', 'rb') as fh:
        data = pickle.load(fh)['model']
        score = np.exp(data['output'][0,:,:,0].cpu().numpy())
        recidx = data['recidx'].cpu().squeeze(0).numpy()
        ligidx = data['ligidx'].cpu().squeeze(0).numpy()
    if os.path.exists(f'{d}/{name2}:{name1}.out.pkl'):
        with open(f'{d}/{name2}:{name1}.out.pkl', 'rb') as fh:
            dataT = pickle.load(fh)['model']['output']
            score += np.exp(dataT[0,:,:,0].cpu().numpy().T)
            score /= 2
    _pos1 = np.repeat(pos1[:,np.newaxis], len(pos2), axis=-1)
    _pos2 = np.repeat(pos2[np.newaxis, :], len(pos1), axis=0)
    ref_pos = np.concatenate(
        (_pos1[...,np.newaxis], _pos2[...,np.newaxis]), axis=-1
    )[recidx, :][:,ligidx]
    top_idx = np.argsort(score.reshape(-1))[::-1]
    ranked_score = score.reshape(-1)[top_idx]
    ranked_pos_pair = ref_pos.reshape(-1, 2)[top_idx]
    rank = []
    for i in range(len(ranked_score)):
        rec_pos, lig_pos = ranked_pos_pair[i]
        rank.append((rec_pos, lig_pos, ranked_score[i]))
    return score, rank

if __name__ == '__main__':
    import sys
    srcdir, name1, name2 = sys.argv[1:4]
    pos1 = read_residue_positions(f'{srcdir}/{name1}/{name1}.pos')
    pos2 = read_residue_positions(f'{srcdir}/{name2}/{name2}.pos')
    score, rank = show(srcdir, name1, name2, pos1, pos2)
    with open(f'{srcdir}/score_mat.pkl', 'wb') as fh:
        pickle.dump(score, fh)
    with open(f'{srcdir}/ranked_pairs.txt', 'wt') as fh:
        for p1, p2, s in rank:
            fh.write(f'{int(p1)} {int(p2)} {float(s):.4f}\n')
