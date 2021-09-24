from Bio import pairwise2
from glinter.protein import read_seqs

def aln_to_cigar(aln):
    start, end = aln.start, aln.end
    qbeg = len([ _ for _ in aln.seqA[:start] if _ != '-' ]) + 1
    tbeg = len([ _ for _ in aln.seqB[:start] if _ != '-' ]) + 1
    cigar = []
    for wa, wb in zip(aln.seqA[start:end], aln.seqB[start:end]):
        if wa != '-' and wb != '-':
            cigar.append('M')
        elif wa == '-' and wb != '-':
            cigar.append('D')
        elif wb == '-' and wa != '-':
            cigar.append('I')
        else:
            print('gap and gap are aligned together, bug in Bio.pairwise2?')
    cigar_str = ''
    prev = ['', 0]
    for k in cigar:
        if k == prev[0]:
            prev[1] += 1
        else:
            cigar_str += f'{prev[1]}{prev[0]}' if prev[0] else ''
            prev = [k, 1]
    else:
        if prev[0] and prev[1] > 0:
            cigar_str += f'{prev[1]}{prev[0]}'
            
    return qbeg, tbeg, cigar_str

if __name__ == '__main__':
    import sys
    # query
    srcseqs = read_seqs(sys.argv[1], topk=1)
    srcname, srcseq = [ (n, s) for n, s in srcseqs.items() ][0]
    
    # tgt
    tgtseqs = read_seqs(sys.argv[2], topk=1)
    tgtname, tgtseq = [ (n, s) for n, s in tgtseqs.items() ][0]
    if tgtseq != srcseq:
        aln = pairwise2.align.localms(srcseq, tgtseq, 2, -1, -2, -0.5)[0]
        print('\t'.join(
            [srcname, tgtname] + [str(_) for _ in aln_to_cigar(aln)]
        ))
    else:
        print(f'{srcname}\t{tgtname}\t1\t1\t{len(srcseq)}M')
