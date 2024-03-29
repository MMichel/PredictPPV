import sys
import math
import os.path

import numpy as np

def get_seqlen(filename):
    alifile = open(filename, 'r')
    l = alifile.readline()
    if l.startswith('>'):
        L = len(alifile.readline().strip())
    else:
        L = len(l.strip())
    alifile.close()
    return L


def get_numseq_coverage(filename, coverage=0.9, start=0, end=-1):
    L = get_seqlen(filename)
    if end == -1:
        end = L
    L = L - (start + (L - end))
    #print L
    N = 0
    alifile = open(filename, 'r')
    for line in alifile:
        if line.startswith('>'):
            continue
        line = line[start:end]
        frac_gaps = 0.
        ngaps = line.count('-')
        frac_gaps += ngaps/float(L)
        #print L,ngaps,frac_gaps,1-coverage
        if frac_gaps < 1-coverage:
            N += 1
    alifile.close()
    return N


"""
def str_to_byte(str):
    return np.frombuffer(str, dtype=np.byte)

# this is way too slow for big alignments...
# need to run cd-hit instead
def get_numseq_id(filename, identity=0.9):
    alifile = open(filename, 'r')
    ali_str = alifile.readlines()
    ali_seqs = [seq for seq in ali_str if not seq.startswith('>')]
    ali = map(str_to_byte, ali_seqs)
    L = float(len(ali[0]))
    N = 0 
    for i, seq1 in enumerate(ali):
        #ids = np.array(map(lambda x: (seq1 == x).sum()/float(L), ali))
        for seq2 in ali[i+1:]:
            if (seq1 == seq2).sum()/L > identity:
                N += 1
                break
        # for all groups of similar sequences count only one
        # i.e.: count indices of similar sequences (inner np.where) 
        # and count those above the diagonal (outer np.where)
        #N += len(np.where(np.where(ids > identity)[0] > i)[0])
    alifile.close()
    return len(ali) - N
"""


def get_numseq_id(filename, identity=0.9):

    cdhit_fname = filename + '.cd%s.clstr' % int(identity * 100)
    stats_fname = '.'.join(filename.split('.')[:-1] + '.stats')
    if os.path.isfile(cdhit_fname):
        cdhit_clst = [l for l in open(cdhit_fname).readlines() if l.startswith('>')]
        numseq_id = len(cdhit_clst)
    elif os.path.isfile(stats_fname):
        with open(stats_fname) as f:
            for l in f:
                if 'Count at %s' % int(identity * 100) in l:
                    numseq_id = int(l.split()[-1])
    else:
        sys.stderr.write("File %s does not exist.\nPlease run CD-HIT (automatically ran by PconsC3) on %s.\n" % (cdhit_fname, filename))
        sys.exit(1)
    return numseq_id


def get_meff(filename):
    
    meff_fname = '.'.join(filename.split('.')[:-1]) + '.gneff'
    if not os.path.isfile(meff_fname):
        sys.stderr.write("File %s does not exist.\nPlease run GaussDCA (automatically run by PconsC3) on %s.\n" % (meff_fname, filename))
        sys.exit(1)
    # File content looks for example like this:
    # theta = 0.3862907131851712 threshold = 50.0
    # M = 65522 N = 130 Meff = 1367.7051932101365
    # MeffPerPos = [...]
    with open(meff_fname) as f:
        for l in f:
            if 'Meff =' in l:
                meff = l.split()[-1]
    return float(meff)


def main(path_to_aln, cov=0.9, id=0.9, start=0, end=-1):

    seqlen = get_seqlen(path_to_aln) 
    if end == -1:
        end = seqlen
    seqlen = seqlen - (start + (seqlen - end))

    ncov = get_numseq_coverage(path_to_aln, coverage=cov, start=start, end=end)
    ncov_log10 = math.log10(ncov)
    nid = get_numseq_id(path_to_aln, identity=id)
    nid_log10 = math.log10(nid)
    meff = get_meff(path_to_aln)
    meff_log10 = math.log10(meff)

    #print "#Columns are: ncov, log10(ncov), nid, log10(nid), meff, log10(meff)"
    #print ncov, ncov_log10, nid, nid_log10, meff, meff_log10
    #return np.array([ncov, ncov_log10, nid, nid_log10, meff, meff_log10])

    print path_to_aln, seqlen, ncov, nid, nid_log10, meff, meff_log10
    return np.array([seqlen, ncov, nid, nid_log10, meff, meff_log10])

if __name__ == "__main__":
    path_to_aln = sys.argv[1]
    if len(sys.argv) > 2:
        start = int(sys.argv[2])
        end = int(sys.argv[3])
        main(path_to_aln, cov=0.9, id=0.9, start=start, end=end)
    else:
        main(path_to_aln, cov=0.9, id=0.9)
