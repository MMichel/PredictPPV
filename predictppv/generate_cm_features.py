import sys
import math
import numpy as np
from scipy.ndimage.filters import generic_filter

from parsing import parse_contacts


def feat_score_gt(cmap, score_threshold=0.2):

    """ Get number of contacts greater than certain score.
    @param  cmap                contact map
    @param  score_threshold     score threshold
    @return N                   number of contacts, int
    """

    bool_cmap = cmap > score_threshold
    N = len(np.where(bool_cmap)[0])

    return N


def feat_score_gt_norm(cmap, score_threshold=0.2):

    """ Get number of contacts greater than certain score,
    normalized by sequence length.
    @param  cmap                contact map
    @param  score_threshold     score threshold
    @return N_norm              normalized number of contacts, float
    """

    L = cmap.shape[0]
    N_norm = feat_score_gt(cmap, score_threshold) / float(L)

    return N_norm


def feat_max_score(cmap):

    """ Get maximum score of contact map.
    @param  cmap                contact map
    @return sc_max              max contact score, float
    """

    return np.max(cmap) 


def feat_nth_score(cmap, frac=1.0):

    """ Get score of N-th contact.
    @param  cmap                contact map
    @param  frac                fraction of sequence length
    @return sc_max              N-th contact score, float
    """
    
    L = cmap.shape[0]

    # number of top ranked contacts relative to sequence length
    N = math.ceil(L * frac)

    #idx = np.argpartition(cmap, -N)[-N:]
    #print cmap[idx]
    flat = cmap.flatten()
    flat.sort()
    flat = flat[::-1]
    return flat[N-1]


def feat_avg_score(cmap, frac=1.0):

    """ Get average score of top ranked contacts.
    Number of top ranked contacts is here a fraction of sequence 
    length.
    @param  cmap                contact map
    @param  frac                fraction of sequence length
    @return sc_avg              average contact score, float
    """

    L = cmap.shape[0]

    # number of top ranked contacts relative to sequence length
    N = math.ceil(L * frac)
    
    # get indices of top ranked contacts
    #idx_lst = np.argpartition(cmap, -N)[-N:]
    #print idx_lst
    #print cmap[idx_lst]
    #sc_avg = np.mean(cmap[idx_lst])
    flat = cmap.flatten()
    flat.sort()
    flat = flat[::-1]

    sc_avg = np.mean(flat[:N])

    return sc_avg


def feat_avg_dist(clist, score_threshold=0.2):

    """ Calculate average distance to nearest contact.
    @param  clist               list of contacts
    @param  score_threshold     score threshold
    @return avg_d               average distance to nearest contact
    """

    bool_cmap = cmap > score_threshold

    L = cmap.shape[0]
    N = len(np.where(bool_cmap)[0])
    
    dist_to_diag = np.abs(np.add.outer(np.arange(L), -np.arange(L)))
    avg_d = np.mean(dist_to_diag[np.where(bool_cmap)])
    #TODO: use
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist
    return avg_d


def feat_contact_order(cmap, score_threshold=0.2):

    """ Calculate contact order of the given contact map.
    @param  cmap                contact map
    @param  score_threshold     score threshold
    @return co                  contact order, float
    """

    bool_cmap = cmap > score_threshold

    L = cmap.shape[0]
    N = len(np.where(bool_cmap)[0])

    if N == 0:
        return 0.
    
    dist_to_diag = np.abs(np.add.outer(np.arange(L), -np.arange(L)))
    sum_seq_sep = np.sum(dist_to_diag[np.where(bool_cmap)])
    co = (1./(L * N)) * sum_seq_sep

    return co


def feat_numc_window(cmap, winsize=3, score_threshold=0.2):

    """ Get distribution over number of contacts in certain window,
    where the window is a 2d field around each contacts.
    The distribution is represented as an array of counts for each
    number of contacts with in the window (i.e. length=winsize^2)
    @param  cmap                contact map
    @param  winsize             size of the window
    @param  score_threshold     score threshold
    @return nw                  array for the distribution
    """

    L = cmap.shape[0]

    # function to apply to each window
    def count(win):
        return np.sum(win > score_threshold)

    # applies "count" in a 2D sliding window of size "winsize"
    cmap_counts = generic_filter(cmap, count, size=winsize).astype(int)

    # generate histogram over all winsize^2 possible counts
    nw = np.bincount(cmap_counts.flatten())
    nw.resize(winsize*winsize + 1)
    
    # normalize by number of possible sliding windows
    num_windows = math.pow(L - (winsize-1), 2)

    return nw / float(num_windows)


def feat_numc_diag(cmap, diaglen=21, score_threshold=0.2):

    """ Get distribution of number of contacts along both diagonals
    centered at each contact. Measure of clustering along secondary
    structure interactions. The distribution is represented as counts
    per distance to the contact.
    @param  cmap                contact map
    @param  diaglen             length of the diagonal
    @param  score_threshold     score threshold
    @return nd                  array with counts for each distance
    """
    
    nd = np.zeros(diaglen/2)
    for dist in range(1, diaglen, 2):

        # function to apply to each window
        # counts contacts in corners of window if there is a contact
        # in the center
        # IDEA: apply this function to sliding windows of different sizes
        # ==> equal to count along both diagonals
        def count_corners(win):
            win = win.reshape((dist,dist))
            bool_win = win > score_threshold
            center = bool_win[dist/2, dist/2]
            if center:
                corners = bool_win[[0,0,-1,-1],[0,-1,0,-1]]
            else:
                corners = 0
            return np.sum(corners)

        cmap_counts = generic_filter(cmap, count_corners, size=dist)
        nd[dist/2] = np.sum(cmap_counts)
    
    # normalize by number of diagonal elements for all contacts
    bool_cmap = cmap > score_threshold
    N = len(np.where(bool_cmap)[0])
    if N == 0:
        return 0.

    n_diag = diaglen-1.

    return nd / (N * n_diag)



def main(path_to_cmap, th=0.4, frac=1.0, start=0, end=-1):

    # guessing separator of constraint file
    with open(path_to_cmap) as cmap_file:
        line = cmap_file.readline()
        if len(line.split(',')) != 1:
            sep = ','
        elif len(line.split(' ')) != 1:
            sep = ' '
        else:
            sep = '\t'
        clist = parse_contacts.parse(cmap_file, sep, min_dist=5)
    cmap = parse_contacts.get_numpy_cmap(clist)

    if end == -1:
        end = cmap.shape[0]
    cmap = cmap[start:end, start:end]

    gt = []
    gt_norm = []
    for i in range(1, 10):
        th_gt = i/10.0
        gt.append(feat_score_gt(cmap, score_threshold=th_gt))
        gt_norm.append(feat_score_gt_norm(cmap, score_threshold=th_gt))
    nth = feat_nth_score(cmap, frac=frac)
    avg = feat_avg_score(cmap, frac=frac)
    
    co = feat_contact_order(cmap, score_threshold=th)
    nw = feat_numc_window(cmap, score_threshold=th)
    nd = feat_numc_diag(cmap, score_threshold=th)
    #print "#Columns are: ntop0.1-0.9, gt_norm0.1-0.9, max, avg, co, nw0-9, nd1-10"
    #print len([path_to_cmap] + list(gt) + list(gt_norm) + [max] + [avg] + [co] + list(nw) + list(nd))
    print path_to_cmap, ' '.join(map(str,gt)), ' '.join(map(str,gt_norm)), nth, avg, co, ' '.join(map(str,nw)), ' '.join(map(str,nd))

    #print path_to_cmap, co, ' '.join(map(str,nw)), ' '.join(map(str,nd))
    #return np.array(np.hstack((gt, gt_norm, max, avg, co, nw, nd)))


if __name__ == "__main__":

    path_to_cmap = sys.argv[1]
    th = float(sys.argv[2])
    if len(sys.argv) > 3:
        start = int(sys.argv[3])
        end = int(sys.argv[4])
        feat_arr = main(path_to_cmap, th=th, start=start, end=end)
    else:
        feat_arr = main(path_to_cmap, th=th)
    #print feat_arr
