import sys
from collections import defaultdict
from collections import OrderedDict

from sklearn import preprocessing
import numpy as np

def get_features(flist):

    print "Input files are:"
    print flist
    feat_dict = defaultdict(list)
    
    for f in flist:
        for l in open(f):
            l_lst = l.strip().split()
            prot = l_lst[0].upper()
            val = map(float, l_lst[1:])
            feat_dict[prot] += val

    print "Collected features for %d proteins." % len(feat_dict)
    return feat_dict


def get_labels(feat_dict, label_fname):
    
    print "Labels are in file:"
    print label_fname

    for l in open(label_fname):
        l_lst = l.strip().split()
        prot = l_lst[0].upper()
        val = float(l_lst[-1])
        feat_dict[prot].append(val)
       
    return feat_dict


def make_scikit_input(feat_dict):

    ordered_feat_dict = OrderedDict(sorted(feat_dict.items()))
    X = []
    y = []
    for prot, lst in ordered_feat_dict.iteritems():
        X.append(lst[:-2])
        y.append(lst[-2])

    X = np.array(X)
    y = np.array(y)
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled, y


def make_scikit_input_predict(feat_dict, scaler):

    ordered_feat_dict = OrderedDict(sorted(feat_dict.items()))
    X = []
    y = []
    for prot, lst in ordered_feat_dict.iteritems():
        X.append(lst)

    X = np.array(X)
    X_scaled = scaler.transform(X)
    return X_scaled


def make_scikit_input_kcv(feat_dict, set_prefix='set', k=5):

    # define partitions in training set
    for i in range(1,k+1):
        id_lst = [prot.strip() for prot in open(set_prefix + str(i)).readlines()]
        for id_i in id_lst:
            if id_i in feat_dict: # avoid creating new entries
                feat_dict[id_i].append(i)

    ordered_feat_dict = OrderedDict(sorted(feat_dict.items()))
    #print ordered_feat_dict.keys()

    # define remaining proteins as independent test set
    max_val_lst_len = max(map(len,feat_dict.values()))
    for prot, val_lst in feat_dict.iteritems():
        if len(val_lst) == max_val_lst_len - 1:
            feat_dict[prot].append(0)

    # STATE:
    # feat_dict -> {prot: [feat_0, ..., feat_n, label, fold_i]}
    # with fold_i = 0 -> test set
    # else -> fold_i in training set

    # now partition dataset according to definition above
    
    ordered_feat_dict = OrderedDict(sorted(feat_dict.items()))
    X_train = {}
    y_train = {}
    X_test = []
    y_test = []

    for i in range(1,k+1):
        X_train_i = []
        y_train_i = []
        for prot, val_lst in ordered_feat_dict.iteritems():
            curr_i = val_lst[-1]
            if curr_i == i:
                X_train_i.append(val_lst[:-2])
                y_train_i.append(val_lst[-2])
        X_train[i] = X_train_i
        y_train[i] = y_train_i

    for prot, val_lst in ordered_feat_dict.iteritems():
        curr_i = val_lst[-1]
        if curr_i == 0:
            X_test.append(val_lst[:-2])
            y_test.append(val_lst[-2])

    return X_train, y_train, X_test, y_test



if __name__ == "__main__":

    feat_dict = get_features(sys.argv[1:-1])
    feat_dict = get_labels(feat_dict, sys.argv[-1])

    max_feat_len = max(map(len,feat_dict.values()))
    print max_feat_len
    del_items = []
    for prot, feat_lst in feat_dict.iteritems():
        if len(feat_lst) != max_feat_len:
            print "WARNING: %s is incomplete and will be ignored: " % prot, feat_lst
            del_items.append(prot)

    for prot in del_items:
        del feat_dict[prot]

    print "WARNING: %d proteins will be ignored." % len(del_items) 
    
    X, y = make_scikit_input(feat_dict)
