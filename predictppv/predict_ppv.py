import sys
from collections import defaultdict
from collections import OrderedDict
   
import scipy as sp
import numpy as np
import pylab as pl

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import prepare_data


if __name__ == "__main__":

    flist = sys.argv[1:-1]
    feat_dict = prepare_data.get_features(flist)

    max_feat_len = max(map(len,feat_dict.values()))
    print max_feat_len
    del_items = []
    for prot, feat_lst in feat_dict.iteritems():
        if len(feat_lst) != max_feat_len:
            print "WARNING: %s is incomplete and will be ignored: " % prot#, feat_lst
            del_items.append(prot)

    for prot in del_items:
        del feat_dict[prot]

    print "WARNING: %d proteins will be ignored." % len(del_items) 
    
    scaler = joblib.load('scaler.pkl')
    X = prepare_data.make_scikit_input_predict(feat_dict, scaler)
    N = X.shape[1]
    r = np.zeros((N,1))
   
    svr_rbf = joblib.load('rbf.pkl')
    y_rbf = svr_rbf.predict(X)
    print len(y_rbf)
    pl.hist(y_rbf, bins=20, range=(0,1), label='RBF kernel', color="gray")
    pl.title(sys.argv[-1] + '_rbf')
    pl.xlabel("predicted PPV")
    pl.savefig(sys.argv[-1] + '_rbf_prediction.png')
    #pl.show()
    pl.close()
