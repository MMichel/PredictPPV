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


def run_cv(X_train, y_train):

    C_lst = [1, 10, 100, 1000]
    gamma_lst = [1e-2, 1e-3, 1e-4, 1e-5]
    #n_est_lst = [10, 20, 50, 100, 200]
    n_est_lst = [100]
    #min_samples_lst = [1,2,5,10,20]
    min_samples_lst = [1]

    topscore_lin = float("-inf")
    topscore_rbf = float("-inf")
    topscore_rf = float("-inf")
    topC_lin = -1
    topC_rbf = -1
    topgamma_rbf = -1
    topnest_rf = -1
    topsamples_rf = -1

    score_dict = defaultdict(list)
    y_dict = defaultdict(list)

    # k-fold cv on training set
    fold_lst = range(1,k+1)
    for i in fold_lst:
        
        X_train_i = []
        for j in fold_lst:
            if j != i:
                X_train_i += X_train[j]
        X_train_i = np.array(X_train_i)

        y_train_i = []
        for j in fold_lst:
            if j != i:
                y_train_i += y_train[j]
        y_train_i = np.array(y_train_i)

        X_test_i = np.array(X_train[i])
        y_test_i = np.array(y_train[i])
       
        scaler = StandardScaler().fit(X_train_i)
        X_train_i = scaler.transform(X_train_i)
        X_test_i = scaler.transform(X_test_i)

        # gridsearch for rbf kernel
        for C in C_lst:
            for gamma in gamma_lst:
                svr_rbf = SVR(kernel='rbf', C=C, gamma=gamma, max_iter=200000)
                y_rbf = svr_rbf.fit(X_train_i, y_train_i).predict(X_test_i)
                score_key = 'RBF;C=%s;gamma=%s' % (C, gamma)
                #score_val = r2_score(y_test_i, y_rbf)
                #score_val = -mean_squared_error(y_test_i, y_rbf)
                score_val = sp.stats.pearsonr(y_test_i, y_rbf)[0]
                if score_val > topscore_rbf:
                    topscore_rbf = score_val
                    topC_rbf = C
                    topgamma_rbf = gamma
                score_dict[score_key].append(score_val)
                y_dict[score_key].append(y_rbf)

        # gridsearch for linear kernel
        for C in C_lst:
            svr_lin = SVR(kernel='linear', C=C, max_iter=200000)
            y_lin = svr_lin.fit(X_train_i, y_train_i).predict(X_test_i)
            score_key = 'Lin;C=%s' % C
            #score_val = r2_score(y_test_i, y_lin)
            #score_val = -mean_squared_error(y_test_i, y_lin)
            score_val = sp.stats.pearsonr(y_test_i, y_lin)[0]
            if score_val > topscore_lin:
                topscore_lin = score_val
                topC_lin = C
            score_dict[score_key].append(score_val)
            y_dict[score_key].append(y_lin)

        # gridsearch for random forest
        for n in n_est_lst:
            for ms in min_samples_lst:
                rf = RandomForestRegressor(n_estimators=n,
                        min_samples_leaf=ms)
                y_rf = rf.fit(X_train_i, y_train_i).predict(X_test_i)
                score_key = 'RF;nTrees=%s;minSamplesLeaf=%s' % (n, ms)
                #score_val = r2_score(y_test_i, y_rf)
                #score_val = -mean_squared_error(y_test_i, y_rf)
                score_val = sp.stats.pearsonr(y_test_i, y_rf)[0]
                if score_val > topscore_rf:
                    topscore_rf = score_val
                    topnest_rf = n
                    topsamples_rf = ms
                score_dict[score_key].append(score_val)
                y_dict[score_key].append(y_rf)

    return score_dict, y_dict, topC_lin, topC_rbf, topgamma_rbf, topnest_rf, topsamples_rf


if __name__ == "__main__":

    k = 5

    flist = sys.argv[1:-2]
    feat_dict = prepare_data.get_features(flist)
    feat_dict = prepare_data.get_labels(feat_dict, sys.argv[-2])

    max_feat_len = max(map(len,feat_dict.values()))
    print max_feat_len
    print map(len,feat_dict.values())
    del_items = []
    for prot, feat_lst in feat_dict.iteritems():
        #if len(feat_lst) == max_feat_len:
        #    print 'BLA' + prot, feat_lst
        if len(feat_lst) != max_feat_len:
            print "WARNING: %s is incomplete and will be ignored: " % prot#, feat_lst
            #print feat_lst
            del_items.append(prot)

    for prot in del_items:
        del feat_dict[prot]

    print "WARNING: %d proteins will be ignored." % len(del_items) 
    
    #X_train, y_train, X_test, y_test = prepare_data.make_scikit_input_kcv(feat_dict, k=k, set_prefix='ind_set')
    #X_train, y_train, X_test, y_test = prepare_data.make_scikit_input_kcv(feat_dict, k=k, set_prefix='comb_set')
    X_train, y_train, X_test, y_test = prepare_data.make_scikit_input_kcv(feat_dict, k=k, set_prefix='pfam_set')
    #X_train, y_train, X_test, y_test = prepare_data.make_scikit_input_kcv(feat_dict, k=k)

    X, y = prepare_data.make_scikit_input(feat_dict)
    N = X.shape[1]
    r = np.zeros((N,1))
    for i in xrange(N):
        r[i] = sp.stats.pearsonr(X[:,i], y)[0]
   
    #print r
    feat_lst = []
    for f in flist:
        if 'ind' in f:
            continue
        feat = f.split('/')[-1].replace('.txt', '').replace('_full', '')
        if feat == 'cm_feat':
            feat_lst += [feat + '_co']
            feat_lst += [feat + '_nw_%s' % i for i in range(10)]
            feat_lst += [feat + '_nd_%s' % i for i in range(1,12)]
        else:
            feat_lst += [feat]
    #print feat_lst, len(feat_lst), N
    fig = pl.figure()
    pl.grid(True)
    ax = fig.add_subplot(111)
    ax.bar(np.arange(N) + 1, r, align='center')
    pl.xticks(np.arange(N+1), [''] + feat_lst, rotation=90, size='x-small')
    pl.savefig("feat_cor_tm.pdf", bbox_inches="tight")
    pl.close()

    """
    for col in X.T:
        print sp.stats.pearsonr(col, y)[0]
    print ''
    for col_i in X.T:
        for col_j in X.T:
            print sp.stats.pearsonr(col_i, col_j)[0]
        print ''
    """
    score_dict, y_dict, topC_lin, topC_rbf, topgamma_rbf, topnest_rf, topsamples_rf = run_cv(X_train, y_train)

    for key, val in score_dict.iteritems():
        #print '%s: %s -> avg=%s' % (key, val, np.mean(val))
        print '%s: %s' % (key, np.mean(val))

    print max(map(np.mean, score_dict.values()))
    print min(map(np.mean, score_dict.values()))

    max_key = max(score_dict.iterkeys(), key=(lambda key: np.mean(score_dict[key])))
    y_cv = np.hstack(y_dict[max_key])
    y_train_np = np.hstack((y_train[1],y_train[2],y_train[3],y_train[4],y_train[5]))
    print y_cv
    print y_train_np
    print r2_score(y_train_np, y_cv)
    print mean_squared_error(y_train_np, y_cv)
    print sp.stats.pearsonr(y_train_np, y_cv)[0]
    pl.scatter(y_train_np, y_cv, c='k', label='%s - cor=%.2f' % (max_key, sp.stats.pearsonr(y_train_np, y_cv)[0]))
    pl.title(sys.argv[-1] + '_cv')
    pl.xlabel("Labels")
    pl.ylabel("Predictions")
    pl.legend(loc=2, prop={'size':10})
    pl.savefig(sys.argv[-1] + '_cv.png')
    #pl.show()
    pl.close()

    fold_lst = range(1,k+1)
    X_train_all = []
    for i in fold_lst:
        X_train_all += X_train[i]
    X_train_all = np.array(X_train_all)
    scaler = StandardScaler().fit(X_train_all)
    joblib.dump(scaler, 'scaler.pkl')
    print X_train_all[0]
    print X_train_all.mean(axis=0)
    X_train_all = scaler.transform(X_train_all)

    y_train_all = []
    for i in fold_lst:
        y_train_all += y_train[i]
    y_train_all = np.array(y_train_all)

    svr_rbf = SVR(kernel='rbf', C=topC_rbf, gamma=topgamma_rbf, max_iter=200000)
    fit_rbf = svr_rbf.fit(X_train_all, y_train_all)
    joblib.dump(svr_rbf, 'rbf.pkl')


    X_test = np.array(X_test)
    y_test = np.array(y_test)


    #print X_test.mean(axis=0)
    #print X_test[0]
    X_test = scaler.transform(X_test)
    #print scaler.mean_

    print len(X_test)
    print len(y_test)
    print len(X_train_all)


    svr_rbf = SVR(kernel='rbf', C=topC_rbf, gamma=topgamma_rbf, max_iter=200000)
    #y_rbf = svr_rbf.fit(X_train_all, y_train_all).predict(np.array(X_test))
    fit_rbf = svr_rbf.fit(X_train_all, y_train_all)
    y_rbf = fit_rbf.predict(X_test)
    print len(y_rbf)
    print r2_score(y_test, y_rbf)
    print mean_squared_error(y_test, y_rbf)
    print sp.stats.pearsonr(y_test, y_rbf)[0]
    pl.scatter(y_test, y_rbf, c='k', label='RBF kernel - cor=%.2f' % sp.stats.pearsonr(y_test, y_rbf)[0])
    pl.title(sys.argv[-1] + '_rbf')
    pl.xlabel("Labels")
    pl.ylabel("Predictions")
    pl.legend(loc=2, prop={'size':10})
    pl.savefig(sys.argv[-1] + '_rbf.png')
    #pl.show()
    pl.close()

    svr_lin = SVR(kernel='linear', C=topC_lin, max_iter=200000)
    #y_lin = svr_lin.fit(X_train_all, y_train_all).predict(np.array(X_test))
    fit_lin = svr_lin.fit(X_train_all, y_train_all)
    y_lin = fit_lin.predict(X_test)
    print len(y_lin)
    print r2_score(y_test, y_lin)
    print mean_squared_error(y_test, y_lin)
    print sp.stats.pearsonr(y_test, y_lin)[0]
    pl.scatter(y_test, y_lin, c='k', label='Linear kernel - cor=%.2f' % sp.stats.pearsonr(y_test, y_lin)[0])
    pl.title(sys.argv[-1] + '_lin')
    pl.xlabel("Labels")
    pl.ylabel("Predictions")
    pl.legend(loc=2, prop={'size':10})
    pl.savefig(sys.argv[-1] + '_lin.png')
    #pl.show()
    pl.close()
 
    rf = RandomForestRegressor(n_estimators=topnest_rf,
            min_samples_leaf=topsamples_rf)
    fit_rf = rf.fit(X_train_all, y_train_all)
    y_rf = fit_rf.predict(X_test)
    print len(y_rf)
    print r2_score(y_test, y_rf)
    print mean_squared_error(y_test, y_rf)
    print sp.stats.pearsonr(y_test, y_rf)[0]
    pl.scatter(y_test, y_rf, c='k', label='Random Forest - cor=%.2f' % sp.stats.pearsonr(y_test, y_rf)[0])
    pl.title(sys.argv[-1] + '_rf')
    pl.xlabel("Labels")
    pl.ylabel("Predictions")
    pl.legend(loc=2, prop={'size':10})
    pl.savefig(sys.argv[-1] + '_rf.png')
    #pl.show()
    pl.close()

