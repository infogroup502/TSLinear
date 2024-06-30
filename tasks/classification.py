import numpy as np
from tasks import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score

#聚类任务的包
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.special import comb

def eval_classification(model,data, train_data, train_labels, test_data, test_labels, eval_protocol='linear'):
    assert train_labels.ndim == 1 or train_labels.ndim == 2

    #利用表征做任务
    repr=model.encode(data)
    train_repr = repr[0:train_data.shape[0], :]
    test_repr = repr[train_data.shape[0]:train_data.shape[0] + test_data.shape[0], :]

    #利用原始数据做任务
    # train_data=train_data[:,1:,:]
    # test_data = test_data[:, 1:, :]
    # train_repr = train_data.reshape(train_data.shape[0], -1)
    # test_repr = test_data.reshape(test_data.shape[0], -1)


    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    clf = fit_clf(train_repr, train_labels)

    acc = clf.score(test_repr, test_labels)
    if eval_protocol == 'linear':
        y_score = clf.predict_proba(test_repr)
    else:
        y_score = clf.decision_function(test_repr)
    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    auprc = average_precision_score(test_labels_onehot, y_score)



    return { 'acc': ' ' + str(acc) + ' ', 'auprc':' '+ str(auprc) + ' '}
    # return acc,auprc
