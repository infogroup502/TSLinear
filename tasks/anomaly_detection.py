import numpy as np
import time
from sklearn.metrics import *
import bottleneck as bn
import torch


# consider delay threshold and missing segments
def get_range_proba(predict, label, delay=7):
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict


# set missing = 0
def reconstruct_label(timestamp, label):
    timestamp = np.asarray(timestamp, np.int64)
    index = np.argsort(timestamp)

    timestamp_sorted = np.asarray(timestamp[index])
    interval = np.min(np.diff(timestamp_sorted))

    label = np.asarray(label, np.int64)
    label = np.asarray(label[index])

    idx = (timestamp_sorted - timestamp_sorted[0]) // interval

    new_label = np.zeros(shape=((timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1,), dtype=np.int)
    new_label[idx] = label

    return new_label


def eval_ad_result(test_pred_list, test_labels_list, delay):
    labels = []
    pred = []
    for test_pred, test_labels, in zip(test_pred_list, test_labels_list):
        assert test_pred.shape == test_labels.shape
        test_pred = get_range_proba(test_pred, test_labels, delay)
        labels.append(test_labels)
        pred.append(test_pred)
    labels = np.concatenate(labels)
    pred = np.concatenate(pred)
    return {
        'f1': f1_score(labels, pred),
        'precision': precision_score(labels, pred),
        'recall': recall_score(labels, pred),
        'acc ': accuracy_score(labels, pred),
        'roc_auc' : roc_auc_score(labels, pred),
        # 'auc': auc(labels, pred),
    }


def np_shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def eval_anomaly_detection(model,pre_len,batchsize, train_data, test_data, test_labels, delay, c):
    rate = 0.1
    t = time.time()

    data = np.concatenate([train_data, test_data], axis=0)
    if (pre_len < c):
        raise Exception(" pre_len  <   c  !!")
    temp=data
    pre = torch.zeros(pre_len, data.shape[1])
    data = np.concatenate([pre, data], axis=0)

    input = np.stack([data[i:1 + data.shape[0] + i - (pre_len+1)] for i in range(pre_len+1)], axis=1)
    input_mask=input.copy()
    input_mask[:,-1,:]=0
    for i in range(0,input.shape[0],batchsize):
        x=input[i:i+batchsize].copy()
        x_mask= input_mask[i:i + batchsize].copy()
        item = model.encode(
            x_mask,
        )
        item=item[:,-1]
        if (i == 0):
            full_repr = item
        else:
            full_repr = np.concatenate([full_repr, item], axis=0)

        item = model.encode(
            x,
        )
        item = item[:, -1]
        if (i ==0):
            full_repr_wom = item
        else:
            full_repr_wom = np.concatenate([full_repr_wom, item], axis=0)

        if (i % batchsize*100 == 0):
            print("    ", i, " /  ", data.shape[0])

    s=0


    train_repr = full_repr[:len(train_data)]
    test_repr = full_repr[len(train_data):]
    train_repr_wom = full_repr_wom[:len(train_data)]
    test_repr_wom = full_repr_wom[len(train_data):]
    train_err = np.abs(train_repr_wom - train_repr).sum(axis=1)
    test_err = np.abs(test_repr_wom - test_repr).sum(axis=1)

    ###############################原始时间序列做表征
    # train_err = np.abs(train_data).sum(axis=1)
    # test_err = np.abs(test_data).sum(axis=1)
    ###############################

    res_log = []
    labels_log = []
    a=np.concatenate([train_err, test_err])
    b=bn.move_mean(np.concatenate([train_err, test_err]), 21)
    ma = np_shift(bn.move_mean(np.concatenate([train_err, test_err]), 21), 1)
    train_err_adj = (train_err - ma[:len(train_err)]) / ma[:len(train_err)]
    test_err_adj = (test_err - ma[len(train_err):]) / ma[len(train_err):]
    train_err_adj = train_err_adj[22:]

    thr = np.mean(train_err_adj) + rate * np.std(train_err_adj)
    test_res = (test_err_adj > thr) * 1

    for i in range(len(test_res)):
        if i >= delay and test_res[i - delay:i].sum() >= 1:
            test_res[i] = 0

    res_log.append(test_res)
    labels_log.append(test_labels)

    t = time.time() - t

    eval_res = eval_ad_result(res_log, labels_log, delay)
    # eval_res['infer_time'] = t
    return res_log, eval_res


# def eval_anomaly_detection(model,pre_len,batchsize, train_data, test_data, test_labels, delay, c):
#     rate = 0.1
#     t = time.time()
#
#     # len_test = 200
#     # train_data = train_data[0:len_test]
#     # test_data = test_data[0:len_test]
#     # test_labels = test_labels[0:len_test]
#     ##############################################################
#     data = np.concatenate([train_data, test_data], axis=0)
#     if (pre_len < c):
#         raise Exception(" pre_len  <   c  !!")
#
#     pre = torch.zeros(pre_len, data.shape[1])
#     data = np.concatenate([pre, data], axis=0)
#
#     for i in range(pre_len, data.shape[0]):
#         input = data[i - pre_len:i + 1].copy()
#         input_mask = data[i - pre_len:i + 1].copy()
#         input_mask[-1] = 0
#
#         item = model.encode(
#             input_mask,
#         ).squeeze()  ####把当前时间戳给遮盖掉
#         if (i == pre_len):
#             full_repr = item[-1].reshape(1, -1)
#         else:
#             full_repr = np.concatenate([full_repr, item[-1].reshape(1, -1)], axis=0)
#
#         item = model.encode(
#             input,
#         ).squeeze()
#
#         if (i == pre_len):
#             full_repr_wom = item[-1].reshape(1, -1)
#         else:
#             full_repr_wom = np.concatenate([full_repr_wom, item[-1].reshape(1, -1)], axis=0)
#         if (i % 1000 == 0):
#             print("    ", i, " /  ", data.shape[0])
#
#     train_repr = full_repr[:len(train_data)]
#     test_repr = full_repr[len(train_data):]
#     train_repr_wom = full_repr_wom[:len(train_data)]
#     test_repr_wom = full_repr_wom[len(train_data):]
#     train_err = np.abs(train_repr_wom - train_repr).sum(axis=1)
#     test_err = np.abs(test_repr_wom - test_repr).sum(axis=1)
#
#     ###############################原始时间序列做表征
#     # train_err = np.abs(train_data).sum(axis=1)
#     # test_err = np.abs(test_data).sum(axis=1)
#     ###############################
#
#     res_log = []
#     labels_log = []
#     ma = np_shift(bn.move_mean(np.concatenate([train_err, test_err]), 21), 1)
#     train_err_adj = (train_err - ma[:len(train_err)]) / ma[:len(train_err)]
#     test_err_adj = (test_err - ma[len(train_err):]) / ma[len(train_err):]
#     train_err_adj = train_err_adj[22:]
#
#     thr = np.mean(train_err_adj) + rate * np.std(train_err_adj)
#     test_res = (test_err_adj > thr) * 1
#
#     for i in range(len(test_res)):
#         if i >= delay and test_res[i - delay:i].sum() >= 1:
#             test_res[i] = 0
#
#     res_log.append(test_res)
#     labels_log.append(test_labels)
#
#     t = time.time() - t
#
#     eval_res = eval_ad_result(res_log, labels_log, delay)
#     # eval_res['infer_time'] = t
#     return res_log, eval_res



def eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data,
                                     all_test_labels, all_test_timestamps, delay):
    t = time.time()

    all_data = {}
    all_repr = {}
    all_repr_wom = {}
    for k in all_train_data:
        all_data[k] = np.concatenate([all_train_data[k], all_test_data[k]])
        all_repr[k] = model.encode(
            all_data[k].reshape(1, -1, 1),
            mask='mask_last',
            casual=True,
            sliding_length=1,
            sliding_padding=200,
            batch_size=256
        ).squeeze()
        all_repr_wom[k] = model.encode(
            all_data[k].reshape(1, -1, 1),
            casual=True,
            sliding_length=1,
            sliding_padding=200,
            batch_size=256
        ).squeeze()

    res_log = []
    labels_log = []
    timestamps_log = []
    for k in all_data:
        data = all_data[k]
        labels = np.concatenate([all_train_labels[k], all_test_labels[k]])
        timestamps = np.concatenate([all_train_timestamps[k], all_test_timestamps[k]])

        err = np.abs(all_repr_wom[k] - all_repr[k]).sum(axis=1)
        ma = np_shift(bn.move_mean(err, 21), 1)
        err_adj = (err - ma) / ma

        MIN_WINDOW = len(data) // 10
        thr = bn.move_mean(err_adj, len(err_adj), MIN_WINDOW) + 1 * bn.move_std(err_adj, len(err_adj), MIN_WINDOW)
        res = (err_adj > thr) * 1

        for i in range(len(res)):
            if i >= delay and res[i - delay:i].sum() >= 1:
                res[i] = 0

        res_log.append(res[MIN_WINDOW:])
        labels_log.append(labels[MIN_WINDOW:])
        timestamps_log.append(timestamps[MIN_WINDOW:])
    t = time.time() - t

    eval_res = eval_ad_result(res_log, labels_log, timestamps_log, delay)
    eval_res['infer_time'] = t
    return res_log, eval_res

