import numpy as np
import time
import torch
from tasks import _eval_protocols as eval_protocols
from torch.utils.data import TensorDataset, DataLoader
import gc
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
# 提升比较明显的模型
def generate_pred_samples(features, data, pred_len, drop=0):
    n = data.shape[1]
    #######
    # pred_len=1
    features = features[:, :-pred_len]
    for i in range(pred_len):
        temp=data[:, i:1+n+i-pred_len]
    labels = np.stack([ data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
    #######
    # drop=0
    features = features[:, drop:]
    labels = labels[:, drop:]
    return features.reshape(-1, features.shape[-1]), \
            labels.reshape(-1, labels.shape[2]*labels.shape[3])

def cal_metrics(pred, target):
    return {
        'MSE': ' ' + str(((pred - target) ** 2).mean()) + ' ',
        'MAE': ' ' + str(np.abs(pred - target).mean()) + ' '
    }

def eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols):
    # 创建一个简单的DataFrame

    padding = 200

    t = time.time()
    all_repr = model.encode(
        data
    )
    all_repr=all_repr.reshape(1,all_repr.shape[0],-1)


    #########原始数据作为表征
    # all_repr=data
    ##########

    ts2vec_infer_time = time.time() - t

    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]

    train_data = data[:, train_slice, n_covariate_cols:]
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]
    # train_data = data[:, train_slice,:]
    # valid_data = data[:, valid_slice, :]
    # test_data = data[:, test_slice,:]

    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    for pred_len in pred_lens:
    # for pred_len in [96]:
        # pred_len=672
        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len,drop=padding)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)


        t = time.time()
        lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
        lr_train_time[pred_len] = time.time() - t

        # del train_features
        # del train_labels
        # del valid_features
        # del valid_labels
        # gc.collect()

        t = time.time()
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t

        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        test_pred = test_pred.reshape(ori_shape)
        test_labels = test_labels.reshape(ori_shape)
        # np.savez('sin+cos_NEW-DATA-1.T15.npz', array1=test_pred, array2=test_labels)

        # if test_data.shape[0] > 1:
        #     test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
        #     test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
        # else:
        #     test_pred_inv = scaler.inverse_transform(test_pred)
        #     test_labels_inv = scaler.inverse_transform(test_labels)

        # out_log[pred_len] = {
        #     'norm': test_pred,
        #     'raw': test_pred_inv,
        #     'norm_gt': test_labels,
        #     'raw_gt': test_labels_inv
        # }
        ours_result[pred_len] = {
            'norm': cal_metrics(test_pred, test_labels),
            # 'raw': cal_metrics(test_pred_inv, test_labels_inv)
        }
        # print("           ",pred_len,ours_result[pred_len])
        print("", pred_len, ours_result[pred_len])
        # print(pred_len, ours_result[pred_len])

    eval_res = {
        'ours': ours_result,
        'ts2vec_infer_time': ts2vec_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    return out_log, eval_res

# #原模型
# def generate_pred_samples(features, data, pred_len, drop=0):
#     n = data.shape[1]
#     #######
#     # pred_len=1
#     features = features[:, :-pred_len]
#     labels = np.stack([data[:, i:1 + n + i - pred_len] for i in range(pred_len)], axis=2)[:, 1:]
#     #######
#     # drop=0
#     features = features[:, drop:]
#     labels = labels[:, drop:]
#     return features.reshape(-1, features.shape[-1]), \
#         labels.reshape(-1, labels.shape[2] * labels.shape[3])
#
#
# def cal_metrics(pred, target):
#     return {
#         'MSE': ((pred - target) ** 2).mean(),
#         'MAE': np.abs(pred - target).mean()
#     }
#
#
# def eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols):
#     padding = 200
#
#     t = time.time()
#     all_repr = model.encode(
#         data,
#         casual=True,
#         sliding_length=1,
#         sliding_padding=padding,
#         batch_size=256
#     )
#     ts2vec_infer_time = time.time() - t
#
#     train_repr = all_repr[:, train_slice]
#     valid_repr = all_repr[:, valid_slice]
#     test_repr = all_repr[:, test_slice]
#
#     train_data = data[:, train_slice, n_covariate_cols:]
#     valid_data = data[:, valid_slice, n_covariate_cols:]
#     test_data = data[:, test_slice, n_covariate_cols:]
#
#     ours_result = {}
#     lr_train_time = {}
#     lr_infer_time = {}
#     out_log = {}
#     for pred_len in pred_lens:
#         train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
#         valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
#         test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)
#
#         t = time.time()
#         lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
#         lr_train_time[pred_len] = time.time() - t
#
#         t = time.time()
#         test_pred = lr.predict(test_features)
#         lr_infer_time[pred_len] = time.time() - t
#
#         ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
#         test_pred = test_pred.reshape(ori_shape)
#         test_labels = test_labels.reshape(ori_shape)
#
#         if test_data.shape[0] > 1:
#             test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
#             test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
#         else:
#             test_pred_inv = scaler.inverse_transform(test_pred)
#             test_labels_inv = scaler.inverse_transform(test_labels)
#
#         out_log[pred_len] = {
#             'norm': test_pred,
#             'raw': test_pred_inv,
#             'norm_gt': test_labels,
#             'raw_gt': test_labels_inv
#         }
#         ours_result[pred_len] = {
#             'norm': cal_metrics(test_pred, test_labels),
#             'raw': cal_metrics(test_pred_inv, test_labels_inv)
#         }
#
#     eval_res = {
#         'ours': ours_result,
#         'ts2vec_infer_time': ts2vec_infer_time,
#         'lr_train_time': lr_train_time,
#         'lr_infer_time': lr_infer_time
#     }
#     return out_log, eval_res
#
#
