import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from TSLinear_anom import TSLinear
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
# import warnings
# warnings.filterwarnings("ignore")
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import column_or_1d
import pandas as pd
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.mcd import MCD
from pyod.models.lscp import LSCP


import ast
def parse_epoch_dict(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Invalid dictionary format")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='NEW-DATA-1.T15',help='The dataset name')
    parser.add_argument('--epoch_dict', type=str,
                        default="{'epoch_0': 4, 'epoch_1': 2, 'epoch_2': 9, 'epoch_3': 2, 'text_epoch': 1}",
                        help="Dictionary containing epoch values.")
    parser.add_argument('--a_3', type=int, default=1200, help="Value for dimension")
    parser.add_argument('--c', type=int, default=19, help="Value for windo length")
    parser.add_argument('--pred_len', type=int, default=1, help="Prediction length")
    parser.add_argument('--port', type=int, default=5, help="NUmber of segmentation")
    parser.add_argument('--p', type=float, default=0.5, help="Probability value")
    parser.add_argument('--max_len', type=int, default=1000, help="Maximum length")
    parser.add_argument('--name', type=str, default='SWaT', help="Name of the dataset or model")
    parser.add_argument('--name_1', type=str, default='SWaT', help="Name of the subdataset or model")
    parser.add_argument('--batchsize', type=int, default=640, help="Batch size for training")
    parser.add_argument('--pre_len', type=int, default=20, help="Length of preprocessing window")
    parser.add_argument('--num', type=int, default=4, help="no use")
    parser.add_argument('--gpu', type=int, default=0,
                        help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, help='The learning rate (defaults to 0.001)', default=0.001)
    parser.add_argument('--seed', type=int, default=42, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=8,
                        help='The maximum allowed number of threads used by this process')
    args = parser.parse_args()

    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    print('Loading data... ', end='')


    name = args.name
    name_1=args.name_1
    batchsize =args.batchsize
    pre_len =args.pre_len
    if (name == 'SMD' or name == 'SMAP' or name=='MSL'):
        train_data = pd.read_csv(f'datasets/anomaly/{name}/{name_1}_train.csv', sep=',').to_numpy()
        test_data = pd.read_csv(f'datasets/anomaly/{name}/{name_1}_test.csv', sep=',').to_numpy()
        test_label = pd.read_csv(f'datasets/anomaly/{name}/{name_1}_labels.csv', sep=',').to_numpy()
        test_label = np.any(test_label == 1, axis=1).astype(int)
        s=0
    elif (name == 'Anomaly_Detection_Falling_People'):
        train_data = pd.read_csv(f'datasets/anomaly/{name}/train.csv', sep=',').to_numpy()
        test_data = pd.read_csv(f'datasets/anomaly/{name}/test.csv', sep=',').to_numpy()
        train_data = train_data[:, :-1]
        test_label = test_data[:, -1]
        test_data = test_data[:, :-1]
    else:
        train_data = pd.read_csv(f'datasets/anomaly/{name}/train.csv', sep=',').to_numpy()
        test_data = pd.read_csv(f'datasets/anomaly/{name}/test.csv', sep=',').to_numpy()
        test_label = pd.read_csv(f'datasets/anomaly/{name}/labels.csv', sep=',').to_numpy()
        test_label = np.any(test_label == 1, axis=1).astype(int)
        s=0
    data = np.concatenate((train_data, test_data), axis=0)
    #############################################################

    args.epoch_dict = parse_epoch_dict(args.epoch_dict)
    args.num = len(args.epoch_dict) - 1
    model = TSLinear(
        input_dims=train_data.shape[1],
        data_train=train_data,
        time_step_train=train_data.shape[0],
        device=device,
        num=args.num,
        a_3=args.a_3,
        c=args.c,
        port=args.port,
        p=args.p,
        pred_len=args.pred_len,
        max_len=args.max_len,
        epoch_dict=args.epoch_dict,
    )
    t = time.time()
    c = model.fit(
        train_data
    )
    t = time.time() - t

    # print(lidu)



    out, eval_res = tasks.eval_anomaly_detection(model,pre_len, batchsize,train_data, test_data, test_label, 7,c)
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
    print(name)
    if (name == 'SMD' or name == 'SMAP' or name == 'MSL'):
        print(name_1)
    model.output()
    print(" 下游算法补全长度len ",pre_len)
    # print('Evaluation result:', eval_res)
    for key, value in eval_res.items():
        print(f"{key}:  {value}")
    # print("此处为部分代码加上eval()")
