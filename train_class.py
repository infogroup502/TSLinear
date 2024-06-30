import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from TSLinear_class import TSLinear
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
from scipy.interpolate import splprep, splev
from scipy.interpolate import interp1d


import ast
def parse_epoch_dict(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Invalid dictionary format")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ArrowHead',help='The dataset name')
    parser.add_argument('--epoch_dict', type=str,
                        default="{'epoch_0': 1, 'epoch_1': 1, 'epoch_2': 1, 'epoch_3': 10, 'text_epoch': 1}",
                        help="Dictionary containing epoch values.")
    parser.add_argument('--a_3', type=int, default=200, help="Value for dimension")
    parser.add_argument('--c', type=int, default=1, help="Value for windo length")
    parser.add_argument('--pred_len', type=int, default=1, help="Prediction length")
    parser.add_argument('--port', type=int, default=9, help="NUmber of segmentation")
    parser.add_argument('--p', type=float, default=0.5, help="Probability value")
    parser.add_argument('--max_len', type=int, default=1000, help="Maximum length")
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
    task_type = 'classification'
    train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)


# 原始版本
    #合并前记录测试集和训练集的分界线
    train_len=train_data.shape[0]
    #合并数据集并安排序号
    data = np.concatenate((train_data, test_data), axis=0)
    #重新划分数据集
    train_data=data[0:train_len]
    test_data=data[train_len:]
    ##############插值
    n_rows, n_cols,_ = train_data.shape
    if(n_rows<100):
        x_train=train_data.squeeze(-1)
        x_original = np.arange(n_rows)

        # 新的行数
        n_new_rows = 100
        x_new = np.linspace(0, n_rows - 1, n_new_rows)

        # 插值后的新矩阵
        new_data = np.zeros((n_new_rows, n_cols))

        for i in range(n_cols):
            f = interp1d(x_original, x_train[:, i], kind='linear')
            new_data[:, i] = f(x_new)
        train_data_temp= np.expand_dims(new_data,axis=-1)
    else:
        train_data_temp=train_data
    args.epoch_dict = parse_epoch_dict(args.epoch_dict)
    args.num=len(args.epoch_dict)-1
    model = TSLinear(
        input_dims=train_data_temp.shape[1],
        time_step_train=train_data_temp.shape[0],
        data_train=train_data_temp,
        num=args.num,
        a_3=args.a_3,
        c=args.c,
        pred_len=args.pred_len,
        port=args.port,
        p=args.p,
        max_len=args.max_len,
        epoch_dict=args.epoch_dict,
        device=device,

    )
    t = time.time()
    n_covariate_cols = model.fit(
        train_data_temp
    )
    print(args.dataset)
    model.output()


    t = time.time() - t
    # print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    result = tasks.eval_classification(model, data,train_data, train_labels, test_data, test_labels,
                     eval_protocol='svm')


    print(result)

