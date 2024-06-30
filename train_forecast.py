import torch
import numpy as np
import argparse
import time
import datetime
from TSLinear_forecast import TSLinear
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
import warnings
warnings.filterwarnings("ignore")
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
                        default="{'epoch_0': 1, 'epoch_1': 19, 'epoch_2': 1, 'epoch_3': 1, 'text_epoch': 32}",
                        help="Dictionary containing epoch values.")
    parser.add_argument('--a_3', type=int, default=900, help="Value for dimension")
    parser.add_argument('--c', type=int, default=11, help="Value for windo length")
    parser.add_argument('--pred_len', type=int, default=3, help="Prediction length")
    parser.add_argument('--port', type=int, default=22, help="NUmber of segmentation")
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

    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(
            args.dataset)
    train_data = data[:, train_slice]




    data=data.reshape(data.shape[1],data.shape[2])
    data=data.reshape(1,data.shape[0],-1)
    data_train = data[:, train_slice]
    data_test = data[:, test_slice]

    args.epoch_dict = parse_epoch_dict(args.epoch_dict)
    args.num = len(args.epoch_dict)-1
    model = TSLinear(
        input_dims=data_train.shape[2],
        time_step_train=data_train.shape[1],
        device=device,
        n_covar=n_covariate_cols,
        data_train=data_train,
        num=args.num,
        a_3=args.a_3,
        c=args.c,
        pred_len=args.pred_len,
        port=args.port,
        p=args.p,
        max_len=args.max_len,
        epoch_dict=args.epoch_dict
    )
    model.output()

    t = time.time()
    ep = model.fit(
        data_train,data_test
    )
    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")


    print(args.dataset)
    model.output()

    ################################
    #
    # rate=0.2
    # num_noisy_elements = int(rate * data[train_slice].size)
    # noisy_indices = np.random.choice(data[train_slice].size, num_noisy_elements, replace=False)
    # noise_level = 1 * data[train_slice].std()
    # noise = np.random.normal(0, noise_level, num_noisy_elements)
    # data[train_slice].flat[noisy_indices] += noise
    # data_1 = data[train_slice]
    ################################
    # print(f'rate: {rate}')

    out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens,
                                                   n_covariate_cols)





