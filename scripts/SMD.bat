python train_anom.py --dataset SMD --epoch_dict  "{'epoch_0': 1, 'epoch_1': 1, 'epoch_2': 1, 'epoch_3': 6, 'text_epoch': 1}"  --a_3  300  --c 3 --pred_len 9 --port 15 --name SMD --name_1 machine-1-1

python train_anom.py --dataset SMD --epoch_dict  "{'epoch_0': 5, 'epoch_1': 2, 'epoch_2': 6, 'epoch_3': 6, 'text_epoch': 2}"  --a_3  300  --c 11 --pred_len 21 --port 9 --name SMD --name_1 machine-1-3

python train_anom.py --dataset SMD --epoch_dict  "{'epoch_0': 3, 'epoch_1': 3, 'epoch_2': 6, 'epoch_3': 6, 'text_epoch': 6}"  --a_3  300  --c 1 --pred_len 12 --port 65 --name SMD --name_1 machine-2-2