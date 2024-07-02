python train_anom.py --dataset MSL --epoch_dict  "{'epoch_0': 10, 'epoch_1': 10, 'epoch_2': 13, 'epoch_3': 4, 'text_epoch': 10}"  --a_3  300  --c 1 --pred_len 3 --port 62 --name MSL --name_1 C-1

python train_anom.py --dataset MSL --epoch_dict  "{'epoch_0': 2, 'epoch_1': 2, 'epoch_2': 1, 'epoch_3': 1, 'text_epoch': 1}"  --a_3  300  --c 15 --pred_len 7 --port 83 --name MSL --name_1 D-15

python train_anom.py --dataset MSL --epoch_dict  "{'epoch_0': 10, 'epoch_1': 10, 'epoch_2': 10, 'epoch_3': 10, 'text_epoch': 10}"  --a_3  300  --c 1 --pred_len 1 --port 70 --name MSL --name_1 F-4