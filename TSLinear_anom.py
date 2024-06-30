import torch
from torch.utils.data import TensorDataset, DataLoader
from models import joint,joint_test
import numpy as np
from sklearn.cluster import KMeans


# import keras
torch.backends.cudnn.enabled = False
torch.set_printoptions(precision=8)
import matplotlib.pyplot as plt



#三个个尺度
class TSLinear:
    def __init__(
            self,
            input_dims,
            data_train,
            time_step_train,
            num,
            a_3,
            c,
            pred_len,
            port,
            p,
            max_len,
            epoch_dict,
            device='cuda',
            lr=0.001,

            after_iter_callback=None,
            after_epoch_callback=None,
    ):

        super().__init__()
        self.device = device
        self.lr =0.001
        self.num = num
        self.size_dict = {}
        for i in range(self.num):
            key = f'size_{i}'
            self.size_dict[key] = 0
        self.size_dict['text_size'] = 0

        ep = 1
        self.epoch_dict = epoch_dict
        self.a_3 = a_3
        self.c = c
        self.p = p
        self.max_len = max_len
        self.n_covar = 0
        self.pred_len=pred_len
        ############top  k处理
        self.port = port    ###切分的份数  此次改动的方向就是希望这个参数往大了调，期望效果变好，但不能太大而报错
        data_train = torch.from_numpy(data_train).to(torch.float)

        remain = data_train.shape[0] % self.port
        if (remain != 0):
            data_train = data_train[0:-remain]
        data_train = data_train.reshape(self.port, -1, data_train.shape[1])

        xf = torch.fft.rfft(data_train, dim=0)
        frequency_list = abs(xf)
        # _, top_list = torch.topk(frequency_list, k)
        top_list = frequency_list.detach().cpu().numpy()

        period = top_list

        period = np.mean(np.mean(period, axis=-1),axis=0)

        period = data_train.shape[1] // period
        period = np.sort(period, axis=-1)[1:-1]

        j = 0

        for i in range(period.shape[0]):
            if (int(period[i]) > self.c):
                length = period.shape[0] - i - 1
                break
        self.dif = length // len(self.size_dict)
        while (True):
            if (int(period[i]) > self.c):
                self.size_dict['size_%d' % (j)] = int(period[i])
                j = j + 1
                i += self.dif
                if (j == (len(self.size_dict)-1)):
                    self.size_dict['text_size'] = int(period[i])
                    break
            else:
                i += 1
        ############


        self.count = len(self.size_dict)
        self.epo=0
        self.list_model = []
        #sin模块处定义
        for i in range(self.count):
            if(i!=self.count-1):
                size=self.size_dict['size_%d'%(i)]
                epoch=self.epoch_dict['epoch_%d'%(i)]
                self.epo += (epoch)
                self.list_model.append(joint(dimension=input_dims,dime_nihe=input_dims,a_3=self.a_3,pred_len=self.pred_len,p=self.p).to(self.device))
            else:
                size = self.size_dict['text_size']
                epoch = self.epoch_dict['text_epoch']
                self.epo += (epoch)
                self.list_model.append(joint_test(dimension=input_dims,dime_nihe=input_dims,count=self.count,a_3=self.a_3,c=self.c,pred_len=self.pred_len).to(self.device))
        #上下文模块定义

        self.n_epochs = 0
        self.n_iters = 0
        self.input_dims=input_dims

    def fit(self, data):

        ##################多尺度第一阶段训练##########################
        #########################################################
        epoch = 0
        pred_len = self.pred_len


        data_y = data

        # data_look=data[:,8:]
        data_x = data[0:data_y.shape[0]]

        # data_valid_y = np.stack(
        #     [data_valid[i:1 + data_valid.shape[0] + i - pred_len, 8:] for i in range(pred_len)], axis=1)
        # data_valid_y = data_valid_y.reshape(data_valid_y.shape[0], -1)
        # data_valid_x = data_valid[0:data_valid_y.shape[0]]

        loss_valid = 10
        epoch_valid = 0

        for j in range(self.count-1):


            self.model=self.list_model[j]
            self.size=self.size_dict['size_%d'%(j)]
            self.epoch=self.epoch_dict['epoch_%d'%(j)]
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            self.model.train()
            for i in range(self.epoch):
                loss = 0
                itr = 0

                index = 0
                train_dataset = TensorDataset(torch.from_numpy(data_x).to(torch.float),
                                              torch.from_numpy(data_y).to(torch.float))
                train_loader = DataLoader(train_dataset, batch_size=self.size,
                                          shuffle=False, drop_last=True)
                for x, y in train_loader:
                    optimizer.zero_grad()

                    y =y.cuda()  # 多变量
                    x = x.numpy()

                    output_1, repr = self.model(x,True)

                    mse_1 = torch.nn.MSELoss()
                    loss_1 = mse_1(output_1, y)
                    loss_1 = loss_1.requires_grad_()
                    loss_1.backward()

                    loss+=loss_1
                    itr+=1

                    optimizer.step()


                    index += self.size
                    # index += (int)(3*self.time2vec_size/4)


                    torch.cuda.empty_cache()
                epoch = epoch + 1
                print('train_', j, epoch, "/", self.epo, "  loss ", loss / itr)

        data_y = np.stack([data[i:1 + data.shape[0] + i - pred_len] for i in range(pred_len)], axis=1)
        # data_y = data_y[:, 1:]
        data_y = data_y.reshape(data_y.shape[0], -1)
        #获取上一阶段的表征
        self.repr=[]
        for j in range(self.count-1):
            self.list_model[j].eval()
            _,item=self.list_model[j](data_x,False)
            self.repr.append(item.detach())

        self.size = self.size_dict['text_size']
        self.epoch = self.epoch_dict['text_epoch']
        self.model=self.list_model[-1]
        self.model.train()

        optimizer = torch.optim.Adam( self.model.parameters(), lr=self.lr)
        data_y = torch.from_numpy(data_y).to(torch.float)
        for i in range(self.epoch):
            flag=True
            loss=0
            itr=0
            for j in range(0, data_y.shape[0] -self.size, self.size - self.c + 1):
                x = []
                for e in range(self.count-1):
                    x.append(self.repr[e][j:j + self.size])
                if (flag == True):
                    y = data_y[j:j +self.size].cuda()
                else:
                    y= data_y[j+ self.c - 1:j + self.size].cuda()  #
                optimizer.zero_grad()

                # print(itr)

                output_1,_ = self.model(x, flag,True)
                if(flag==True): flag=False


                mse_1 = torch.nn.MSELoss()
                loss_1 = mse_1(output_1, y)
                loss_1 = loss_1.requires_grad_()
                loss_1.backward(retain_graph=True)

                loss+=loss_1
                itr+=1
                optimizer.step()


                torch.cuda.empty_cache()
            epoch = epoch + 1
            print('train_', 4, epoch, "/", self.epo, "  loss ", loss/itr)



        return self.c

    def encode(self, data):

        self.repr = []
        for j in range(self.count - 1):
            self.list_model[j].eval()
            _, item = self.list_model[j].parallel(data,False)
            self.repr.append(item.detach())

        self.model.eval()

        _, rep = self.model.parallel(self.repr,True, False)


        featu = rep.cpu().detach().numpy()


        return featu


    # def encode(self, data):
    #
    #     self.repr = []
    #     for j in range(self.count - 1):
    #         self.list_model[j].eval()
    #         _, item = self.list_model[j](data, False)
    #         self.repr.append(item.detach())
    #
    #     self.model.eval()
    #
    #     # max_len = self.size_dict['text_size']
    #     max_len = self.max_len
    #     flag = True
    #     for i in range(0, data.shape[0] - self.c, max_len - self.c + 1):
    #         x_input = []
    #         for j in range(len(self.repr)):
    #             x_input.append(self.repr[j][i:i + max_len])
    #         _, rep = self.model(x_input, flag, False)
    #         if (flag == True): flag = False
    #         if (i == 0):
    #             featu = rep.cpu().detach().numpy()
    #         else:
    #             featu = np.concatenate((featu, rep.cpu().detach().numpy()), axis=0)
    #
    #
    #     return featu




    def output(self):
        print(f'port:{self.port}')
        for key,value in self.epoch_dict.items():
            print(f"{key}={value}")
        print('self.a_3: ',self.a_3,' self.c ',self.c,' self.pred_len ',self.pred_len)

