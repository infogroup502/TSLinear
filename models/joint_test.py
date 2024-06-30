import torch
from torch import nn
import numpy as np
from torch.nn import Parameter, ParameterList
# import tensorly as tl
# tl.set_backend('pytorch')

import matplotlib.pyplot as plt

import math
torch.pi=math.pi
torch.e=math.e
torch.inf=math.inf
torch.nan=math.nan
import torch.fft as fft

from models.weight_matrix import w_cheng,w_jia,w_ju

torch.backends.cudnn.enabled = False

import torch.nn.functional as F

#这个备份之后立马就做离散傅里叶变换
class joint_test(nn.Module):
    def __init__(self, dimension,dime_nihe,count, a_3,c,pred_len):
        super(joint_test, self).__init__()
        self.device = 'cuda'
        self.dime = dimension  # 总维度

        self.count=count
        self.a_3 = a_3
        self.c=c
        self.pred_len=pred_len
        self.dime_nihe=dime_nihe


        self.dan_nihe_1 = nn.ModuleList(nn.Sequential(
                                                        nn.Linear(self.a_3, self.a_3),
            # nn.BatchNorm1d(self.a_3),
            # nn.ELU(),
            # nn.Dropout(0.5),
            # nn.Linear(self.a_3, self.a_3),
            # nn.BatchNorm1d(self.a_3),
            # nn.ELU(),
            # nn.Dropout(0.5),
            # nn.Linear(self.a_3, self.a_3),
                                                      nn.BatchNorm1d(self.a_3),
                                                      nn.ELU(),
                                                      nn.Dropout(0.5),
                                                      nn.Linear(self.a_3, self.pred_len),
                                                      ) for i in range(self.dime_nihe))


        #上下文部分
        self.linear_w = torch.nn.Parameter(torch.randn(1,self.c, self.a_3).to(self.device).requires_grad_())
        self.linear_b = torch.nn.Parameter(torch.randn(1,self.c, self.a_3).to(self.device).requires_grad_())
        self.shuzhihua = torch.nn.Parameter(
            torch.randn((self.count - 1) * self.a_3, self.a_3).to(self.device).requires_grad_())
        nn.init.kaiming_uniform_(self.shuzhihua, a=math.sqrt(5))

    def forward(self,x,flag,label):  # x: B x T x input_dims

        for i in range(len(x)):
            if (i == 0):
                t = x[i]
            else:
                t = torch.cat([t, x[i]], dim=1)
        t= torch.matmul(t,self.shuzhihua)

        t = F.normalize(t, dim=-1)
        ###############################
        # 常规的上下文操作
        for i in range(self.c - 1, t.shape[0]):
            temp_1 = t[i - self.c + 1:i + 1]
            temp_1=temp_1.reshape(1,temp_1.shape[0],-1)
            if (i == self.c - 1):
                rep = temp_1
            else:
                rep = torch.cat([rep, temp_1], dim=0)

        # 补全处理的上下文操作
        if (flag == True and self.c != 1):
            for i in range(self.c - 1):
                temp_1 = t[0:i + 1]
                temp = t[i].repeat(self.c - (i + 1)).reshape(-1, self.a_3)
                temp_1 = torch.cat([temp_1, temp], dim=0)
                temp_1 = temp_1.reshape(1, temp_1.shape[0], -1)
                if (i == 0):
                    pre = temp_1
                else:
                    pre = torch.cat([pre, temp_1], dim=0)
            rep = torch.cat([pre, rep], dim=0)
        rep=self.linear_w*rep+self.linear_b
        rep=torch.sum(rep,dim=1)
        #################################
        # rep=t

        for i in range(self.dime_nihe):
            temp = self.dan_nihe_1[i](rep)  # 将综合表征乘上不同的权重得做单变量拟合
            temp = temp.reshape(temp.shape[0], 1, -1)
            if (i == 0):
                y_1 = temp
            else:
                y_1 = torch.cat([y_1, temp], dim=1)
        y_1 = y_1.permute(0, 2, 1)
        y_1 = y_1.reshape(y_1.shape[0], -1)
        return y_1,rep
    def parallel(self,x,flag,label):  # x: B x T x input_dims

        for i in range(len(x)):
            if (i == 0):
                t = x[i]
            else:
                t = torch.cat([t, x[i]], dim=2)
        ####################并行处理
        batchsize=t.shape[0]
        t=t.reshape(-1,t.shape[2])
        #####################
        t= torch.matmul(t,self.shuzhihua)
        #####################并行处理
        t=t.reshape(batchsize,-1,t.shape[1])
        ########################

        t = F.normalize(t, dim=-1)
        # 常规的上下文操作
        for i in range(self.c - 1, t.shape[1]):
            temp_1 = t[:,i - self.c + 1:i + 1]
            temp_1=temp_1.unsqueeze(0)    #############这里和上面的forward不太一样
            if (i == self.c - 1):
                rep = temp_1
            else:
                rep = torch.cat([rep, temp_1], dim=0)

        # 补全处理的上下文操作
        if (flag == True and self.c != 1):
            for i in range(self.c - 1):
                temp_1 = t[:,0:i + 1]
                temp = t[:,i].unsqueeze(1).repeat(1,self.c - (i + 1),1)     #############这里和上面的forward不太一样
                temp_1 = torch.cat([temp_1, temp], dim=1)
                temp_1 = temp_1.unsqueeze(0)       #############这里和上面的forward不太一样
                if (i == 0):
                    pre = temp_1
                else:
                    pre = torch.cat([pre, temp_1], dim=0)
            rep = torch.cat([pre, rep], dim=0)


        rep=self.linear_w.unsqueeze(0)*rep+self.linear_b.unsqueeze(0)
        rep=torch.sum(rep,dim=2)
        ###########################并行化
        rep=rep.permute(1,0,2)
        y_1=0
        return y_1,rep

