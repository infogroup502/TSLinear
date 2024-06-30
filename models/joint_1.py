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

#�������֮�����������ɢ����Ҷ�任
class joint(nn.Module):
    def __init__(self, dimension,dime_nihe, a_3,pred_len,p):
        super(joint, self).__init__()
        self.device = 'cuda'
        self.dime = dimension  # ��ά��
        self.pred_len=pred_len
        self.dime_nihe=dime_nihe
        self.p=p

        self.a_3 = a_3
        # �������ԭʼ���ݵ�ӳ��

        self.tran_t = nn.Sequential(
            nn.Linear(self.dime, self.a_3),
            nn.BatchNorm1d(self.a_3),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(self.a_3, self.a_3),
        )

        self.b_wai_yinshe = nn.Sequential(
            nn.Linear(self.dime, self.a_3),
            nn.BatchNorm1d(self.a_3),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(self.a_3, self.a_3),
        )

        # ��Ӧ��ʽ�еĸ�������

        self.A = torch.nn.Parameter(torch.randn(1, self.a_3).to(self.device).requires_grad_())
        self.W =torch.nn.Parameter(torch.randn(1, self.a_3).to(self.device).requires_grad_())
        self.b_nei =torch.nn.Parameter(torch.randn(1, self.a_3).to(self.device).requires_grad_())

        self.sin = torch.sin
        self.cos = torch.cos
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
                                                    nn.Linear(self.a_3, 1),
                                                    ) for i in range(self.dime_nihe))

        self.drop=nn.Dropout(p)
    def forward(self, x,label):  # x: B x T x input_dims
        x = torch.from_numpy(x.astype(np.float32)).cuda()
        t = self.tran_t(x)
        #########################����ʵ��ȥ��
        t = self.W*t  # W*t
        t = self.b_nei+t  # W*t+b
        t_sin = self.sin(t)  # sin(W*t+b)
        t_cos = self.cos(t)
        t_sin = self.A * t_sin  # A*sin(W*t+b)  �˴��Ĺ��ܺ���������Բ㹦���غ���
        t_cos = self.A * t_cos

        b_wai = self.b_wai_yinshe(x)
        t =  t_sin + b_wai+t_cos + b_wai  # A*sin(W*t+b)+b
        t = F.normalize(t, dim=-1)
        ########################
        repr = t

        for i in range(self.dime_nihe):
            temp = self.dan_nihe_1[i](t)  # ���ۺϱ������ϲ�ͬ��Ȩ�ص������������
            temp = temp.reshape(temp.shape[0], 1, -1)
            if (i == 0):
                y_1 = temp
            else:
                y_1 = torch.cat([y_1, temp], dim=1)

        y_1 = y_1.reshape(y_1.shape[0], -1)

        return y_1,repr

    def parallel(self,x,label):
        x = torch.from_numpy(x.astype(np.float32)).cuda()
        #############################���л�
        batchsize=x.shape[0]
        x=x.reshape(-1,x.shape[2])
        #############################���л�
        t = self.tran_t(x)

        t = self.W * t  # W*t
        t = self.b_nei + t  # W*t+b
        t = self.sin(t)  # sin(W*t+b)
        t = self.A * t  # A*sin(W*t+b)  �˴��Ĺ��ܺ���������Բ㹦���غ���

        b_wai = self.b_wai_yinshe(x)
        t = t + b_wai  # A*sin(W*t+b)+b
        t = F.normalize(t, dim=-1)
        repr = t

        for i in range(self.dime_nihe):
            temp = self.dan_nihe_1[i](t)  # ���ۺϱ������ϲ�ͬ��Ȩ�ص������������
            temp = temp.reshape(temp.shape[0], 1, -1)
            if (i == 0):
                y_1 = temp
            else:
                y_1 = torch.cat([y_1, temp], dim=1)

        y_1 = y_1.reshape(y_1.shape[0], -1)
        #############################���л�
        y_1=y_1.reshape(batchsize,-1,y_1.shape[1])
        repr = repr.reshape(batchsize, -1, repr.shape[1])
        return y_1, repr
