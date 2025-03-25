import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

# 下面是从其他文件中引入对应的定义模型
from DL_Model import MLP_Imputer
from DL_Model import RNN_Imputer
from DL_Model import VAE_Imputer
from DL_Model import LSTM_Imputer
from DL_Model import GAIN_Imputer
from DL_Model import DBN_Imputer
from DL_Model import GRU_Imputer

## 下面是从其他文件中引入的相关的训练函数，用于训练深度学习模型
from Train_Model import Train_DL_Model
##  --------------------------------    数据预处理  方法  -------------------------------##  深度学习方法需要预处理

# *********************************        ////////////////  深度学习方法   //////////////////////////        **************************************
def MLP_Imputation_Of_Missing_Data(Missing_Data):
    Temp_Data = Missing_Data.copy()
    imputer = MLP_Imputer(epochs=50)
    imputer.fit(Temp_Data)
    Filled_Data = imputer.transform(Temp_Data)
    return Filled_Data
def RNN_Imputation_Of_Missing_Data(Missing_Data):
    Temp_Data = Missing_Data.copy()
    imputer = RNN_Imputer(epochs=50)
    imputer.fit(Temp_Data)
    Filled_Data = imputer.transform(Temp_Data)
    return Filled_Data
def VAE_Imputation_Of_Missing_Data(Missing_Data):   #变分自编码器
    Temp_Data = Missing_Data.copy()
    imputer = VAE_Imputer(epochs=50,hidden_dim=128)  #设置最大循环次数和隐藏神经元个数
    imputer.fit(Temp_Data)
    Filled_Data = imputer.transform(Temp_Data)
    return Filled_Data
def LSTM_Imputation_Of_Missing_Data(Missing_Data):
    Temp_Data = Missing_Data.copy()
    imputer = LSTM_Imputer(seq_len=3,epochs=50, batch_size=32)  # 设置时间序列前序填充长度5，最大循环次数，batchsize
    imputer.fit(Temp_Data)
    Filled_Data = imputer.transform(Temp_Data)
    return Filled_Data
def GAIN_Imputation_Of_Missing_Data(Missing_Data):
    Temp_Data = Missing_Data.copy()
    imputer = GAIN_Imputer(epochs=50, batch_size=256)  # 设置时间序列前序填充长度5，最大循环次数，batchsize
    imputer.fit(Temp_Data)
    Filled_Data = imputer.transform(Temp_Data)
    return Filled_Data
def DBN_Imputation_Of_Missing_Data(Missing_Data):  #深度信念网络
    Temp_Data = Missing_Data.copy()
    imputer = DBN_Imputer(hidden_dims=[64,32], pretrain_epochs=20, finetune_epochs=50)  # 设置时间序列前序填充长度5，最大循环次数，batchsize
    imputer.fit(Temp_Data)
    Filled_Data = imputer.transform(Temp_Data)
    return Filled_Data
def GRU_Imputation_Of_Missing_Data(Missing_Data):
    Temp_Data = Missing_Data.copy()
    imputer = GRU_Imputer(seq_len=3,epochs=50, batch_size=32)  #
    imputer.fit(Temp_Data)
    Filled_Data = imputer.transform(Temp_Data)
    return Filled_Data

def VBN_Imputation_Of_Missing_Data(Missing_Data):
    return []


