import numpy as np
from Generate_Miss_Data import Gengerate_Incomplete_Data_From_Time_Series
import pandas as pd
from Statistical_Imputation_Methods import Mean_Imputation_Of_Missing_Data
from Statistical_Imputation_Methods import MA_Imputation_Of_Missing_Data
from Statistical_Imputation_Methods import EM_Imputation_Of_Missing_Data
from Statistical_Imputation_Methods import Near_Imputation_Of_Missing_Data
from Statistical_Imputation_Methods import Mode_Imputation_Of_Missing_Data
from Statistical_Imputation_Methods import Median_Imputation_Of_Missing_Data
from Statistical_Imputation_Methods import Regression_Imputation_Of_Missing_Data
from Statistical_Imputation_Methods import MICE_Imputation_Of_Missing_Data

from Machine_Learning_Imputation_Methods import KNN_Imputation_Of_Missing_Data
from Machine_Learning_Imputation_Methods import MF_Imputation_Of_Missing_Data
from Machine_Learning_Imputation_Methods import RF_Imputation_Of_Missing_Data
from Machine_Learning_Imputation_Methods import SVR_Imputation_Of_Missing_Data
from Machine_Learning_Imputation_Methods import DT_Imputation_Of_Missing_Data
from Machine_Learning_Imputation_Methods import SoftImpute_Imputation_Of_Missing_Data
from Machine_Learning_Imputation_Methods import Low_Rank_Imputation_Of_Missing_Data
from Machine_Learning_Imputation_Methods import PCA_Imputation_Of_Missing_Data
######################################   ------------- 下面是评价指标的计算函数 ----------------- ###############################################
def Cal_RMSE(Data_obs, Data_sim):
    # Data_obs 是原始数据，观测的  ，Data_sim是预测或者处理后的数据
    # 默认二者数据都是dataframe
    RMSE_Data = 0
    Row_Number = Data_obs.shape[0]
    Col_Number = Data_obs.shape[1] #得到行列值

    for i in range(0,Row_Number):
        for j in range(0,Col_Number):
            RMSE_Data = RMSE_Data + (Data_sim.iloc[i,j]-Data_obs.iloc[i,j])**2
    RMSE = (RMSE_Data /(Row_Number*Col_Number)) ** 0.5
    # RMSE = RMSE_Data
    return RMSE
def Cal_MAE(Data_obs, Data_sim):
    # Data_obs 是原始数据，观测的  ，Data_sim是预测或者处理后的数据
    # 默认二者数据都是dataframe
    MAE_Data = 0
    Row_Number = Data_obs.shape[0]
    Col_Number = Data_obs.shape[1] #得到行列值
    for i in range(0,Row_Number):
        for j in range(0,Col_Number):
            MAE_Data = MAE_Data + abs(Data_sim.iloc[i,j]-Data_obs.iloc[i,j])
    MAE = (MAE_Data /(Row_Number*Col_Number))
    return MAE
def Cal_BIAS(Data_obs, Data_sim):
    # Data_obs 是原始数据，观测的  ，Data_sim是预测或者处理后的数据
    # 默认二者数据都是dataframe
    BIAS_Data = 0
    Row_Number = Data_obs.shape[0]
    Col_Number = Data_obs.shape[1] #得到行列值
    for i in range(0,Row_Number):
        for j in range(0,Col_Number):
            BIAS_Data = BIAS_Data + Data_sim.iloc[i,j]-Data_obs.iloc[i,j]
    BIAS = BIAS_Data
    return BIAS
def Cal_R2(Data_obs, Data_sim):
    #这是计算拟合优度R2的值（也叫决定系数） ,平均值函数是mean
    R2_Data = 0
    Row_Number = Data_obs.shape[0]
    Col_Number = Data_obs.shape[1]  # 得到行列值
    Mean_Data = Data_obs.mean()
    Upper = 0
    Down = 0
    for i in range(Col_Number):
        for j in range(Row_Number):
            Upper = Upper + (Data_sim.iloc[j,i] - np.mean(Data_obs.iloc[:,i])) ** 2
            Down = Down + (Data_obs.iloc[j, i] - np.mean(Data_obs.iloc[:, i])) ** 2
    Temp = Upper / Down
    R2_Data = Temp
    return R2_Data
def Cal_AE(Data_obs,Data_sim):
    AE_Data = 0
    Row_Number = Data_obs.shape[0]
    Col_Number = Data_obs.shape[1]  # 得到行列值
    Temp = 0
    Mean_Data = Data_obs.mean()  # 平均数据列
    Mean_Data = Mean_Data.values  #这样得到的就是平均的数据矩阵，就可以用列数字来进行排序
    for i in range(0,Col_Number):
        for j in range(0,Row_Number):
            Temp = Temp + abs(Data_obs.iloc[j,i]-Data_sim.iloc[j,i])
    AE_Data = Temp / (Row_Number * Col_Number) * 100
    return AE_Data
def Cal_D2(Data_obs,Data_sim):
    D2_Data = 0
    Row_Number = Data_obs.shape[0]
    Col_Number = Data_obs.shape[1]  # 得到行列值
    Upper = 0
    Down = 0
    Mean_Data = Data_obs.mean()  # 平均数据列
    Mean_Data = Mean_Data.values  #这样得到的就是平均的数据矩阵，就可以用列数字来进行排序
    for i in range(0,Col_Number):
        for j in range(0,Row_Number):
            Upper = Upper + (Data_obs.iloc[j,i]-Data_sim.iloc[j,i])**2
            Down = Down + (abs(Data_obs.iloc[j,i] - Mean_Data[i])+abs(Data_sim.iloc[j,i]-Mean_Data[i]))**2
    D2_Data = 1.0-Upper*1.0/Down
    return D2_Data
def Cal_MAPE(Data_obs,Data_sim):   #记得检查Cal_MAPE
    MAPE_Data = 0
    MAPE_Sum = 0
    Row_Number = Data_obs.shape[0]
    Col_Number = Data_obs.shape[1]  # 得到行列值
    for i in range(0,Col_Number):
        for j in range(0,Row_Number):
            MAPE_Sum = MAPE_Sum + abs((Data_obs.iloc[j,i]-Data_sim.iloc[j,i])/Data_obs.iloc[j,i])
    MAPE_Data = MAPE_Sum / (Row_Number*Col_Number) * 100.0
    return MAPE_Data
def Imputation_Result(Input_Data,Miss_Rate,Method_Name,Pattern_Name,Dict_Method,Dict_Pattern):
    # Input_Data代表的是原始输入数据，这里就是TestData，直接将想要原始输入的数据进行输入 ，主程序中只需要运行读取原始数据即可
    # MethodName指的是实现的各类方法，通过缩写记录各类方法，也可以在输入时进行增减达到不同输出结果
    # PatternName指的是实现的各类缺失数据模式，这里默认是三种
    # Miss_Rate 代表的是各个比例的输入
    Continuous_Number = 4
    Origin_Data = Input_Data.copy()
    Indicator = list()   #返回的矩阵，记录了各类方法的填补结果
    Random_Indicator = list()
    InRow_Indicator = list()
    InCol_Indicator = list()
    Univariant_Indicator = list()
    All_RMSE = []
    All_MAE = []
    All_BIAS = []
    All_MAPE = []
    All_AE = []
    for Miss_rate in Miss_Rate:  #循环记录不同比例缺失结果
        for i in Pattern_Name:
            locals()[i+'_miss'] = Gengerate_Incomplete_Data_From_Time_Series(Origin_Data,Miss_rate,i,Continuous_Number)  #注意这里没有考虑加入连续丢失数据大小，默认是5
        for i in Pattern_Name:
            for j in Method_Name:
                locals()[j+'_'+i]= globals()[j+"_Imputation_Of_Missing_Data"](locals()[i+'_miss'])
        for i in Pattern_Name:
            for j in Method_Name:
                Temp = locals()[j+'_'+i].copy() #得到变量
                locals()['RMSE_'+j+'_'+i] = Cal_RMSE(Origin_Data, locals()[j+'_'+i])
                locals()['MAE_'+j+'_' + i] = Cal_MAE(Origin_Data, locals()[j+'_'+i])
                locals()['BIAS_'+j+'_' + i] = Cal_BIAS(Origin_Data, locals()[j+'_'+i])
                locals()['MAPE_' + j + '_' + i] = Cal_MAPE(Origin_Data, locals()[j+'_'+i])
                locals()['AE_' + j + '_' + i] = Cal_AE(Origin_Data, locals()[j+'_'+i])
        for i in Pattern_Name:
            locals()['All_RMSE_'+i] = []
            locals()['All_MAE_'+i] = []
            locals()['All_BIAS_'+i] = []
            locals()['All_MAPE_'+i] = []
            locals()['All_AE_' + i] = []
        for j in Method_Name:
            for i in Pattern_Name:
                locals()['All_RMSE_' + i].append(locals()['RMSE_' + j + '_' + i])
                locals()['All_MAE_' + i].append(locals()['MAE_' + j + '_' + i])
                locals()['All_BIAS_' + i].append(locals()['BIAS_' + j + '_' + i])
                locals()['All_MAPE_' + i].append(locals()['MAPE_' + j + '_' + i])
                locals()['All_AE_' + i].append(locals()['AE_' + j + '_' + i])

        Indicator_All = []
        for i in Pattern_Name:
            All_RMSE.append(locals()['All_RMSE_' + i])
            All_MAE.append(locals()['All_MAE_' + i])
            All_BIAS.append(locals()['All_BIAS_' + i])
            All_MAPE.append(locals()['All_MAPE_' + i])
            All_AE.append(locals()['All_AE_' + i])
    All_RMSE_Frame = pd.DataFrame(All_RMSE)
    All_MAE_Frame = pd.DataFrame(All_MAE)
    All_BIAS_Frame = pd.DataFrame(All_BIAS)
    All_MAPE_Frame = pd.DataFrame(All_MAPE)
    All_AE_Frame = pd.DataFrame(All_AE)
    All_RMSE_Frame.rename(columns=Dict_Method,index=Dict_Pattern, inplace=True)  #修改DataFrame的列名和index
    All_MAE_Frame.rename(columns=Dict_Method,index=Dict_Pattern, inplace=True)
    All_BIAS_Frame.rename(columns=Dict_Method,index=Dict_Pattern, inplace=True)
    All_MAPE_Frame.rename(columns=Dict_Method,index=Dict_Pattern, inplace=True)
    All_AE_Frame.rename(columns=Dict_Method,index=Dict_Pattern, inplace=True)
        # for i in Pattern_Name:
        #     locals()[i + '_Indicator'].append(All_RMSE_Frame.iloc[0,:])  # 初始化记录结果矩阵
        #     locals()[i + '_Indicator'].append(All_MAE_Frame.iloc[0,:])
        #     locals()[i + '_Indicator'].append(All_BIAS_Frame.iloc[0, :])
        #     locals()[i + '_Indicator'].append(All_D2_Frame.iloc[0, :])
        #     locals()[i + '_Indicator'].append(All_AE_Frame.iloc[0, :])  # 初始化记录结果矩阵
    # Random_Indicator = pd.DataFrame(Random_Indicator)  # 转化为dataframe
    return All_RMSE_Frame,All_MAE_Frame,All_BIAS_Frame,All_MAPE_Frame,All_AE_Frame