import warnings
import time
import pandas as pd
import matplotlib.pyplot as plt  #画图
from PreProcess_Data import Original_Data_Read  #引入
from PreProcess_Data import PreProcess_Data
from Generate_Miss_Data import Gengerate_Incomplete_Data_From_Time_Series
from Deep_Learning_Imputation_Methods import MLP_Imputation_Of_Missing_Data
from Deep_Learning_Imputation_Methods import RNN_Imputation_Of_Missing_Data
from Deep_Learning_Imputation_Methods import VAE_Imputation_Of_Missing_Data
from Deep_Learning_Imputation_Methods import LSTM_Imputation_Of_Missing_Data
from Deep_Learning_Imputation_Methods import GAIN_Imputation_Of_Missing_Data
from Deep_Learning_Imputation_Methods import DBN_Imputation_Of_Missing_Data
from Deep_Learning_Imputation_Methods import GRU_Imputation_Of_Missing_Data

from Cal_Attitude import AdaptiveAttitudeEstimator
from Cal_Attitude import calculate_attitude
from Euler_Change_To_Tra import euler_to_rotation_matrix
from Euler_Change_To_Tra import compute_trajectory
from Euler_Change_To_Tra import plot_3d_trajectory
from Euler_Change_To_Tra import calculate_trajectory
from Euler_Change_To_Tra import plot_2d_trajectory
if __name__ == "__main__":
    FileFolder = "./DataSet_Selected_All/"
    File_Sub_1 = "AWESOME_GINS"
    File_Sub_2 = "BASEPROD"
    File_Sub_3 = "NEURIT"
    File_Sub_4 = "OPPORTUNITY"
    File_Sub_5 = "OUTBACK"
    File_Sub_6 = "PAMAP"
    File_Sub_7 = "ROSARIO"
    File_Sub_8 = "WISDM"

    File_Total = [File_Sub_1,File_Sub_2,File_Sub_3,File_Sub_4,File_Sub_5,File_Sub_6,File_Sub_7,File_Sub_8]
    File_Frequency = [200,50,200,30,60,140,20,20]  #各个数据集中的采集频率，这个采集频率按照不同的设置是不同的，需要输入到滤波器中
    Dict_Pattern_Keys = File_Total  #得到模式的键值，和数值进行配对  ,这个是键
    Dict_Pattern = {k:v for k,v in zip(Dict_Pattern_Keys,File_Frequency)}   #这个是value  ,创建完成键值对

    File_Sub = File_Sub_8

    FileFolder_All = FileFolder + File_Sub + "/"+ File_Sub + ".txt"   #得到需要读取的文件名的绝对路径，这里直接使用相对路径下的文件名读取，同时修改对应的文件名为相应的文件夹名字方便读取
    # 根据原始数据的列数来判断使用哪一个列名，有两种情况，一种是六列的，一种是九列的，分别代表六轴和九轴

    Temp_Data = pd.read_csv(FileFolder_All,sep = ' ',header=None)
    if Temp_Data.shape[1] == 6:
        Column_Name = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']  # 六轴的列名
    elif Temp_Data.shape[1] == 9:
        Column_Name = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z','Magn_X','Magn_Y','Magn_Z']  # 六轴的列名
    Temp_Data.columns = Column_Name #修改列名
    print("***********************************")
    print(Temp_Data)
    Temp_Data.plot(subplots=True,
            figsize=(10, 2 * len(Temp_Data.columns)),  # 根据列数调整图像高度
            layout=(len(Temp_Data.columns), 1))  # 每列占据一行

    plt.tight_layout()  # 自动调整子图间距

    Acc_Data = Temp_Data.iloc[:,0:3]
    Gyro_Data = Temp_Data.iloc[:,3:6]
    if Temp_Data.shape[1] == 9:
        Mang_Data = Temp_Data.iloc[:,6:9]  #如果有九轴

    # filter = AttitudeEstimator(dt=0.01)   #采样时间0.01
    if Temp_Data.shape[1] == 6:
        Euler_Result = calculate_attitude(Acc_Data,Gyro_Data, None, Frequency= 1/Dict_Pattern[File_Sub])
    elif Temp_Data.shape[1] == 9:
        Euler_Result = calculate_attitude(Acc_Data,Gyro_Data,Mang_Data,Frequency= 1/Dict_Pattern[File_Sub])  #有九轴的时候调用
    print(Euler_Result)
    Euler_Result.plot(subplots=True,
            figsize=(10, 2 * len(Euler_Result.columns)),  # 根据列数调整图像高度
            layout=(len(Euler_Result.columns), 1))  # 每列占据一行

    # 计算轨迹并绘图
    # trajectory = compute_trajectory(Euler_Result)
    trajectory = calculate_trajectory(Euler_Result, step_size=0.1)
    # plot_3d_trajectory(trajectory)
    plot_2d_trajectory(trajectory)
    plt.show()
    # Data = Original_Data_Read(".","1_Activity_4_all.txt")
    #
    # print(Data)
    # print(type(Data))
    # Missing_Data = Gengerate_Incomplete_Data_From_Time_Series(Data,0.2,"Random")
    # Pre_Data = pd.DataFrame()
    # # Pre_Data,missing_indices,missing_cols,missing_rows,Mask_Matrix = PreProcess_Data(Missing_Data,'mean')
    # print(Pre_Data)
    # # print(missing_indices)
    # # print(missing_rows)
    # # print(missing_cols)
    # # print(Mask_Matrix)

    print("***********************************")
    print("")





