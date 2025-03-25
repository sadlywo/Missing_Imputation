import pandas as pd
import matplotlib.pyplot as plt  #画图

from Generate_Miss_Data import Gengerate_Incomplete_Data_From_Time_Series

#本文件主要用于将 收集到的IMU数据进行缺失数据人工生成，显示缺失的表现
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

    File_Sub = File_Sub_5

    FileFolder_All = FileFolder + File_Sub + "/"+ File_Sub + ".txt"   #得到需要读取的文件名的绝对路径，这里直接使用相对路径下的文件名读取，同时修改对应的文件名为相应的文件夹名字方便读取
    # 根据原始数据的列数来判断使用哪一个列名，有两种情况，一种是六列的，一种是九列的，分别代表六轴和九轴

    Temp_Data = pd.read_csv(FileFolder_All,sep = ' ',header=None)
    if Temp_Data.shape[1] == 6:
        Column_Name = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']  # 六轴的列名
    elif Temp_Data.shape[1] == 9:
        Column_Name = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z','Magn_X','Magn_Y','Magn_Z']  # 六轴的列名
    Temp_Data.columns = Column_Name #修改列名
    print("***********************************")

    Missing_Data_Random = Gengerate_Incomplete_Data_From_Time_Series(Temp_Data,0.4,"Random")
    Missing_Data_Random.plot(subplots=True,
            figsize=(10, 2 * len(Missing_Data_Random.columns)),  # 根据列数调整图像高度
            layout=(len(Missing_Data_Random.columns), 1))  # 每列占据一行
    Missing_Data_InRow = Gengerate_Incomplete_Data_From_Time_Series(Temp_Data,0.4,"InRow")
    Missing_Data_InRow.plot(subplots=True,
            figsize=(10, 2 * len(Missing_Data_InRow.columns)),  # 根据列数调整图像高度
            layout=(len(Missing_Data_InRow.columns), 1))  # 每列占据一行
    Missing_Data_InCol = Gengerate_Incomplete_Data_From_Time_Series(Temp_Data, 0.4, "InCol")
    Missing_Data_InCol.plot(subplots=True,
            figsize=(10, 2 * len(Missing_Data_InCol.columns)),  # 根据列数调整图像高度
            layout=(len(Missing_Data_InCol.columns), 1))  # 每列占据一行
    Missing_Data_Univariant = Gengerate_Incomplete_Data_From_Time_Series(Temp_Data, 0.4, "Univariant")
    Missing_Data_Univariant.plot(subplots=True,
            figsize=(10, 2 * len(Missing_Data_Univariant.columns)),  # 根据列数调整图像高度
            layout=(len(Missing_Data_Univariant.columns), 1))  # 每列占据一行

    plt.show()







