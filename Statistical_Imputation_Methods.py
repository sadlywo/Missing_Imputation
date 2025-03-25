import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import missingno as msn
import sys
from sklearn.impute import KNNImputer      #这是sklearn自带库内的统计学方法
from sklearn.linear_model import BayesianRidge, LinearRegression, LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import impyute
from sklearn.metrics import r2_score
# *********************************        ////////////////   统计学方法   /////////////////////////        *****************************************
def Mean_Imputation_Of_Missing_Data(Data, Average_Imputation_Flag=0):
    # 此函数用于将缺失数据的Data进行数据填补，Data是具有缺失值NaN的时间序列，使用平均填补法 ，第二个参数决定了使用全部平均还是使用前后单个平均  0 代表是全部平均， 1代表前后单个平均
    # if(Data.isnull().any(axis=0)).isempty()== False:
    #     return "无缺失值"
    # else:

    Temp_Data = Data.copy()
    if Average_Imputation_Flag == 0:
        All_Average_Imputation_Data = Temp_Data.fillna(value=Data.mean(), inplace=False)   #直接使用现有的API调用填补
        return All_Average_Imputation_Data
    else:
        # 这里是非默认的情况下,使用上下值的平均进行填补
        NULL_Index = np.where(pd.isna(Temp_Data))[0]  # 按照行来判断是否有空值,得到索引值
        Mean_Result = []
        for i in iter(NULL_Index):
            if i == 0:  # 避免了第一个是null超出边界
                j = i + 1
                while Data.loc[j].isnull().any():
                    j = j + 1
                Upper_Number = Temp_Data.loc[j]
                Down_Number = Temp_Data.loc[j]
            elif i == NULL_Index[-1]:  # 避免了最后一个是null的情况超出边界
                j = i - 1
                while Temp_Data.loc[j].isnull().any():
                    j = j - 1
                Upper_Number = Temp_Data.loc[j]
                Down_Number = Temp_Data.loc[j]
            else:
                j = i - 1
                while Temp_Data.loc[j].isnull().any():
                    j = j - 1
                Upper_Number = Temp_Data.loc[j]
                j = i + 1
                while Temp_Data.loc[j].isnull().any():
                    if j == NULL_Index[-1]:
                        break
                    j = j + 1
                if j == NULL_Index[-1]:
                    Down_Number = Upper_Number
                else:
                    Down_Number = Temp_Data.loc[j]  # 避免了最后一直重复是 null的情况
            # Upper_Number = Data.loc[i-1]
            # Down_Number = Data.loc[i+1]
            Mean_Number = (Upper_Number + Down_Number) / 2
            Mean_Result.append(Mean_Number)  # 将需要插补的平均数据append到创建的list中
        Temp_Data = Data
        Temp_Data.loc[NULL_Index] = Mean_Result  # 首次进行平均处理，但是可能存在两个NaN数据在一起的情况
        if Temp_Data.isna().any().any():  # any指的是只要有一个true就是true，这里如果有就是说明有两个NaN在一起的情况
            Temp_Data.fillna(method='ffill')
            return Temp_Data
        else:
            return Temp_Data
def Median_Imputation_Of_Missing_Data(Data):
    #这是中位数填补方法，
    Temp_Data = Data.copy()
    Median = Temp_Data.median()
    Temp_Data.fillna(Median,inplace=True)
    return Temp_Data
def Mode_Imputation_Of_Missing_Data(Data):
    #这是众数填补方法
    Temp_Data = Data.copy()
    Mode = Temp_Data.mode()
    Temp_Data.fillna(Mode,inplace=True)
    return Temp_Data
def Near_Imputation_Of_Missing_Data(Data,Fill_Direction="Forward"):
    #默认前向填充，指的就是使用前一个数据去填补找到的NaN  ， Flag代表 填补是使用前一个数据还是使用后一个数据
    Temp_Data = Data.copy()#拷贝一份数据，防止改变原始数据
    if Fill_Direction == "Forward":
        #使用前向填充，method = ffill
        for i in range(Data.shape[1]):
            Temp_Data.iloc[:,i].fillna(method='ffill',inplace = True)
        Temp_Data.fillna(method = 'ffill',inplace=True)  #填充完成后第一个可能还是NaN，所以要进行二次排查
        if Temp_Data.isna().any().any() == True:
            #此时说明还有空值，并且空值应该是从第一个开始重复，因此再使用一次向后填充即可
            Temp_Data.fillna(method= 'bfill',inplace=True)  #后向填充
    elif Fill_Direction == "Backward":
        #使用后向填充 ，method = bfill
        for i in range(Data.shape[1]):
            Temp_Data.iloc[:,i].fillna(method='bfill',inplace = True)
        Temp_Data.fillna(method= 'bfill',inplace=True)
        if Temp_Data.isna().any().any() == True:
            Temp_Data.fillna(method = 'ffill',inplace=True)  #进行前向填充
    else:
        Temp_Data = []
        print("前向填补或后向填补补缺算法输入有误")
    if Temp_Data.isna().sum().any() == True:
        #此时说明用列来填充不合适
        if Fill_Direction == "Forward":
            Temp_Data.fillna(method='backfill',inplace=True)  # 后向填充
        elif Fill_Direction == "Backward":
            Temp_Data.fillna(method='pad',inplace=True)
    return Temp_Data
def EM_Imputation_Of_Missing_Data(Missing_Data):
    #该函数包下要求使用ndarray类型
    Temp_Data = Missing_Data.copy()
    Columns_Name = Temp_Data.columns.tolist()
    Temp_Array = Temp_Data.values  #变为ndarray类型
    Out_Data = impyute.em(Temp_Array,loops=30)  #50是循环次数
    Temp_Out = pd.DataFrame(Out_Data)
    Temp_Out.columns = Columns_Name
    return Temp_Out   #E

def MA_Imputation_Of_Missing_Data(Missing_Data,windowsize = 3):
    Temp_Data = Missing_Data.copy()
    Columns_Name = Temp_Data.columns.tolist()
    Temp_Array = Temp_Data.values  #变为ndarray类型
    Out_Data = impyute.moving_window(Temp_Array,nindex=None, wsize=windowsize)  #50是循环次数
    Temp_Out = pd.DataFrame(Out_Data)
    Temp_Out.columns = Columns_Name
    return Temp_Out   #E
def Regression_Imputation_Of_Missing_Data(Missing_Data,regression_model = "Linear", RoundDown_Number = 4):
    # RoundDown_Number 代表的是向下取整的数字大小，默认认为是3，也可以改成2，这样可以对不同的序列大小进行判断
    random.seed(12345)  # 设置随机种子
    if Missing_Data.shape[1] <= 2:
        RoundDown_Number = 1 #如果列数太少，可能导致生成不了初始填补的矩阵，会报错，因此太小的时候对其重新赋值
    Temp_Data = Missing_Data.copy()
    Rand_Fill_Number = Temp_Data.shape[1]   #目前生成的类型 没有单独一列完整的，后续可以补充上，如果没有完整一列，需要通过随机数先生成填补结果
    Null_Columns_Number = Temp_Data.isna().any().sum()  #得到 有多少数量的列数是包含 缺失值的  ,注意这里是先any再sum
    if Null_Columns_Number <=2:    #缺失的列数太少直接返回 填补的结果即可
        RoundDown_Number = 1
        Temp_Median = Temp_Data.median()
        Temp_Data.fillna(Temp_Median,inplace = True)  #得到初始填补后的原始数据，之后从该序列出发，该序列应该
        return Temp_Data
     ###          -------------- 得到训练模型，根据选择的不同，得到不同的 训练模型 ---------------          #########
    if regression_model == "Linear":
        Regression_Model = LinearRegression()
    elif regression_model == "Logistic":
        Regression_Model = LogisticRegression()
    elif regression_model == "Bayes":
        Regression_Model = BayesianRidge()
        ##         -------------- 后面是对缺失数据的处理，回归的输入输出的处理  ----------  ##
    if Null_Columns_Number < Missing_Data.shape[1]:
        #此时代表缺失的列数和总的列数不等，代表这时候本来就有不缺失的列
        Initial_Fill_Columns_List = list(Temp_Data.isna().any()[~Temp_Data.isna().any()].index) #找到不含有缺失值的列list
        Residual_Fill_Columns_List = list(Temp_Data.isna().any()[Temp_Data.isna().any()].index)  #找到含有缺失值的列list  ，index
        # 这样再找到对应的行名字就可以得到 对应的输入输出表
        Initial_Fill_Row_List = list(Temp_Data.isna().any(axis=1)[~Temp_Data.isna().any(axis=1)].index)  #找到不含有缺失值的行index
        Residual_Fill_Row_List = list(Temp_Data.isna().any(axis=1)[Temp_Data.isna().any(axis=1)].index)  #找到含有缺失值的行index
        Input_X_Row = Initial_Fill_Row_List
        Input_X_Columns = Initial_Fill_Columns_List
        Input_Y_Row = Initial_Fill_Row_List
        Input_Y_Coumns = Residual_Fill_Columns_List
        Input_X = Temp_Data.loc[Input_X_Row,Input_X_Columns]  #得到输入的DataFrame作为X， 这里的X应该是对应 行列均无缺失，完整的列
        Input_Y = Temp_Data.loc[Input_Y_Row,Input_Y_Coumns]   #得到输入的Y

        Output_X_Row = Residual_Fill_Row_List
        Output_X_Columns = Initial_Fill_Columns_List
        Output_Y_Row = Residual_Fill_Row_List
        Output_Y_Columns = Residual_Fill_Columns_List
        Output_X = Temp_Data.loc[Output_X_Row,Output_X_Columns]
        Output_Y = Temp_Data.loc[Output_Y_Row,Output_Y_Columns]

        Regression_Model.fit(Input_X,Input_Y)   #训练对应的模型
        Prediction_Y = Regression_Model.predict(Output_X)   #根据模型结果输出预测的值
        Prediction_Result = pd.DataFrame(Prediction_Y,index=Output_Y_Row,columns=Output_Y_Columns)
        # 使用上面的结果对缺失的值进行插补，使用fillna函数进行插补
        Temp_Data.fillna(Prediction_Result,inplace=True)
        return Temp_Data   #返回补缺结果矩阵，结束
    elif Null_Columns_Number == Missing_Data.shape[1]:   ##########  -------------  此时多一步，即创建随机选取的列进行初始插补----------- ######
        # 使用rand生成原始数据一半列数的完整列，将其作为输入特征，训练回归器件
        Initial_Fill_Number = round(Null_Columns_Number/RoundDown_Number)  #一半向下取整  // 或者改成除3向下取整
        Initial_Fill_Columns_List = random.sample(range(0,Rand_Fill_Number),Initial_Fill_Number)   #初始填补的列
        Initial_Fill_Number_Row = round(Missing_Data.shape[0]/RoundDown_Number)
        Initial_Fill_Row_List = random.sample(range(0, Missing_Data.shape[0]), Initial_Fill_Number_Row)  # 初始填补的列
        for columns_number in Initial_Fill_Columns_List:
            Temp_Median = Temp_Data.iloc[:,columns_number].median()
            Temp_Data.iloc[:, columns_number].fillna(Temp_Median,inplace = True)  #得到初始填补后的原始数据，之后从该序列出发，该序列应该
        # 除了对列进行初始插补外，还需要对行进行初始化插补    /////////////////////   =========== 加上这一段后误差会很大  ---========  /////////////
        # for rows_number in Initial_Fill_Row_List:
        #     Temp_Median = Temp_Data.iloc[rows_number,:].median()
        #     Temp_Data.iloc[rows_number,:].fillna(Temp_Median,inplace=True)

        Residual_Fill_Number = Null_Columns_Number-Initial_Fill_Number  #得到剩余的列的数量
        Residual_Fill_Columns_List = list(set(range(0,Rand_Fill_Number))-set(Initial_Fill_Columns_List))  #得到两个list的差集，得到还需要填补的列
        # 这里是对其初始填补之后的结果
        Initial_Fill_Columns_List = list(Temp_Data.isna().any()[~Temp_Data.isna().any()].index)  # 找到不含有缺失值的列list
        Residual_Fill_Columns_List = list(Temp_Data.isna().any()[Temp_Data.isna().any()].index)  # 找到含有缺失值的列list  ，index
        # 这样再找到对应的行名字就可以得到 对应的输入输出表
        Initial_Fill_Row_List = list(
            Temp_Data.isna().any(axis=1)[~Temp_Data.isna().any(axis=1)].index)  # 找到不含有缺失值的行index
        Residual_Fill_Row_List = list(
            Temp_Data.isna().any(axis=1)[Temp_Data.isna().any(axis=1)].index)  # 找到含有缺失值的行index
        Input_X_Row = Initial_Fill_Row_List
        Input_X_Columns = Initial_Fill_Columns_List
        Input_Y_Row = Initial_Fill_Row_List
        Input_Y_Colmns = Residual_Fill_Columns_List
        Input_X = Temp_Data.loc[Input_X_Row,Input_X_Columns]  #得到输入的DataFrame作为X， 这里的X应该是对应 行列均无缺失，完整的列
        Input_Y = Temp_Data.loc[Input_Y_Row,Input_Y_Colmns]   #得到输入的Y

        Output_X_Row = Residual_Fill_Row_List
        Output_X_Columns = Initial_Fill_Columns_List
        Output_Y_Row = Residual_Fill_Row_List
        Output_Y_Columns = Residual_Fill_Columns_List
        Output_X = Temp_Data.loc[Output_X_Row,Output_X_Columns]
        Output_Y = Temp_Data.loc[Output_Y_Row,Output_Y_Columns]
        Regression_Model.fit(Input_X,Input_Y)   #训练对应的模型
        Prediction_Y = Regression_Model.predict(Output_X)   #根据模型结果输出预测的值
        Prediction_Result = pd.DataFrame(Prediction_Y,index=Output_Y_Row,columns=Output_Y_Columns)
        # 使用上面的结果对缺失的值进行插补，使用fillna函数进行插补
        Temp_Data.fillna(Prediction_Result,inplace=True)
        return Temp_Data
def MICE_Imputation_Of_Missing_Data(Missing_Data,estimator = BayesianRidge(),Max_iter_Times = 30):
    #使用库函数IterativeImputer 进行插补，这是使用回归模型进行插补
    # MICE包（链式方程的多变量插补）
    # MissingData代表有缺失的数据，Max——iter代表最大循环次数,estimator 设置线性估计器，这里默认设置为贝叶斯估计器 Data_Colums_Number是数据列数
    Imputer = IterativeImputer(estimator=estimator, max_iter=Max_iter_Times,random_state=10)
    #randomstate代表随机种子，可以改变种子编号
    Temp_Data = Missing_Data.copy()
    # if Data_Colums_Number == 1:
    #     Temp_Data['index'] = Temp_Data.index  #增加一列索引
    Columns_Name = Temp_Data.columns.to_list()
    Temp_Array = Temp_Data.values
    Temp_Imputation_Array = Imputer.fit_transform(Temp_Array)
    Temp_Out = pd.DataFrame(Temp_Imputation_Array)
    Temp_Out.columns = Columns_Name
    return Temp_Out
