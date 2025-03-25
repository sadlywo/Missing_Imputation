import random
import sys
from sklearn.impute import IterativeImputer
import pandas as pd
import sklearn
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from scipy.linalg import svd   #引入奇异值分解算法
import numpy as np
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base  #修正版本不同导致的错误，需要加这一句不然会报错
from missingpy import MissForest
from sklearn.impute import KNNImputer      #这是sklearn自带库内的统计学方法
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from fancyimpute import SoftImpute,BiScaler    #调用fancy中的
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
# *********************************        ////////////////  机器学习方法   //////////////////////////        **************************************
def KNN_Imputation_Of_Missing_Data(Missing_Data,N_neighbors = 2,Data_Columns_Number = 1):
    # 此函数使用KNN方法对丢失的数据进行插补，KNN方法需要指定近邻数量大小，而如果只有一列数据，需要加入索引一列帮助KNN进行判断
    # 默认输入的Missing_Data是DataFrames类型，如果不是建议转换为该类型，函数内也提供判断和转换
    # Nneighbors是近邻的数量，Datacolumnsnumber是Missingdata的列数
    if type(Missing_Data) != pd.core.frame.DataFrame:
        Missing_Data = pd.DataFrame(Missing_Data)
    Temp_Data = Missing_Data.copy()#拷贝一份进行后续修改，防止更改已有的数据
    imputer = KNNImputer(n_neighbors=N_neighbors)  #创建imputer对象
    if Data_Columns_Number == 1:
        Temp_Data['index'] = Temp_Data.index
        #增加一列索引列，如果只有一行
    elif Data_Columns_Number == 0:
        return "列数有问题，一般不会发生"
    Columns_Name = Temp_Data.columns.to_list()  # 得到已有的列名
    Temp_Data_Array = Temp_Data.values  #需要把DataFrames转化为array类型，使用values得到array类型
    Temp_Data_Array_Imputation = imputer.fit_transform(Temp_Data_Array)  #得到插补后的矩阵,这样得到的是矩阵，将其转化为DataFrame
    Temp_Data_out = pd.DataFrame(Temp_Data_Array_Imputation)
    Temp_Data_out.columns = Columns_Name
    return Temp_Data_out
def MF_Imputation_Of_Missing_Data(Missing_Data,Data_Columns = 4):
    #输入的Missing_Data默认是DataFrame类型的  ,且默认不能是一行数据输入
    if Data_Columns>1:
        Temp_Data = Missing_Data.copy()
        Columns_Name = Temp_Data.columns.tolist()
        Temp_Array = pd.DataFrame(Temp_Data)
        Imputer = MissForest(max_features = 'sqrt',random_state=10)
        Temp_Imputation_Array = Imputer.fit_transform(Temp_Array)
        Temp_Out = pd.DataFrame(Temp_Imputation_Array)
        Temp_Out.columns = Columns_Name
        return Temp_Out
    elif Data_Columns == 1:
        Temp_Data = Missing_Data.copy()
        Temp_Data['index'] = Temp_Data.index
        Columns_Name = Temp_Data.columns.tolist()
        Temp_Array = pd.DataFrame(Temp_Data)
        Imputer = MissForest(max_features='sqrt',random_state=10)
        Temp_Imputation_Array = Imputer.fit_transform(Temp_Array)
        Temp_Out = pd.DataFrame(Temp_Imputation_Array)
        Temp_Out.columns = Columns_Name
        return Temp_Out
def RF_Imputation_Of_Missing_Data(Missing_Data, RoundDown_Number = 4,Initial_Fill_Method = 'Mean'):
    # RoundDown_Number 代表的是向下取整的数字大小，默认认为是3，也可以改成2，这样可以对不同的序列大小进行判断
    random.seed(12345)  #设置随机种子
    if Missing_Data.shape[1] <= 2:
        RoundDown_Number = 1 #如果列数太少，可能导致生成不了初始填补的矩阵，会报错，因此太小的时候对其重新赋值
    Temp_Data = Missing_Data.copy()
    Rand_Fill_Number = Temp_Data.shape[1]   #目前生成的类型 没有单独一列完整的，后续可以补充上，如果没有完整一列，需要通过随机数先生成填补结果
    Null_Columns_Number = Temp_Data.isna().any().sum()  #得到 有多少数量的列数是包含 缺失值的  ,注意这里是先any再sum
    if Null_Columns_Number <= 2:    #缺失的列数太少直接返回 填补的结果即可
        if Initial_Fill_Method == 'Mean':
            Temp_Mean = Temp_Data.mean()
            Temp_Data.fillna(Temp_Mean, inplace=True)
        elif Initial_Fill_Method == 'Median':
            Temp_Median = Temp_Data.median()
            Temp_Data.fillna(Temp_Median,inplace = True)  #得到初始填补后的原始数据，之后从该序列出发，该序列应该
        return Temp_Data
     ###          -------------- 得到训练模型，根据选择的不同，得到不同的 训练模型 ---------------          #########
    Model = RandomForestRegressor(n_estimators=100,random_state=42)  #初始化随机森林模型 ,后续使用该模型
    # n——estimators代表决策树的数量，其他暂时默认，randomstate代表随机种子，保证每次一样
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

        Model.fit(Input_X,Input_Y)   #训练对应的模型
        Prediction_Y = Model.predict(Output_X)   #根据模型结果输出预测的值
        Prediction_Result = pd.DataFrame(Prediction_Y,index=Output_Y_Row,columns=Output_Y_Columns)
        # 使用上面的结果对缺失的值进行插补，使用fillna函数进行插补
        Temp_Data.fillna(Prediction_Result,inplace=True)
        return Temp_Data   #返回补缺结果矩阵，结束
    elif Null_Columns_Number == Missing_Data.shape[1]:   ##########  -------------  此时多一步，即创建随机选取的列进行初始插补----------- ######
        # 使用rand生成原始数据一半列数的完整列，将其作为输入特征，训练回归器件
        Initial_Fill_Number = round(Null_Columns_Number/RoundDown_Number)  #一半向下取整  // 或者改成除3向下取整
        Initial_Fill_Columns_List = random.sample(range(0,Rand_Fill_Number),Initial_Fill_Number)   #初始填补的列
        for columns_number in Initial_Fill_Columns_List:
            if Initial_Fill_Method == 'Median':
                Temp_Median = Temp_Data.iloc[:,columns_number].median()
                Temp_Data.iloc[:, columns_number].fillna(Temp_Median,inplace = True)  #得到初始填补后的原始数据，之后从该序列出发，该序列应该
            elif Initial_Fill_Method == 'Mean':
                Temp_Mean = Temp_Data.iloc[:,columns_number].mean()
                Temp_Data.iloc[:,columns_number].fillna(Temp_Mean,inplace=True)
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
        Model.fit(Input_X,Input_Y)   #训练对应的模型
        Prediction_Y = Model.predict(Output_X)   #根据模型结果输出预测的值
        Prediction_Result = pd.DataFrame(Prediction_Y,index=Output_Y_Row,columns=Output_Y_Columns)
        # 使用上面的结果对缺失的值进行插补，使用fillna函数进行插补
        Temp_Data.fillna(Prediction_Result,inplace=True)
        return Temp_Data
def DT_Imputation_Of_Missing_Data(Missing_Data, RoundDown_Number = 4,Initial_Fill_Method = 'Mean'):
    # RoundDown_Number 代表的是向下取整的数字大小，默认认为是3，也可以改成2，这样可以对不同的序列大小进行判断
    random.seed(12345)  #设置随机种子
    if Missing_Data.shape[1] <= 2:
        RoundDown_Number = 1 #如果列数太少，可能导致生成不了初始填补的矩阵，会报错，因此太小的时候对其重新赋值
    Temp_Data = Missing_Data.copy()
    Rand_Fill_Number = Temp_Data.shape[1]   #目前生成的类型 没有单独一列完整的，后续可以补充上，如果没有完整一列，需要通过随机数先生成填补结果
    Null_Columns_Number = Temp_Data.isna().any().sum()  #得到 有多少数量的列数是包含 缺失值的  ,注意这里是先any再sum
    if Null_Columns_Number <= 2:    #缺失的列数太少直接返回 填补的结果即可
        if Initial_Fill_Method == 'Mean':
            Temp_Mean = Temp_Data.mean()
            Temp_Data.fillna(Temp_Mean, inplace=True)
        elif Initial_Fill_Method == 'Median':
            Temp_Median = Temp_Data.median()
            Temp_Data.fillna(Temp_Median,inplace = True)  #得到初始填补后的原始数据，之后从该序列出发，该序列应该
        return Temp_Data
     ###          -------------- 得到训练模型，根据选择的不同，得到不同的 训练模型 ---------------          #########
    Model = DecisionTreeRegressor(random_state=42)  #初始化随机森林模型 ,后续使用该模型
    # n——estimators代表决策树的数量，其他暂时默认，randomstate代表随机种子，保证每次一样
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

        Model.fit(Input_X,Input_Y)   #训练对应的模型
        Prediction_Y = Model.predict(Output_X)   #根据模型结果输出预测的值
        Prediction_Result = pd.DataFrame(Prediction_Y,index=Output_Y_Row,columns=Output_Y_Columns)
        # 使用上面的结果对缺失的值进行插补，使用fillna函数进行插补
        Temp_Data.fillna(Prediction_Result,inplace=True)
        return Temp_Data   #返回补缺结果矩阵，结束
    elif Null_Columns_Number == Missing_Data.shape[1]:   ##########  -------------  此时多一步，即创建随机选取的列进行初始插补----------- ######
        # 使用rand生成原始数据一半列数的完整列，将其作为输入特征，训练回归器件
        Initial_Fill_Number = round(Null_Columns_Number/RoundDown_Number)  #一半向下取整  // 或者改成除3向下取整
        Initial_Fill_Columns_List = random.sample(range(0,Rand_Fill_Number),Initial_Fill_Number)   #初始填补的列
        for columns_number in Initial_Fill_Columns_List:
            if Initial_Fill_Method == 'Median':
                Temp_Median = Temp_Data.iloc[:,columns_number].median()
                Temp_Data.iloc[:, columns_number].fillna(Temp_Median,inplace = True)  #得到初始填补后的原始数据，之后从该序列出发，该序列应该
            elif Initial_Fill_Method == 'Mean':
                Temp_Mean = Temp_Data.iloc[:,columns_number].mean()
                Temp_Data.iloc[:,columns_number].fillna(Temp_Mean,inplace=True)
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
        Model.fit(Input_X,Input_Y)   #训练对应的模型
        Prediction_Y = Model.predict(Output_X)   #根据模型结果输出预测的值
        Prediction_Result = pd.DataFrame(Prediction_Y,index=Output_Y_Row,columns=Output_Y_Columns)
        # 使用上面的结果对缺失的值进行插补，使用fillna函数进行插补
        Temp_Data.fillna(Prediction_Result,inplace=True)
        return Temp_Data
def SVR_Imputation_Of_Missing_Data(Missing_Data, Kernel_Function = 'rbf',RoundDown_Number = 4,Initial_Fill_Method = 'Mean'):   #支持向量机回归模型
    # Temp_Data = Missing_Data.copy()
    random.seed(12345)  #设置随机种子
    if Missing_Data.shape[1] <= 2:
        RoundDown_Number = 1 #如果列数太少，可能导致生成不了初始填补的矩阵，会报错，因此太小的时候对其重新赋值
    Temp_Data = Missing_Data.copy()
    Temp_Data_Miss = Missing_Data.copy()  #这是对其进行重复的copy，防止初始填补对TempData数据进行修改；所以进行重复的copy

    Rand_Fill_Number = Temp_Data.shape[1]   #目前生成的类型 没有单独一列完整的，后续可以补充上，如果没有完整一列，需要通过随机数先生成填补结果
    Null_Columns_Number = Temp_Data.isna().any().sum()  #得到 有多少数量的列数是包含 缺失值的  ,注意这里是先any再sum
    if Null_Columns_Number <= 1:    #缺失的列数太少直接返回 填补的结果即可
        if Initial_Fill_Method == 'Mean':
            Temp_Mean = Temp_Data.mean()
            Temp_Data.fillna(Temp_Mean, inplace=True)
        elif Initial_Fill_Method == 'Median':
            Temp_Median = Temp_Data.median()
            Temp_Data.fillna(Temp_Median,inplace = True)  #得到初始填补后的原始数据，之后从该序列出发，该序列应该是完整的序列，但是需要记录下填补的位置
        return Temp_Data
     ###          -------------- 得到训练模型，根据选择的不同，得到不同的 训练模型 ---------------          #########
    Model = SVR(kernel=Kernel_Function, C=10, gamma=0.1, epsilon=0.1)
    # C 是正则化参数，越大代表其参数拟合程度越高，太大也容易过拟合
    # Kernel代表核函数，默认的有rbf，poly，sigmoid，linear
    # gamma 代表核函数系数，越大代表模型越复杂
    # epsilon代表不敏感损失参数，默认一般为0.1
    # 遍历每一列进行SVR填补
        ##         -------------- 后面是对缺失数据的处理，回归的输入输出的处理  ----------  ##
    if Null_Columns_Number < Missing_Data.shape[1]:  #这里代表一开始初始的时候就有无缺失列，这样可以直接定下来输入列
        Initial_Fill_Number = round(Null_Columns_Number/RoundDown_Number)  #一半向下取整  // 或者改成除3向下取整
        Initial_Fill_Columns_List = random.sample(range(0,Rand_Fill_Number),Initial_Fill_Number)   #初始填补的列
        for columns_number in Initial_Fill_Columns_List:
            if Initial_Fill_Method == 'Median':
                Temp_Median = Temp_Data.iloc[:,columns_number].median()
                Temp_Data.iloc[:, columns_number].fillna(Temp_Median,inplace = True)  #得到初始填补后的原始数据，之后从该序列出发，该序列应该
            elif Initial_Fill_Method == 'Mean':
                Temp_Mean = Temp_Data.iloc[:,columns_number].mean()
                Temp_Data.iloc[:,columns_number].fillna(Temp_Mean,inplace=True)
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
        for col in Input_Y_Colmns:
            Input_X = Temp_Data.loc[Input_X_Row, Input_X_Columns]  # 得到输入的DataFrame作为X， 这里的X应该是对应 行列均无缺失，完整的列
            Input_Y = Temp_Data.loc[Input_Y_Row, col]  # 得到输入的Y
            ##       2025.3.3 修改输入张量格式为一维，修改尝试    ##
            Input_X = Input_X.to_numpy()
            Input_Y = Input_Y.to_numpy()
            Model.fit(Input_X, Input_Y)  # 训练对应的模型
            Output_X_Row = Residual_Fill_Row_List
            Output_X_Columns = Initial_Fill_Columns_List
            Output_Y_Row = Residual_Fill_Row_List
            Output_Y_Columns = Residual_Fill_Columns_List
            Output_X = Temp_Data.loc[Output_X_Row, Output_X_Columns]
            Output_X = Output_X.to_numpy()  # 这里输入也要转化
            Prediction_Y = Model.predict(Output_X)  # 根据模型结果输出预测的值
            Temp_Data.loc[Output_X_Row, col] = Prediction_Y
        return Temp_Data
    elif Null_Columns_Number == Missing_Data.shape[1]:   ##########  -------------  此时多一步，即创建随机选取的列进行初始插补----------- ######
        # 使用rand生成原始数据一半列数的完整列，将其作为输入特征，训练回归器件
        Initial_Fill_Number = round(Null_Columns_Number/RoundDown_Number)  #一半向下取整  // 或者改成除3向下取整
        Initial_Fill_Columns_List = random.sample(range(0,Rand_Fill_Number),Initial_Fill_Number)   #初始填补的列
        for columns_number in Initial_Fill_Columns_List:
            if Initial_Fill_Method == 'Median':
                Temp_Median = Temp_Data.iloc[:,columns_number].median()
                Temp_Data.iloc[:, columns_number].fillna(Temp_Median,inplace = True)  #得到初始填补后的原始数据，之后从该序列出发，该序列应该
            elif Initial_Fill_Method == 'Mean':
                Temp_Mean = Temp_Data.iloc[:,columns_number].mean()
                Temp_Data.iloc[:,columns_number].fillna(Temp_Mean,inplace=True)
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
        for col in Input_Y_Colmns:
            Input_X = Temp_Data.loc[Input_X_Row, Input_X_Columns]  # 得到输入的DataFrame作为X， 这里的X应该是对应 行列均无缺失，完整的列
            Input_Y = Temp_Data.loc[Input_Y_Row, col]  # 得到输入的Y
        ##       2025.3.3 修改输入张量格式为一维，修改尝试    ##
            Input_X = Input_X.to_numpy()
            Input_Y = Input_Y.to_numpy()
            Model.fit(Input_X, Input_Y)  # 训练对应的模型
            Output_X_Row = Residual_Fill_Row_List
            Output_X_Columns = Initial_Fill_Columns_List
            Output_Y_Row = Residual_Fill_Row_List
            Output_Y_Columns = Residual_Fill_Columns_List
            Output_X = Temp_Data.loc[Output_X_Row,Output_X_Columns]
            Output_X = Output_X.to_numpy()  #这里输入也要转化
        # Output_Y = Temp_Data.loc[Output_Y_Row,Output_Y_Columns]
        #
            Prediction_Y = Model.predict(Output_X)   #根据模型结果输出预测的值
            Temp_Data.loc[Output_X_Row,col] = Prediction_Y
        return Temp_Data
def SoftImpute_Imputation_Of_Missing_Data(Missing_Data,Max_iter = 100,Tol = 1e-4):  #缺省值代表最大循环次数为100次，阈值为1e-4
    Temp_Data = Missing_Data.copy()
    Columns_Name = Temp_Data.columns.tolist()
    Temp_Array = pd.DataFrame(Temp_Data)
    Temp_Imputation_Array = SoftImpute().fit_transform(Temp_Array)
    Temp_Out = pd.DataFrame(Temp_Imputation_Array)
    Temp_Out.columns = Columns_Name
    return Temp_Out
def Low_Rank_Imputation_Of_Missing_Data(df, rank=2, max_iter=1000, tol=1e-6):
    """
    使用低秩矩阵分解方法填补缺失数据。
    参数:
    df (pandas.DataFrame): 包含缺失数据的 DataFrame，缺失值用 NaN 表示。
    rank (int): 目标低秩矩阵的秩。
    max_iter (int): 最大迭代次数。
    tol (float): 收敛阈值。
    返回:
    pandas.DataFrame: 填补后的 DataFrame。
    """
    # 将 DataFrame 转换为 numpy 数组
    matrix = df.to_numpy()
    # 找到缺失值的位置
    mask = ~np.isnan(matrix)
    # 初始化缺失值为 0
    matrix[~mask] = 0
    for i in range(max_iter):
        # 对矩阵进行奇异值分解
        U, s, Vt = svd(matrix, full_matrices=False)
        # 保留前 rank 个奇异值
        s[rank:] = 0
        # 重构矩阵
        matrix_reconstructed = U @ np.diag(s) @ Vt
        # 仅在缺失值处更新矩阵
        matrix[~mask] = matrix_reconstructed[~mask]
        # 检查收敛性
        if np.linalg.norm(matrix_reconstructed - matrix) < tol:
            break
    # 将填补后的矩阵转换回 DataFrame
    filled_df = pd.DataFrame(matrix, index=df.index, columns=df.columns)
    return filled_df
def PCA_Imputation_Of_Missing_Data(df, n_components=None, max_iter=100, tol=1e-3):
    """
    使用 PCA 方法填补缺失数据。

    参数:
    df (pandas.DataFrame): 包含缺失数据的 DataFrame，缺失值用 NaN 表示。
    n_components (int or None): PCA 的主成分数量。如果为 None，则自动选择所有成分。
    max_iter (int): 最大迭代次数。
    tol (float): 收敛阈值。
    返回:
    pandas.DataFrame: 填补后的 DataFrame。
    """
    # 将 DataFrame 转换为 numpy 数组
    matrix = df.to_numpy()

    # 使用 IterativeImputer 进行初步填补
    imputer = IterativeImputer(max_iter=max_iter, tol=tol, random_state=0)
    matrix_imputed = imputer.fit_transform(matrix)

    # 使用 PCA 进行降维和重构
    pca = PCA(n_components=n_components)
    matrix_reduced = pca.fit_transform(matrix_imputed)  # 降维
    matrix_reconstructed = pca.inverse_transform(matrix_reduced)  # 重构

    # 将填补后的矩阵转换回 DataFrame
    filled_df = pd.DataFrame(matrix_reconstructed, index=df.index, columns=df.columns)

    return filled_df









