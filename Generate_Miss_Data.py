import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import missingno as msn
def Gengerate_Incomplete_Data_From_Time_Series(Data, Rate_Of_Incomplete=0.2, Missing_Mode = "Random", Missing_Row_Number = 5, Missing_Mechanism = "MCAR") -> object:
    # 默认Data是DataFrame类型的数据  ，且为M*1  ,该函数的作用是输入M*1的Data，将其中 X%的数据变为None即为空值
    # Data_columns_Number是输入数据的列数，默认是1，就是时间序列
    # 增加生成NaN数据的函数性能，增加三种缺失性能不同的功能，默认是random就是都随机，另外sparse代表稀疏，NaN两者之间不会产生相连   #删除Sparse模式 2024.12.21
    # 2025.1.8 增加一个新的缺失数据模式，叫做 univariant 单变量模式，该模式更加简单  ,默认是单列缺失， 需要注意，如果比例过高可能造成 单列不够缺失，这时候需要判断，大于单列全部数量一般的时候，增加一列
    # Missing_Row 代表缺失所有数据都是行相同的，列可以不同， Missing_Col代表所有数据都是列相同的，行可以不同, 这个数字代表连续缺失的数量（包括行和列）
    # Missing_Mechanism 代表缺失数据模式，这里默认是全部随机MCAR，可供选择的参数有 MCAR， MAR 以及 MNAR ，需要注意的是选择MNAR默认是 选择去掉较小的值，可以更改为MNAR_H来变成去掉更高值
    Temp_Data = Data.copy()    #创建原始数据的备份
    if Rate_Of_Incomplete > 1:
        Rate_Of_Incomplete = Rate_Of_Incomplete / 100  # 可能会输入成 30，进行修改，默认是0.3
        if Rate_Of_Incomplete > 1:   #仍然错误代表 设置错误
            return "输入格式有误，比例设置错误"
    if type(Temp_Data) == pd.core.frame.DataFrame:    #默认使用DataFrame格式
        ErrorFlag = 0
        Data_columns_Number = Temp_Data.shape[1]
    else:
        ErrorFlag = 1
        return "输入格式有误，数据格式有误"
    ###########--------------------------------------  MCAR  ----------------------------------------############
    if Missing_Mechanism == "MCAR":
        if Missing_Mode == "Univariant":   # 单变量的缺失生成模式，生成数量有问题
            Column_Number = 1 #默认初始为1
            Missing_Total_Number = round(Temp_Data.shape[0]*Temp_Data.shape[1] * Rate_Of_Incomplete) #计算得到总的丢失数量
            Univariant_Max_Number = round(Temp_Data.shape[0]*0.6*Column_Number)  #最大允许的数量应该是列数的1/2 再乘以列数 ,这里的系数可以修改，变大就是可以接受更大的缺失在一列里面
            while Univariant_Max_Number < Missing_Total_Number:
                Column_Number = Column_Number + 1
                Univariant_Max_Number = round(Temp_Data.shape[0]*0.6*Column_Number)
            # 此时得到了ColumnsNumber，该数量代表最后缺失的列数
            if Column_Number > Temp_Data.shape[1]-1:
                return "出现过大比例"   ##注意这里，改成字符串会进入这里，需要重新检查
            # 接下来生成 列的索引， 数量为上面生成的Columns数量，范围是从0到列数
            Missing_Column_Index = random.sample(list(range(Temp_Data.shape[1])),Column_Number)   #随机取样得到索引矩阵
            Missing_Column = Temp_Data.iloc[:,Missing_Column_Index]  #得到对应的列矩阵
            Temp_Miss_Column = Missing_Column.copy()
            # 此时数据为类别数据，类似 鸢尾花（iris）数据集  ，此时需要生成 M*N数据列的随机数据
            Row_Number = Temp_Miss_Column.shape[0]    #调用下面的随机生成结果
            Col_Number = Temp_Miss_Column.shape[1]
            Result = []
            while len(Result) < round(Temp_Data.shape[0] * Temp_Data.shape[1] * Rate_Of_Incomplete):
                Row_rand_Number = random.randint(0, Row_Number - 1)
                Col_rand_Number = random.randint(0, Col_Number - 1)
                if ([Row_rand_Number, Col_rand_Number] not in Result):
                    Result.append([Row_rand_Number, Col_rand_Number])
                    # 循环结束后，Result内应该是需要变为NaN的原始数据的索引，包含行索引和列索引，第一列是行索引，第二列是列索引
            for i in range(0, len(Result)):
                Temp_Miss_Column.iloc[Result[i][0], Result[i][1]] = None
            #  ###此时需要将原来的矩阵替换到原来的DataFrame里面
            Temp_Data.iloc[:,Missing_Column_Index] = Temp_Miss_Column   #替换完成
            return Temp_Data
        if Missing_Mode == "Random":
            if Data_columns_Number == 1:
                if ErrorFlag == 0:
                # 此时数据类型正确，进行修改操作，生成补缺数据操作
                    Row_Number = Temp_Data.shape[0]  # 得到DataFrame的行数
                    Result = random.sample(list(range(Row_Number)),
                                        math.floor(Rate_Of_Incomplete * Row_Number))  # 随机生成了需要变为NAN值的序号Index
                    Temp_Data.loc[Result, :] = None
                return Temp_Data
            else:
                Temp_Data = Data.copy()
                #此时数据为类别数据，类似 鸢尾花（iris）数据集  ，此时需要生成 M*N数据列的随机数据
                Row_Number = Temp_Data.shape[0]
                Col_Number = Temp_Data.shape[1]
                Result = []
                while len(Result) < round(Row_Number*Col_Number*Rate_Of_Incomplete):
                    Row_rand_Number = random.randint(0,Row_Number-1)
                    Col_rand_Number = random.randint(0,Col_Number-1)
                    if ([Row_rand_Number,Col_rand_Number] not in Result):
                        Result.append([Row_rand_Number,Col_rand_Number])
                        #循环结束后，Result内应该是需要变为NaN的原始数据的索引，包含行索引和列索引，第一列是行索引，第二列是列索引
                for i in range(0,len(Result)):
                    Temp_Data.iloc[Result[i][0],Result[i][1]] = None
                return Temp_Data
        elif Missing_Mode == "InRow":
            #按照行丢失来进行生成  ,最后一个参数在此条件下生效
            if Data_columns_Number == 1:
                if ErrorFlag == 0:
                    Row_Number = Temp_Data.shape[0]
                    Result = []
                    while len(Result) < round(Row_Number*Rate_Of_Incomplete):
                        Random_Number = random.randint(0,Row_Number-1)
                        if (Random_Number not in Result and (Random_Number not in [x-1 for x in Result]) and (Random_Number not in [x+1 for x in Result])):
                            Result.append(Random_Number)  #加入该数字
                            #加入该数字后，向后添加 N个数字
                            if Random_Number + Missing_Row_Number <= Row_Number-1:
                                for i in range(1,Missing_Row_Number):
                                    Result.append(Random_Number+i)
                            elif Random_Number + Missing_Row_Number > Row_Number -1:
                                Sub_Temp = Row_Number-1 - (Random_Number+Missing_Row_Number)
                                for i in range(1,Sub_Temp):
                                    Result.append(Random_Number+i)
                    Temp_Data.loc[Result, :] = None
                    return Temp_Data
            else:
                #此时代表多行数据
                if ErrorFlag == 0:
                    Row_Number = Temp_Data.shape[0]
                    Col_Number = Temp_Data.shape[1]
                    Result = []
                    while len(Result) * Col_Number < round(Row_Number * Col_Number * Rate_Of_Incomplete):  #这里的一个Result就代表了一行，因此需要除列数
                        Row_rand_Number = random.randint(0, Row_Number - 1)  #生成需要缺失的行 索引
                        # Col_rand_Number = random.randint(0, Col_Number - 1)
                        if ([Row_rand_Number] not in Result):
                            Result.append(Row_rand_Number)
                            # if Row_rand_Number + Missing_Row_Number <= Row_Number-1:  #这里代表如果当前行加上连续缺失行都不小于总的行数，那就加上， 可以去掉这一行，不然不对
                            #     for i in range(1,Missing_Row_Number):
                            #         Result.append(Row_rand_Number+i)
                            # elif Row_rand_Number + Missing_Row_Number > Row_Number -1:
                            #     Sub_Temp = Row_Number-1 - (Row_rand_Number+Missing_Row_Number)
                            #     for i in range(1,Sub_Temp):
                            #         Result.append(Row_rand_Number+i)
                    # Result = random.sample(list(range(Row_Number)),
                    #                        math.floor(Rate_Of_Incomplete * Row_Number))  # 随机生成了需要变为NAN值的序号Index  ,sparse情况下需要生成不重复且间隔至少为2的
                    for i in range(0, len(Result)):
                        Temp_Data.iloc[Result[i],:] = None
                    return Temp_Data
        elif Missing_Mode == "InCol":
            if Data_columns_Number == 1:
                if ErrorFlag == 0:
                    Row_Number = Temp_Data.shape[0]
                    Result = []
                    while len(Result) < round(Row_Number*Rate_Of_Incomplete):
                        Random_Number = random.randint(0,Row_Number-1)
                        if (Random_Number not in Result and Random_Number not in [x-1 for x in Result] ):
                            Result.append(Random_Number)  #加入该数字
                            #加入该数字后，向后添加 N个数字
                            if Random_Number + Missing_Row_Number <= Row_Number-1:
                                for i in range(1,Missing_Row_Number):
                                    Result.append(Random_Number+i)
                            elif Random_Number + Missing_Row_Number > Row_Number -1:
                                Sub_Temp = Row_Number-1 - (Random_Number+Missing_Row_Number)
                                for i in range(1,Sub_Temp):
                                    Result.append(Random_Number+i)
                    Temp_Data.loc[Result, :] = None
                    return Temp_Data
            else:
                #此时代表多行数据
                if ErrorFlag == 0:
                    Row_Number = Temp_Data.shape[0]
                    Col_Number = Temp_Data.shape[1]
                    Result = []
                    while len(Result) < round(Row_Number * Col_Number * Rate_Of_Incomplete):
                        Row_rand_Number = random.randint(0, Row_Number - 1)
                        Col_rand_Number = random.randint(0, Col_Number - 1)
                        if ([Row_rand_Number,Col_rand_Number] not in Result):
                            Result.append([Row_rand_Number,Col_rand_Number])
                            if Row_rand_Number + Missing_Row_Number <= Row_Number-1:
                                for i in range(1,Missing_Row_Number):
                                    if [Row_rand_Number+i,Col_rand_Number] not in Result:
                                        Result.append([Row_rand_Number+i,Col_rand_Number])
                            elif Row_rand_Number + Missing_Row_Number > Row_Number-1:
                                Sub_Temp = Row_Number-1 - (Row_rand_Number+Missing_Row_Number)
                                for i in range(1,Sub_Temp):
                                    if [Row_rand_Number + i, Col_rand_Number] not in Result:
                                        Result.append([Row_rand_Number+i,Col_rand_Number])   #按照列生成随机NAN    #这样会造成重复信息,所以需要再加一句判断
                    # Result = random.sample(list(range(Row_Number)),
                    #                        math.floor(Rate_Of_Incomplete * Row_Number))  # 随机生成了需要变为NAN值的序号Index  ,sparse情况下需要生成不重复且间隔至少为2的
                    for i in range(0, len(Result)):
                        Temp_Data.iloc[Result[i][0], Result[i][1]] = None
                    return Temp_Data
   ###########--------------------------------------  MAR  ----------------------------------------############
    elif Missing_Mechanism == "MAR":   #部分随机
        # MAR和MCAR的区别怎么在程序中体现

        return Temp_Data
    ###########--------------------------------------MNAR  ----------------------------------------############
    elif Missing_Mechanism == "MNAR":   #完全非随机，这个的生成程序需要完全重写，可以分为按照较低的值去除，或者按照较高的值去除 ，还有可能缺失生成的值不一样多，继续检查
        if Missing_Mode == "Univariant":   # 单变量的缺失生成模式
            Column_Number = 1 #默认初始为1
            Missing_Total_Number = round(Temp_Data.shape[0]*Temp_Data.shape[1] * Rate_Of_Incomplete) #计算得到总的丢失数量
            Univariant_Max_Number = round(Temp_Data.shape[0]*0.6*Column_Number)  #最大允许的数量应该是列数的1/2 再乘以列数 ,这里的系数可以修改，变大就是可以接受更大的缺失在一列里面
            while Univariant_Max_Number < Missing_Total_Number:
                Column_Number = Column_Number + 1
                Univariant_Max_Number = round(Temp_Data.shape[0]*0.6*Column_Number)
            # 此时得到了ColumnsNumber，该数量代表最后缺失的列数
            if Column_Number > Temp_Data.shape[1]*0.6:
                return "错误，此时列数过大，不合理，请检查程序"
            Missing_Column_Index = random.sample(list(range(Temp_Data.shape[1]-1)),Column_Number)   #随机取样得到索引矩阵
            Missing_Column = Temp_Data.iloc[:,Missing_Column_Index]  #得到对应的列矩阵
            Temp_Miss_Column = Missing_Column.copy()
            # 此时数据为类别数据，类似 鸢尾花（iris）数据集  ，此时需要生成 M*N数据列的随机数据 , 这里是MNAR，这里的索引生成方式需要找到最小的数字来进行生成
            Row_Number = Temp_Miss_Column.shape[0]    #调用下面的随机生成结果
            Col_Number = Temp_Miss_Column.shape[1]
            Result = []
            if Col_Number == 1:
                Row_Count_Number = 0
                Total_Missing_Number = round(Temp_Data.shape[0] * Temp_Data.shape[1] * Rate_Of_Incomplete)  # 取整
                while len(Result) < Total_Missing_Number:
                    Row_rand_Number_Index = Temp_Miss_Column.nlargest(round(Total_Missing_Number/Col_Number)+1,list(Temp_Miss_Column.columns)).index
                    # 这里使用nlargest排序，需要注意，使用dataframe情况下需要 第二个参数  columns名，并且是list类型的
                    Row_rand_Number = Row_rand_Number_Index[Row_Count_Number]  #得到排序后的索引值
                    Result.append([Row_rand_Number, 0])
                    Row_Count_Number = Row_Count_Number + 1 # 每次循环结束这个值加一，代表index加一
            else:   #这里是列数不为1的情况
                Row_Count_Number = 0
                Total_Missing_Number = round(Temp_Data.shape[0]*Temp_Data.shape[1]*Rate_Of_Incomplete)  #取整
                for col_temp in range(Col_Number):
                    Temp_Miss_Column_Temp = Temp_Miss_Column.iloc[:,col_temp]
                    Row_rand_Number = Temp_Miss_Column_Temp.nlargest(round(Total_Missing_Number / Col_Number) + 1,
                                                                           keep='first').index
                    # Row_rand_Number = Row_rand_Number_Index[Row_Count_Number]
                    Result.append([Row_rand_Number, col_temp])
                    # Row_Count_Number = Row_Count_Number + 1  # 每次循环结束这个值加一，代表index加一
                    # 循环结束后，Result内应该是需要变为NaN的原始数据的索引，包含行索引和列索引，第一列是行索引，第二列是列索引
            #@#          @@@@@@@@@@@@  接下来是根据Result里面的行列索引值来将对应的值变为None    @@@@@@@@@@@   ###
            for i in range(0, len(Result)):
                    Temp_Miss_Column.iloc[Result[i][0], Result[i][1]] = None
            #  ###此时需要将原来的矩阵替换到原来的DataFrame里面
            Temp_Data.iloc[:,Missing_Column_Index] = Temp_Miss_Column   #替换完成
            return Temp_Data
        elif Missing_Mode == "Random":
            Temp_Data = Data.copy()
            # 此时数据为类别数据，类似 鸢尾花（iris）数据集  ，此时需要生成 M*N数据列的随机数据
            Row_Number = Temp_Data.shape[0]
            Col_Number = Temp_Data.shape[1]
            Result = []
            Row_Count_Number = 0
            Total_Missing_Number = round(Temp_Data.shape[0] * Temp_Data.shape[1] * Rate_Of_Incomplete)  # 取整  丢失数据总体数量
            for col_temp in Temp_Data.columns:  #遍历列,这里使用列名的‘’str来找到对应的列
                Temp_Miss_Column_Temp = Temp_Data[col_temp]
                Row_rand_Number = Temp_Miss_Column_Temp.nlargest(round(Total_Missing_Number / Col_Number),
                                                                 keep='all').index
                Col_pos = Temp_Data.columns.get_loc(col_temp)   #根据列名来找到对应的列号
                Result.append([Row_rand_Number, Col_pos])
                if len(Result) > Total_Missing_Number:
                    break
                # Row_Count_Number = Row_Count_Number + 1  # 每次循环结束这个值加一，代表index加一  这里没有while循环，不需要这个了
                    # 循环结束后，Result内应该是需要变为NaN的原始数据的索引，包含行索引和列索引，第一列是行索引，第二列是列索引
            for i in range(0, len(Result)):
                Temp_Data.iloc[Result[i][0], Result[i][1]] = None
            return Temp_Data

        elif Missing_Mode == "InRow":   #########  InRow  #########  行模式，MNAR的行模式  ，行模式怎么找MNAR，找这一列的平均值吧
            Row_Number = Temp_Data.shape[0]
            Col_Number = Temp_Data.shape[1]
            Result = []
            Total_Missing_Number = Row_Number*Col_Number*Rate_Of_Incomplete  #总共缺失的数量
            Mean_Temp = Temp_Data.mean(axis = 1)  #得到每一行的平均值
            while len(Result) * Col_Number < round(Row_Number * Col_Number * Rate_Of_Incomplete):  # 这里的一个Result就代表了一行，因此需要除列数
                Row_rand_Number = Mean_Temp.nlargest(round(Total_Missing_Number / Col_Number),keep='all').index   #生成索引值，按照最小值去除
                Result.append(Row_rand_Number)    #这时候由于是按照平均值排序的，所以不需要判断是否在原始Result里，直接添加进去就可以
            for i in range(0, len(Result)):
                Temp_Data.iloc[Result[i], :] = None
            return Temp_Data
        elif Missing_Mode == "InCol":
            Row_Number = Temp_Data.shape[0]
            Col_Number = Temp_Data.shape[1]
            Result = []
            Total_Missing_Number = Row_Number * Col_Number * Rate_Of_Incomplete  # 总共缺失的数量
            Count_Number = 0   #计数值，用于判断得到前n小的值
            while len(Result) < round(Row_Number * Col_Number * Rate_Of_Incomplete):
                # Row_rand_Number = random.randint(0, Row_Number - 1)
                # Col_rand_Number = random.randint(0, Col_Number - 1)
                Row_Mean_Temp = Temp_Data.mean(axis = 1)  #得到每一行的平均值
                Col_Mean_Temp = Temp_Data.mean(axis = 0)  #得到每一列的平均值
                Row_rand_Number_Index = Row_Mean_Temp.sort_values().index  # 生成索引值，按照最小值去除
                Col_rand_Number = random.randint(0, Col_Number - 1)     #这里的列直接用随机了，便于生成
                Row_rand_Number = Row_rand_Number_Index[Count_Number]
                # Col_rand_Number = Col_rand_Number_Index[Count_Number]    #这样得到的数字，每次是一个整数
                if ([Row_rand_Number, Col_rand_Number] not in Result):
                    Result.append([Row_rand_Number, Col_rand_Number])
                    if Row_rand_Number + Missing_Row_Number <= Row_Number - 1:
                        for i in range(1, Missing_Row_Number):
                            if [Row_rand_Number + i, Col_rand_Number] not in Result:
                                Result.append([Row_rand_Number + i, Col_rand_Number])
                    elif Row_rand_Number + Missing_Row_Number > Row_Number - 1:
                        Sub_Temp = Row_Number - 1 - (Row_rand_Number + Missing_Row_Number)
                        for i in range(1, Sub_Temp):
                            if [Row_rand_Number + i, Col_rand_Number] not in Result:
                                Result.append(
                                    [Row_rand_Number + i, Col_rand_Number])  # 按照列生成随机NAN    #这样会造成重复信息,所以需要再加一句判断
            # Result = random.sample(list(range(Row_Number)),
            #                        math.floor(Rate_Of_Incomplete * Row_Number))  # 随机生成了需要变为NAN值的序号Index  ,sparse情况下需要生成不重复且间隔至少为2的
                Count_Number = Count_Number + 1  # 每次循环结束之后Count_Number 加一
            for i in range(0, len(Result)):
                Temp_Data.iloc[Result[i][0], Result[i][1]] = None
            return Temp_Data

def Test_Generate_Missing_Data(Data,Rate_Of_Incomplete,Continuous_Number=5,Show_Form = "matrix"):    #该函数用于测试上面的生成缺失数据函数
    Temp_Data = Data.copy() #获得拷贝数据
    Miss_Random = Gengerate_Incomplete_Data_From_Time_Series(Temp_Data,Rate_Of_Incomplete,"Random",Continuous_Number)
    Miss_InRow = Gengerate_Incomplete_Data_From_Time_Series(Temp_Data,Rate_Of_Incomplete,"InRow",Continuous_Number)
    Miss_InCol = Gengerate_Incomplete_Data_From_Time_Series(Temp_Data,Rate_Of_Incomplete,"InCol",Continuous_Number)
    Miss_Univariant = Gengerate_Incomplete_Data_From_Time_Series(Temp_Data,Rate_Of_Incomplete,"Univariant",Continuous_Number)
    #  这里可以修改需要的 缺失生成原理
    fig, axes = plt.subplots(1,4,figsize = (16,6))
    if Show_Form == "matrix":
        msn.matrix(Miss_Random, ax=axes[0], sparkline=False)
        msn.matrix(Miss_InRow, ax=axes[1], sparkline=False)
        msn.matrix(Miss_InCol, ax=axes[2], sparkline=False)
        msn.matrix(Miss_Univariant, ax=axes[3], sparkline=False)
        axes[0].set_title("Random",fontsize=16)
        axes[1].set_title("In—Row",fontsize=16)
        axes[2].set_title("In-Col",fontsize=16)
        axes[3].set_title("Univariant",fontsize=16)  #自动调整大小
        plt.tight_layout()
        plt.show()
        # 添加一行寻找各个的缺失值的大小
        Random_Miss_Number = Miss_Random.isnull().sum().sum()
        InRow_Miss_Number = Miss_InRow.isnull().sum().sum()
        InCol_Miss_Number = Miss_InCol.isnull().sum().sum()
        Univariant_Miss_Number = Miss_Univariant.isnull().sum().sum()  # 统计整个dataframe里面缺失值的数量

        print("Random缺失:{:.2f}个".format(Random_Miss_Number))
        print("InRow缺失:{:.2f}个".format(InRow_Miss_Number))
        print("InCol缺失:{:.2f}个".format(InCol_Miss_Number))
        print("Univariant缺失:{:.2f}个".format(Univariant_Miss_Number))
    elif Show_Form == "bar":
        msn.bar(Miss_Random, ax=axes[0])
        msn.bar(Miss_InRow, ax=axes[1])
        msn.bar(Miss_InCol, ax=axes[2])
        msn.bar(Miss_Univariant, ax=axes[3])
        plt.show()
    elif Show_Form == "heatmap":
        msn.heatmap(Miss_Random, ax=axes[0])
        msn.heatmap(Miss_InRow, ax=axes[1])
        msn.heatmap(Miss_InCol, ax=axes[2])
        msn.heatmap(Miss_Univariant, ax=axes[3])
        plt.show()
    elif Show_Form == "dendrogram":
        msn.dendrogram(Miss_Random, ax=axes[0])
        msn.dendrogram(Miss_InRow, ax=axes[1])
        msn.dendrogram(Miss_InCol, ax=axes[2])
        msn.dendrogram(Miss_Univariant, ax=axes[3])
        plt.show()
