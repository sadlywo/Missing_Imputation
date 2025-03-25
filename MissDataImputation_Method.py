import warnings
import time
import pandas as pd
from tqdm import tqdm   #进度条库
# import sklearn.neighbors._base
warnings.filterwarnings("ignore")  #关闭warning
#   -----------------------这里是引入的其他文件------------------------#
from Generate_Miss_Data import Test_Generate_Missing_Data
from Eval_Imputation import Imputation_Result
from Save_Eval_Result import Save_Dataframe_To_File   #引入保存到本地的数据

###################################### 使用多个文件来规划整个项目，防止各类函数太混乱，不同函数在不同命名的源文件中  #################################

if __name__ == "__main__":
    Test_Data = pd.read_csv('AllData.txt', sep=' ')  # 记得指定分割符号，不然无法正常读取
    print(Test_Data)  #显示读取的数据
    # Test_Generate_Missing_Data(Test_Data,0.1,3,"matrix")   #测试生成的数据
    #############  ---------------------  这里是仿真所有参数设置的地方，包括各类方法，缺失模式等  --------------------  ###################
    Method_Name = ['SVR']
    # Method_Name = ['MF','DT','Regression']
    # Method_Name = ['PCA']
    Pattern_Name = ['Random', 'InRow', 'InCol','Univariant']   #目前三种缺失模式 ，考虑是是否需要更新一下列缺失模式， 增加一下单列全部缺失？
    # Pattern_Name = ['InCol']
    ##                 ------------------     需要更新 新的列缺失模式    --------------                    ##
    Miss_Rate = [0.05, 0.1, 0.2, 0.3, 0.4]  #表示缺失的比例，最后的结果按照这个顺序排布下来，小的在上
    Dict_Method_Keys = range(0,len(Method_Name))
    Dict_Pattern_Keys = range(0,len(Pattern_Name)*len(Miss_Rate))  #得到模式的键值，和数值进行配对
    Dict_Method = {k:v for k,v in zip(Dict_Method_Keys,Method_Name)}   #创建键值对
    Dict_Pattern = {k:v for k,v in zip(Dict_Pattern_Keys,Pattern_Name * len(Miss_Rate))}
    Continuous_Number = 4   #设置连续缺失数据大小       3############################# 后续对这个变量的变化展开研究    ############################

    Start_Time = time.time()  #记录开始时间
    print("==========   Start  =============")

    All_RMSE_Frame, All_MAE_Frame, All_BIAS_Frame, All_MAPE_Frame, All_AE_Frame = Imputation_Result(Test_Data,Miss_Rate,Method_Name,Pattern_Name,Dict_Method,Dict_Pattern)
    print("===========  RMSE  ==============")
    print(All_RMSE_Frame)
    print("===========  MAE  ==============")
    print(All_MAE_Frame)
    print("===========  BIAS  ==============")
    print(All_BIAS_Frame)
    print("===========  MAPE  ==============")
    print(All_MAPE_Frame)
    print("===========  AE  ==============")
    print(All_AE_Frame)
    End_Time = time.time()  # 记录开始时间
    Spend_Time = End_Time-Start_Time
    print("=========== 消耗时间 =============")
    print("计算耗时:{:.2f}s".format(Spend_Time))  #打印耗时多少

    #############   ++++++++++     将对应的结果保存到本地    ++++++++===   #########################  2025.1.9还没写








