import pandas as pd
import numpy as np
import os
import shutil
import csv
from pathlib import Path
from io import StringIO
from Statistical_Imputation_Methods import Mean_Imputation_Of_Missing_Data
from Statistical_Imputation_Methods import Median_Imputation_Of_Missing_Data
from Statistical_Imputation_Methods import Mode_Imputation_Of_Missing_Data

def Split_DataFileName(FileFold):
    '''
    :param FileFold:  数据路径，读取该路径下所有的文件，需要将所有文件转化为txt文件格式
    :return: 将数据文件按照文件名字分类完成
    '''
    # 创建总保存目录
    all_saved = os.path.join(FileFold, "AllSaved")   # 创建总的保存目录 ，这里是在当前目录下新建一个文件夹
    os.makedirs(all_saved, exist_ok=True)
    # 遍历源文件夹中的每个文件
    for filename in os.listdir(FileFold):
        src_path = os.path.join(FileFold, filename)
        if not os.path.isfile(src_path):
            continue
    # 处理文件名和格式转换
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}.txt"
    dst_path = os.path.join(all_saved, new_filename)

    try:
        # 根据文件类型执行不同转换策略
        if ext.lower() == '.csv':
            # CSV转制表符分隔文本
            with open(src_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                with open(dst_path, 'w') as txt_file:
                    for row in csv_reader:
                        txt_file.write('\t'.join(row) + '\n')
        elif ext.lower() in ['.xlsx', '.xls']:
            # Excel转文本
            df = pd.read_excel(src_path)
            df.to_csv(dst_path, sep='\t', index=False)
        else:
            # 其他文件直接转存
            with open(src_path, 'r', encoding='utf-8') as f_in:
                content = f_in.read()
            with open(dst_path, 'w', encoding='utf-8') as f_out:
                f_out.write(content)
    except UnicodeDecodeError:
        print(f"跳过二进制文件: {filename}")
    except Exception as e:
        print(f"文件转换错误: {filename} - {str(e)}")

        # 文件名分段处理
    segments = [s for s in name.split('_') if s]  # 过滤空字符串
    # 创建分类目录并复制文件
    for segment in segments:
        segment_dir = os.path.join(all_saved, segment)
        os.makedirs(segment_dir, exist_ok=True)
        shutil.copy(dst_path, os.path.join(segment_dir, new_filename))

def Original_Data_Read(FileFold,FileName):
    '''
    该函数用于读取原始的多轴IMU数据，从相应的文件夹下读取数据，返回Data代表的是读取到的数据
    :param FileFold:  指定的目录，可以是绝对目录，也可以是相对目录
    :param FileName:  指定的文件名，这是按照自己的命名规则定义的
    :return: Data 代表对应文件目录下的对应文件名内数据，将其转化为DataFrame格式
    '''
    with open(FileFold+"/"+FileName,'r') as file:
        content = file.read()   #这里输出的是字符串类型str
    ## 默认的content是按照，分离的，因此需要按照，来分割对应的内容结果
    df = pd.read_csv(StringIO(content),header=None)  #利用readcsv来读取对应的文件内容
    Data = df
    # Data = content
    # Data = pd.DataFrame(content)
    return Data

def PreProcess_Data(df,Pre_Method="Mean"):
    '''
    该函数主要用于将对应的Data数据进行预处理，作用是分离缺失数据值，找到缺失值的位置
    :param df: df代表含有缺失值的DataFrame格式数据;需要注意的是，这里的df里面需要包含缺失值
    :return: 返回一个预填补后的矩阵；Pre_Data;返回一个标记矩阵Mask_Matrix 记录了原始缺失值的位置；missingrows返回对应的行丢失索引；missingcols返回对应列丢失索引；
    '''
    # 找到有缺失值的列  ,进行定位
    missing_cols = df.columns[df.isnull().any()].tolist()
    missing_rows = df[df.isnull().any(axis=1)].index
    non_missing_cols = df.columns[~df.isnull().any()].tolist()
    # 找到缺失值的行列索引
    missing_values = df.isnull().stack()
    missing_indices = missing_values[missing_values].index.tolist() #这个是有缺失值的索引
    # 缺失数据标记矩阵
    Mask_Matrix = df.isna().astype(int) #生成全零的矩阵，找到缺失值的地方，按照索引将其赋值为1
    # 预填补
    Pre_Data = pd.DataFrame()  #需要先定义
    if Pre_Method == "Mean":
        Pre_Data = Mean_Imputation_Of_Missing_Data(df,0)
    elif Pre_Method =="Median":
        Pre_Data = Median_Imputation_Of_Missing_Data(df)
    elif Pre_Method == "Mode":
        Pre_Data = Mode_Imputation_Of_Missing_Data(df)
    return Pre_Data,missing_indices,missing_cols,missing_rows,Mask_Matrix


