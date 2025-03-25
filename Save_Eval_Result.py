import os.path
import os
import pandas as pd
from pathlib import Path
from typing import Union
import numpy as np

def Save_Dataframe_To_File(
        df: pd.DataFrame,
        file_path: Union[str, Path],
        file_type: str = "auto",
        **kwargs
) -> None:
    """
    安全保存DataFrame到指定路径，支持多种文件格式
    参数:
        df: 要保存的DataFrame
        file_path: 保存路径（含文件名）
        file_type: 文件类型，支持 auto/csv/excel/json/parquet/feather/html
                   auto模式根据文件扩展名自动判断
        **kwargs: 各格式对应的保存参数（如index, sheet_name等）
    异常:
        ValueError: 文件类型不支持或路径不合法
        PermissionError: 无写入权限
    """
    # 参数校验
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入数据必须是pandas DataFrame")

    file_path = Path(file_path)
    if not file_path.parent.exists():
        os.makedirs(file_path.parent, exist_ok=True)
    # 自动判断文件类型
    if file_type == "auto":
        file_type = file_path.suffix[1:].lower()  # 去除点号
    # 统一转换文件类型标识
    file_type = file_type.replace("xlsx", "excel")  # 处理Excel扩展名
    # 选择保存方法
    save_methods = {
        "csv": df.to_csv,
        "excel": df.to_excel,
        "json": df.to_json,
        "parquet": df.to_parquet,
        "feather": df.to_feather,
        "html": df.to_html,
        "hdf": df.to_hdf,
        "pickle": df.to_pickle
    }
    # 设置格式默认参数
    format_defaults = {
        "csv": {"index": False},
        "excel": {"index": False, "engine": "openpyxl"},
        "parquet": {"engine": "pyarrow"},
        "feather": {}
    }
    if file_type not in save_methods:
        raise ValueError(f"不支持的格式: {file_type}，支持格式：{list(save_methods.keys())}")

    # 合并默认参数和用户参数
    params = format_defaults.get(file_type, {})
    params.update(kwargs)

    try:
        # 特殊处理Excel需要ExcelWriter
        if file_type == "excel":
            with pd.ExcelWriter(file_path, engine=params.pop("engine")) as writer:
                df.to_excel(writer, **params)
        else:
            save_methods[file_type](file_path, **params)

        print(f"成功保存文件：{file_path}（{df.shape} 行×列）")

    except Exception as e:
        raise RuntimeError(f"保存失败: {str(e)}") from e



