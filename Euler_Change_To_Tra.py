import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def euler_to_rotation_matrix(roll, pitch, yaw):
    """将欧拉角转换为旋转矩阵（X-Y-Z固定轴顺序）"""
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    return R_z @ R_y @ R_x  # 固定轴顺序：X→Y→Z


def compute_trajectory(df):
    """计算三维轨迹"""
    positions = np.zeros((len(df), 3))
    current_pos = np.zeros(3)

    for i in range(len(df)):
        row = df.iloc[i]
        R = euler_to_rotation_matrix(row['roll(deg)'], row['pitch(deg)'], row['yaw(deg)'])
        direction = R[:, 0]  # 假设前进方向为局部X轴
        current_pos += direction
        positions[i] = current_pos

    return positions


def plot_3d_trajectory(positions):
    """绘制三维轨迹图"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            marker='o', linestyle='-', linewidth=1, markersize=4)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.title('3D Trajectory from Euler Angles')
    plt.show()


# 示例使用
if __name__ == "__main__":
    # 创建示例数据（M行3列的DataFrame）
    data = {
        'roll': [0, 30, 0, -30, 0],
        'pitch': [0, 0, 20, 0, -20],
        'yaw': [0, 0, 0, 0, 0]
    }
    df = pd.DataFrame(data)

    # 计算轨迹并绘图
    trajectory = compute_trajectory(df)
    plot_3d_trajectory(trajectory)