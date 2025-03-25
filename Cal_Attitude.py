import pandas as pd
import numpy as np
from math import sqrt, atan2, asin, degrees


class AdaptiveAttitudeEstimator:
    def __init__(self, beta=0.1, zeta=0.1, dt=0.01):
        # 算法参数
        self.beta = beta  # Madgwick算法收敛率
        self.zeta = zeta  # 陀螺仪漂移补偿
        self.dt = dt  # 采样周期
        self.use_mag = False  # 磁力计使用标志

        # 四元数初始化
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

        # 磁力计相关参数
        self.mag_ref = np.array([1.0, 0.0, 0.0])  # 参考磁场向量

    def _normalize(self, v):
        return v / np.linalg.norm(v) if np.linalg.norm(v) != 0 else v

    def _quat_mult(self, q1, q2):
        # 四元数乘法
        return np.array([
            q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
            q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
            q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
            q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
        ])

    def update(self, accel, gyro, mag=None):
        # 数据归一化
        accel = self._normalize(accel)
        if mag is not None:
            mag = self._normalize(mag)
            self.use_mag = True

        # Madgwick算法核心
        q = self.q
        gyro = gyro * (np.pi / 180)  # 转换为rad/s

        # 加速度计误差计算
        f = np.array([
            2 * (q[1] * q[3] - q[0] * q[2]) - accel[0],
            2 * (q[0] * q[1] + q[2] * q[3]) - accel[1],
            2 * (0.5 - q[1] ** 2 - q[2] ** 2) - accel[2]
        ])

        # 磁力计误差计算（九轴模式）
        if self.use_mag:
            h = self._quat_mult(q, self._quat_mult([0, *mag], [q[0], -q[1], -q[2], -q[3]]))
            b = np.array([sqrt(h[1] ** 2 + h[2] ** 2), 0, h[3]])
            f_mag = np.array([
                (b[0] * (0.5 - q[2] ** 2 - q[3] ** 2) + b[2] * (q[1] * q[3] - q[0] * q[2])) - mag[0],
                (b[0] * (q[1] * q[2] - q[0] * q[3]) + b[2] * (q[0] * q[1] + q[2] * q[3])) - mag[1],
                (b[0] * (q[0] * q[2] + q[1] * q[3]) + b[2] * (0.5 - q[1] ** 2 - q[2] ** 2)) - mag[2]
            ])
            f = np.concatenate((f, f_mag))

        # 梯度下降法
        J = np.array([
            [-2 * q[2], 2 * q[3], -2 * q[0], 2 * q[1]],
            [2 * q[1], 2 * q[0], 2 * q[3], 2 * q[2]],
            [0, -4 * q[1], -4 * q[2], 0]
        ]) if not self.use_mag else np.vstack((
            np.array([[-2 * q[2], 2 * q[3], -2 * q[0], 2 * q[1]],
                      [2 * q[1], 2 * q[0], 2 * q[3], 2 * q[2]],
                      [0, -4 * q[1], -4 * q[2], 0]]),
            np.array([[0, 0, -2 * q[2], 2 * q[3]],
                      [-2 * q[3], 2 * q[2], 2 * q[1], 2 * q[0]],
                      [2 * q[2], 2 * q[3], 2 * q[0], 2 * q[1]]])
        ))

        step = J.T.dot(f)
        step = self._normalize(step)

        # 四元数更新
        q_dot = 0.5 * self._quat_mult(q, [0, *gyro]) - self.beta * step
        q += q_dot * self.dt
        self.q = self._normalize(q)

        return self._quat_to_euler()

    def _quat_to_euler(self):
        # 四元数转欧拉角（Z-Y-X顺规）
        w, x, y, z = self.q
        roll = degrees(atan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2)))
        pitch = degrees(asin(2 * (w * y - z * x)))
        yaw = degrees(atan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2)))
        return np.array([roll, pitch, yaw])


def calculate_attitude(accel_df, gyro_df, mag_df=None,Frequency = 0.1):
    # 自动判断输入类型
    estimator = AdaptiveAttitudeEstimator(beta=0.1, zeta=0.1, dt=Frequency)
    results = []

    for i in range(len(accel_df)):
        accel = accel_df.iloc[i].values.astype(float)
        gyro = gyro_df.iloc[i].values.astype(float)
        mag = mag_df.iloc[i].values.astype(float) if mag_df is not None else None

        euler = estimator.update(accel, gyro, mag)
        results.append(euler)

    return pd.DataFrame(results, columns=['roll(deg)', 'pitch(deg)', 'yaw(deg)'])