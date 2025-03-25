import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset,Dataset, DataLoader
from typing import Tuple
from torch.cuda.amp import autocast, GradScaler

## 定义常数
MNIST_DIM=784
  #           -----------------   该文件仅仅用于定义使用的各类深度学习模型  --------------------  #
class MLP_Imputer:
    """基于MLP的DataFrame缺失值填补器"""
    def __init__(self, hidden_dim=64, num_layers=3, epochs=100, lr=0.001):
        """
        初始化参数:
            hidden_dim: 隐藏层维度
            num_layers: 网络层数
            epochs: 训练轮数
            lr: 学习率
        """
        self.scaler = MinMaxScaler()
        self.model = None
        self.hparams = {  #初始化参数设置
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'epochs': epochs,
            'lr': lr
        }
    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """输入数据验证"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("输入必须是pandas DataFrame")
        if df.empty:
            raise ValueError("输入DataFrame不能为空")
        if not df.isna().any().any():
            raise ValueError("输入数据必须包含缺失值")
        return df.select_dtypes(include=np.number)  # 仅处理数值列

    def _build_model(self, input_dim: int) -> nn.Module:
        """构建MLP网络"""
        layers = []
        in_dim = input_dim
        for _ in range(self.hparams['num_layers'] - 1):
            layers.append(nn.Linear(in_dim, self.hparams['hidden_dim']))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = self.hparams['hidden_dim']
        layers.append(nn.Linear(in_dim, input_dim))
        return nn.Sequential(*layers)

    def _preprocess(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """数据预处理"""
        # 归一化
        scaled_data = self.scaler.fit_transform(df.fillna(df.mean()))  # 临时填充，将数据临时填充为0，这里应该修改为临时填充为平均值，可以增加一个参乎上
        # 生成缺失掩码
        mask = (~df.isna()).astype(float).values
        return torch.FloatTensor(scaled_data), torch.FloatTensor(mask)

    def fit(self, df: pd.DataFrame) -> None:
        """训练插补模型"""
        # 输入验证与预处理
        clean_df = self._validate_input(df)
        data_tensor, mask_tensor = self._preprocess(clean_df)

        # 模型初始化
        self.model = self._build_model(input_dim=data_tensor.shape[1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams['lr'])
        criterion = nn.MSELoss()

        # 创建数据集
        dataset = torch.utils.data.TensorDataset(data_tensor, mask_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 训练循环
        for epoch in range(self.hparams['epochs']):
            total_loss = 0
            for batch_data, batch_mask in loader:
                optimizer.zero_grad()

                # 前向传播
                pred = self.model(batch_data)

                # 仅计算缺失位置的损失
                loss = criterion(pred * (1 - batch_mask), batch_data * (1 - batch_mask))

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.hparams['epochs']}, Loss: {total_loss / len(loader):.8f}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行插补"""
        if self.model is None:
            raise RuntimeError("需要先调用fit方法训练模型")

        # 输入验证
        clean_df = self._validate_input(df)
        original_index = clean_df.index
        original_columns = clean_df.columns

        # 预处理
        scaled_data, mask = self._preprocess(clean_df)

        # 预测
        self.model.eval()
        with torch.no_grad():
            pred = self.model(scaled_data)
            # 合并已知值与预测值
            filled_data = scaled_data * mask + pred * (1 - mask)

        # 逆标准化
        filled_df = pd.DataFrame(
            self.scaler.inverse_transform(filled_data.numpy()),
            index=original_index,
            columns=original_columns
        )
        # 保留原始非数值列
        df[filled_df.columns] = filled_df
        return df
class RNNModel(nn.Module):
    """RNN模型结构（参考基础RNN实现[[3][6]]）"""

    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x形状：(batch_size, seq_len, input_dim)
        out, _ = self.rnn(x)
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out
class RNN_Imputer:
    """
    RNN缺失值插补器（参考BRITS的掩码处理机制）
    输入：带缺失值的DataFrame
    输出：完整DataFrame
    """

    def __init__(self, seq_len=10, hidden_dim=64, n_layers=2, epochs=100):
        """
        参数：
            seq_len: 时间窗口长度（参考时间序列建模）
            hidden_dim: RNN隐藏层维度
            n_layers: RNN堆叠层数
            epochs: 训练轮数
        """
        self.scaler = MinMaxScaler()
        self.seq_len = seq_len
        self.model = None
        self.hparams = {
            'input_dim': None,
            'hidden_dim': hidden_dim,
            'n_layers': n_layers,
            'epochs': epochs
        }
    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """输入验证"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("输入必须是pandas DataFrame")
        if df.empty:
            raise ValueError("输入DataFrame不能为空")
        if not df.isna().any().any():
            raise ValueError("输入数据必须包含缺失值")
        return df.select_dtypes(include=np.number)

    def _create_sequences(self, data: np.ndarray) -> tuple:
        """创建时间序列样本（参考时间序列处理）"""
        sequences = []
        masks = []
        for i in range(len(data) - self.seq_len + 1):
            seq = data[i:i + self.seq_len]
            mask = (~np.isnan(seq)).astype(float)
            seq = np.nan_to_num(seq, nan=0)  # 临时填充缺失值
            sequences.append(seq)
            masks.append(mask)
        return torch.FloatTensor(sequences), torch.FloatTensor(masks)

    def fit(self, df: pd.DataFrame) -> None:
        """训练模型"""
        clean_df = self._validate_input(df)
        self.hparams['input_dim'] = clean_df.shape[1]

        # 数据预处理
        scaled_data = self.scaler.fit_transform(clean_df)
        sequences, masks = self._create_sequences(scaled_data)

        # 初始化模型
        self.model = RNNModel(
            input_dim=self.hparams['input_dim'],
            hidden_dim=self.hparams['hidden_dim'],
            n_layers=self.hparams['n_layers']
        )
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.MSELoss()

        # 训练循环
        for epoch in range(self.hparams['epochs']):
            total_loss = 0
            for seq, mask in zip(sequences, masks):
                optimizer.zero_grad()

                # 前向传播（参考BRITS的掩码机制）
                output = self.model(seq.unsqueeze(0))  # 添加batch维度

                # 仅计算缺失位置的损失
                loss = criterion(output * (1 - mask), seq * (1 - mask))

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(sequences):.8f}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行插补"""
        if self.model is None:
            raise RuntimeError("需要先调用fit方法训练模型")

        clean_df = self._validate_input(df)
        original_index = clean_df.index
        original_cols = clean_df.columns

        # 预处理
        scaled_data = self.scaler.transform(clean_df.fillna(0))
        sequences, masks = self._create_sequences(scaled_data)

        # 预测
        self.model.eval()
        filled_data = scaled_data.copy()
        with torch.no_grad():
            for i, (seq, mask) in enumerate(zip(sequences, masks)):
                output = self.model(seq.unsqueeze(0))
                seq_filled = seq * (mask) + output.squeeze() * (1 - mask)
                filled_data[i:i + self.seq_len] = seq_filled.numpy()

        # 逆标准化
        filled_df = pd.DataFrame(
            self.scaler.inverse_transform(filled_data),
            index=original_index,
            columns=original_cols
        )
        # 保留原始非数值列
        df[filled_df.columns] = filled_df
        return df
class VAE_Imputer:
    """基于GPU加速的VAE缺失值填补器"""

    def __init__(self, latent_dim=16, hidden_dim=64, epochs=200, batch_size=512):
        """
        参数说明：
            latent_dim: 潜在空间维度（建议8-64）
            hidden_dim: 隐藏层维度（建议32-256）
            epochs: 训练轮次
            batch_size: 批处理大小（根据GPU显存调整）
        """
        self.scaler = MinMaxScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hparams = {
            'input_dim': None,
            'latent_dim': latent_dim,
            'hidden_dim': hidden_dim,
            'epochs': epochs,
            'batch_size': batch_size
        }
        print(f"使用设备：{self.device}")

    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """输入数据验证"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("输入必须是pandas DataFrame")
        if df.empty:
            raise ValueError("输入DataFrame不能为空")
        if not df.isna().any().any():
            raise ValueError("输入数据必须包含缺失值")
        return df.select_dtypes(include=np.number)

    def _build_model(self) -> nn.Module:
        """构建优化后的VAE模型"""
        return VAE_Model(
            input_dim=self.hparams['input_dim'],
            hidden_dim=self.hparams['hidden_dim'],
            latent_dim=self.hparams['latent_dim']
        ).to(self.device)

    def _preprocess(self, df: pd.DataFrame) -> tuple:
        """数据预处理（GPU优化版）"""
        # 处理数值列
        numeric_df = self._validate_input(df)

        # 归一化并转换为Tensor
        scaled_data = self.scaler.fit_transform(numeric_df.fillna(0))
        mask = (~numeric_df.isna()).values.astype(np.float32)

        # 转换为GPU Tensor
        data_tensor = torch.as_tensor(scaled_data, dtype=torch.float32, device=self.device)
        mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device=self.device)

        return data_tensor, mask_tensor, numeric_df.columns, df.index

    def fit(self, df: pd.DataFrame) -> None:
        """训练过程（GPU加速版）"""
        # 数据预处理
        data_tensor, mask_tensor, _, _ = self._preprocess(df)
        self.hparams['input_dim'] = data_tensor.shape[1]

        # 创建数据加载器
        dataset = TensorDataset(data_tensor, mask_tensor)
        loader = DataLoader(dataset,
                            batch_size=self.hparams['batch_size'],
                            shuffle=True,
                            pin_memory=True,  # 加速数据加载
                            num_workers=4)  # 并行加载

        # 初始化模型和优化器
        self.model = self._build_model()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        scaler = GradScaler()  # 混合精度训练

        # 训练循环
        self.model.train()
        for epoch in range(self.hparams['epochs']):
            total_loss = 0.0

            for batch_data, batch_mask in loader:
                # 生成随机缺失掩码（在GPU上）
                rand_mask = (torch.rand_like(batch_data, device=self.device) > 0.2).float()
                inputs = batch_data * rand_mask

                optimizer.zero_grad(set_to_none=True)  # 内存优化

                # 混合精度前向传播
                with autocast():
                    recon, mu, logvar = self.model(inputs)
                    recon_loss = nn.MSELoss()(recon * batch_mask, batch_data * batch_mask)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + kl_loss * 0.1

                # 反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

            # 打印训练进度
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.hparams['epochs']} | Loss: {total_loss / len(loader):.10f}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行填补（GPU加速版）"""
        if self.model is None:
            raise RuntimeError("需要先调用fit方法训练模型")

        # 数据预处理
        data_tensor, mask_tensor, numeric_cols, original_index = self._preprocess(df)

        # 模型预测
        self.model.eval()
        with torch.no_grad(), autocast():
            recon, _, _ = self.model(data_tensor)
            filled_data = data_tensor * mask_tensor + recon * (1 - mask_tensor)

        # 逆标准化并转换为DataFrame
        filled_np = filled_data.cpu().numpy()
        filled_df = pd.DataFrame(
            self.scaler.inverse_transform(filled_np),
            index=original_index,
            columns=numeric_cols
        )
        # 保留原始非数值列
        df[filled_df.columns] = filled_df
        return df
class VAE_Model(nn.Module):
    """优化后的VAE模型结构"""
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 潜在空间参数层
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim // 2, latent_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """参数初始化优化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple:
        # 编码过程
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)

        # 重参数化采样
        z = self.reparameterize(mu, logvar)

        # 解码重构
        return self.decoder(z), mu, logvar
class LSTM_Imputer:
    """基于LSTM的缺失值填补器（支持GPU加速）"""

    def __init__(self, seq_len=10, hidden_dim=64, num_layers=2, epochs=100, batch_size=512):
        """
        参数说明：
            seq_len: 时间窗口长度（参考时间序列建模）
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM堆叠层数
            epochs: 训练轮次
            batch_size: 批处理大小（根据GPU显存调整）
        """
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hparams = {
            'seq_len': seq_len,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'epochs': epochs,
            'batch_size': batch_size,
            'input_dim': None
        }
        print(f"使用设备：{self.device}")

    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """输入验证"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("输入必须是pandas DataFrame")
        if df.empty:
            raise ValueError("输入DataFrame不能为空")
        if not df.isna().any().any():
            raise ValueError("输入数据必须包含缺失值")
        return df.select_dtypes(include=np.number)

    def _create_sequences(self, data: np.ndarray) -> tuple:
        """创建时间序列样本（优化GPU内存）"""
        num_samples = data.shape[0] - self.hparams['seq_len'] + 1
        sequences = torch.zeros((num_samples, self.hparams['seq_len'], data.shape[1]),
                                dtype=torch.float32, device=self.device)
        masks = torch.zeros_like(sequences)

        for i in range(num_samples):
            seq = data[i:i + self.hparams['seq_len']]
            mask = (~np.isnan(seq)).astype(float)
            seq = np.nan_to_num(seq, nan=0)

            sequences[i] = torch.as_tensor(seq, device=self.device)
            masks[i] = torch.as_tensor(mask, device=self.device)

        return sequences, masks

    def fit(self, df: pd.DataFrame) -> None:
        """训练过程（GPU加速版）"""
        clean_df = self._validate_input(df)
        self.hparams['input_dim'] = clean_df.shape[1]

        # 数据预处理
        scaled_data = self.scaler.fit_transform(clean_df)
        sequences, masks = self._create_sequences(scaled_data)

        # 创建数据加载器
        dataset = TensorDataset(sequences, masks)
        loader = DataLoader(dataset,
                            batch_size=self.hparams['batch_size'],
                            shuffle=True,
                            pin_memory=True,
                            num_workers=4)

        # 初始化模型
        self.model = LSTM_Model(
            input_dim=self.hparams['input_dim'],
            hidden_dim=self.hparams['hidden_dim'],
            num_layers=self.hparams['num_layers']
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # 训练循环
        self.model.train()
        for epoch in range(self.hparams['epochs']):
            total_loss = 0.0
            for batch_seq, batch_mask in loader:
                optimizer.zero_grad()

                # 前向传播
                outputs = self.model(batch_seq)

                # 计算缺失位置的损失
                loss = criterion(outputs * (1 - batch_mask), batch_seq * (1 - batch_mask))

                # 反向传播
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.hparams['epochs']} | Loss: {total_loss / len(loader):.10f}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行填补（支持完整序列处理）"""
        if self.model is None:
            raise RuntimeError("需要先调用fit方法训练模型")

        clean_df = self._validate_input(df)
        original_index = clean_df.index
        original_cols = clean_df.columns

        # 数据预处理
        scaled_data = self.scaler.transform(clean_df.fillna(0))
        full_seq = torch.as_tensor(scaled_data, dtype=torch.float32, device=self.device)

        # 滑动窗口填补
        self.model.eval()
        with torch.no_grad():
            filled_data = full_seq.clone()
            for i in range(len(full_seq) - self.hparams['seq_len'] + 1):
                window = full_seq[i:i + self.hparams['seq_len']].unsqueeze(0)
                pred = self.model(window)
                mask = torch.isnan(full_seq[i:i + self.hparams['seq_len']]).float()
                filled_data[i:i + self.hparams['seq_len']] = window[0] * (1 - mask) + pred[0] * mask

        # 逆标准化
        filled_np = filled_data.cpu().numpy()
        filled_df = pd.DataFrame(
            self.scaler.inverse_transform(filled_np),
            index=original_index,
            columns=original_cols
        )

        # 保留原始非数值列
        df[filled_df.columns] = filled_df
        return df
class LSTM_Model(nn.Module):
    """优化后的LSTM网络结构"""

    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x形状: (batch_size, seq_len, input_dim)   #修改为全序列输出
        out, _ = self.lstm(x)  # out形状: (batch_size, seq_len, hidden_dim)  #目前就一层
        return self.fc(out)  # 输出形状: (batch_size, seq_len, input_dim)
class GAIN_Imputer:
    """基于生成对抗网络的缺失数据填补器（参考GAIN论文实现）"""

    def __init__(self, hint_rate=0.7, alpha=100, epochs=10000, batch_size=128):
        """
        参数说明：
            hint_rate: 提示向量生成概率（参考论文Hint机制）
            alpha: 重建损失权重（控制生成质量）
            epochs: 训练轮次
            batch_size: 批处理大小
        """
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hparams = {
            'hint_rate': hint_rate,
            'alpha': alpha,
            'epochs': epochs,
            'batch_size': batch_size,
            'input_dim': None
        }
        self.G = None
        self.D = None

    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """输入验证（参考数据预处理）"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("输入必须是pandas DataFrame")
        if df.empty:
            raise ValueError("输入DataFrame不能为空")
        if not df.isna().any().any():
            raise ValueError("输入数据必须包含缺失值")
        return df.select_dtypes(include=np.number)

    def _generate_mask(self, data: np.ndarray) -> np.ndarray:
        """生成二进制掩码矩阵（参考）"""
        return (~np.isnan(data)).astype(np.float32)

    def _generate_hint(self, mask: np.ndarray) -> np.ndarray:
        """生成提示矩阵（参考GAIN论文）"""
        hint = np.random.binomial(1, self.hparams['hint_rate'], size=mask.shape)
        return mask * hint

    def _build_generator(self) -> nn.Module:
        """生成器网络（参考结构）"""
        return nn.Sequential(
            nn.Linear(self.hparams['input_dim'] * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.hparams['input_dim']),
            nn.Sigmoid()
        ).to(self.device)

    def _build_discriminator(self) -> nn.Module:
        """判别器网络（参考结构）"""
        return nn.Sequential(
            nn.Linear(self.hparams['input_dim'] * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.hparams['input_dim']),
            nn.Sigmoid()
        ).to(self.device)

    def fit(self, df: pd.DataFrame) -> None:
        """训练过程（参考对抗训练思想）"""
        numeric_df = self._validate_input(df)
        self.hparams['input_dim'] = numeric_df.shape[1]

        # 数据预处理
        scaled_data = self.scaler.fit_transform(numeric_df.fillna(0))
        mask = self._generate_mask(numeric_df.values)

        # 转换为PyTorch张量
        data_tensor = torch.tensor(scaled_data, dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device)

        # 初始化网络
        self.G = self._build_generator()
        self.D = self._build_discriminator()
        optimizer_G = torch.optim.Adam(self.G.parameters())
        optimizer_D = torch.optim.Adam(self.D.parameters())

        # 训练循环
        for epoch in range(self.hparams['epochs']):
            # 随机采样批次
            idx = np.random.randint(0, len(data_tensor), self.hparams['batch_size'])
            batch_data = data_tensor[idx]
            batch_mask = mask_tensor[idx]

            # 生成提示向量
            hint = self._generate_hint(batch_mask.cpu().numpy())
            hint_tensor = torch.tensor(hint, dtype=torch.float32, device=self.device)

            # 生成器前向传播
            gen_input = torch.cat([batch_data * batch_mask, batch_mask], dim=1)
            gen_output = self.G(gen_input)
            imputed_data = batch_data * batch_mask + gen_output * (1 - batch_mask)

            # 判别器前向传播
            disc_input = torch.cat([imputed_data, hint_tensor], dim=1)
            D_pred = self.D(disc_input)

            # 计算损失
            # 生成器损失（参考论文公式）
            G_loss_adv = -torch.mean((1 - batch_mask) * torch.log(D_pred + 1e-8))
            G_loss_mse = torch.mean((batch_data * batch_mask - imputed_data * batch_mask) ** 2)
            G_loss = G_loss_adv + self.hparams['alpha'] * G_loss_mse

            # 判别器损失
            D_loss = -torch.mean(batch_mask * torch.log(D_pred + 1e-8) +
                                 (1 - batch_mask) * torch.log(1 - D_pred + 1e-8))

            # 反向传播
            optimizer_G.zero_grad()
            G_loss.backward(retain_graph=True)
            optimizer_G.step()

            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()

            if epoch % 1000 == 0:
                print(f"Epoch {epoch} | G_loss: {G_loss.item():.4f} | D_loss: {D_loss.item():.4f}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行填补（参考实现）"""
        if self.G is None:
            raise RuntimeError("需要先调用fit方法训练模型")

        numeric_df = self._validate_input(df)
        original_index = numeric_df.index
        original_cols = numeric_df.columns

        # 数据预处理
        scaled_data = self.scaler.transform(numeric_df.fillna(0))
        mask = self._generate_mask(numeric_df.values)

        # 生成填补数据
        data_tensor = torch.tensor(scaled_data, dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            gen_input = torch.cat([data_tensor * mask_tensor, mask_tensor], dim=1)
            gen_output = self.G(gen_input)
            filled_data = data_tensor * mask_tensor + gen_output * (1 - mask_tensor)

        # 逆标准化
        filled_np = filled_data.cpu().numpy()
        filled_df = pd.DataFrame(
            self.scaler.inverse_transform(filled_np),
            index=original_index,
            columns=original_cols
        )
        # 保留原始非数值列
        df[filled_df.columns] = filled_df
        return df
class DBN_Imputer:
    """基于深度信念网络（DBN）的缺失值填补器"""

    def __init__(self, hidden_dims=[128, 64], k=3, pretrain_epochs=50, finetune_epochs=100, batch_size=512):
        """
        参数说明：
            hidden_dims: 各隐藏层维度（例如[128,64]表示两层）
            k: 对比散度算法的采样步数
            pretrain_epochs: 预训练轮次
            finetune_epochs: 微调轮次
            batch_size: GPU批处理大小
        """
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hparams = {
            'hidden_dims': hidden_dims,
            'k': k,
            'pretrain_epochs': pretrain_epochs,
            'finetune_epochs': finetune_epochs,
            'batch_size': batch_size,
            'input_dim': None
        }
        self.rbms = []
        self.finetune_model = None

    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """输入验证"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("输入必须是pandas DataFrame")
        if df.empty:
            raise ValueError("输入DataFrame不能为空")
        if not df.isna().any().any():
            raise ValueError("输入数据必须包含缺失值")
        return df.select_dtypes(include=np.number)

    def _preprocess(self, df: pd.DataFrame) -> tuple:
        """数据预处理（GPU优化版）"""
        numeric_df = self._validate_input(df)
        scaled_data = self.scaler.fit_transform(numeric_df.fillna(0))
        mask = (~numeric_df.isna()).values.astype(np.float32)

        data_tensor = torch.as_tensor(scaled_data, dtype=torch.float32, device=self.device)
        mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device=self.device)

        return data_tensor, mask_tensor, numeric_df.columns, df.index

    class RBM(nn.Module):
        """受限玻尔兹曼机（RBM）实现"""

        def __init__(self, visible_dim, hidden_dim):
            super().__init__()
            self.W = nn.Parameter(torch.randn(hidden_dim, visible_dim) * 0.1)
            self.v_bias = nn.Parameter(torch.zeros(visible_dim))
            self.h_bias = nn.Parameter(torch.zeros(hidden_dim))

        def sample_h(self, v):
            """从可见层采样隐藏层"""
            p_h = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
            return p_h, torch.bernoulli(p_h)

        def sample_v(self, h):
            """从隐藏层采样可见层"""
            p_v = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
            return p_v, torch.bernoulli(p_v)

        def contrastive_divergence(self, v, k=3):
            """对比散度算法"""
            # 正向传播
            h0_prob, h0_sample = self.sample_h(v)

            # Gibbs采样
            vk = v.clone()
            for _ in range(k):
                _, hk_sample = self.sample_h(vk)
                vk_prob, vk_sample = self.sample_v(hk_sample)
                vk = vk_sample  # 使用采样值而非概率值

            # 计算梯度
            positive_grad = torch.matmul(h0_prob.t(), v)
            negative_grad = torch.matmul(self.sample_h(vk)[0].t(), vk)

            return (positive_grad - negative_grad) / v.size(0), vk

    def _pretrain(self, train_loader):
        """逐层预训练RBM"""
        visible_dim = self.hparams['input_dim']
        for i, hidden_dim in enumerate(self.hparams['hidden_dims']):
            print(f"预训练第{i + 1}层RBM ({visible_dim}->{hidden_dim})")

            # 初始化RBM
            rbm = self.RBM(visible_dim, hidden_dim).to(self.device)
            optimizer = optim.SGD(rbm.parameters(), lr=0.01)

            # 训练当前层
            for epoch in range(self.hparams['pretrain_epochs']):
                total_loss = 0
                for batch, _ in train_loader:
                    batch = batch.to(self.device)

                    # 对比散度
                    grad, recon = rbm.contrastive_divergence(batch, k=self.hparams['k'])

                    # 手动更新参数
                    for param in rbm.parameters():
                        param.grad = -grad.view_as(param)  # 负梯度方向
                    optimizer.step()

                    total_loss += torch.mean((batch - recon) ** 2).item()

                if epoch % 10 == 0:
                    print(f"Epoch {epoch + 1} | Loss: {total_loss / len(train_loader):.4f}")

            self.rbms.append(rbm)
            visible_dim = hidden_dim  # 下一层的输入维度

    def _build_finetune_model(self):
        """构建微调模型"""
        layers = []
        input_dim = self.hparams['input_dim']

        # 添加编码层
        for rbm in self.rbms:
            layers += [
                nn.Linear(input_dim, rbm.W.shape[0]),
                nn.Sigmoid()
            ]
            input_dim = rbm.W.shape[0]

        # 添加解码层（对称结构）
        for dim in reversed(self.hparams['hidden_dims'][:-1]):
            layers += [
                nn.Linear(input_dim, dim),
                nn.Sigmoid()
            ]
            input_dim = dim

        # 最终重建层
        layers.append(nn.Linear(input_dim, self.hparams['input_dim']))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers).to(self.device)

    def fit(self, df: pd.DataFrame):
        """训练过程"""
        data_tensor, mask_tensor, cols, idx = self._preprocess(df)
        self.hparams['input_dim'] = data_tensor.shape[1]

        # 创建数据集（含掩码）
        dataset = TensorDataset(data_tensor, mask_tensor)
        loader = DataLoader(dataset,
                            batch_size=self.hparams['batch_size'],
                            shuffle=True,
                            pin_memory=True,
                            num_workers=4)

        # 预训练阶段
        self._pretrain(loader)

        # 微调阶段
        self.finetune_model = self._build_finetune_model()
        optimizer = optim.Adam(self.finetune_model.parameters())
        criterion = nn.MSELoss()

        print("\n开始微调训练...")
        for epoch in range(self.hparams['finetune_epochs']):
            total_loss = 0
            for batch_data, batch_mask in loader:
                optimizer.zero_grad()

                # 前向传播
                outputs = self.finetune_model(batch_data)

                # 计算缺失部分的损失
                loss = criterion(outputs * (1 - batch_mask), batch_data * (1 - batch_mask))

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1} | Loss: {total_loss / len(loader):.4f}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行填补"""
        if not self.rbms or not self.finetune_model:
            raise RuntimeError("需要先调用fit方法训练模型")

        # 数据预处理
        data_tensor, mask_tensor, numeric_cols, original_index = self._preprocess(df)

        # 生成填补数据
        self.finetune_model.eval()
        with torch.no_grad():
            outputs = self.finetune_model(data_tensor)
            filled_data = data_tensor * mask_tensor + outputs * (1 - mask_tensor)

        # 逆标准化
        filled_np = filled_data.cpu().numpy()
        filled_df = pd.DataFrame(
            self.scaler.inverse_transform(filled_np),
            index=original_index,
            columns=numeric_cols
        )

        # 保留原始非数值列
        df[filled_df.columns] = filled_df
        return df
class GRU_Imputer:
    """基于GRU的缺失值填补器（支持GPU加速）"""

    def __init__(self, seq_len=10, hidden_dim=64, num_layers=2, epochs=100, batch_size=512):
        """
        参数说明：
            seq_len: 时间窗口长度（建议3-30）
            hidden_dim: GRU隐藏层维度
            num_layers: GRU堆叠层数
            epochs: 训练轮次
            batch_size: 批处理大小（根据GPU显存调整）
        """
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hparams = {
            'seq_len': seq_len,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'epochs': epochs,
            'batch_size': batch_size,
            'input_dim': None
        }
        self.model = None
        print(f"使用设备：{self.device}")

    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """输入数据验证"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("输入必须是pandas DataFrame")
        if df.empty:
            raise ValueError("输入DataFrame不能为空")
        if not df.isna().any().any():
            raise ValueError("输入数据必须包含缺失值")
        return df.select_dtypes(include=np.number)

    def _create_sequences(self, data: np.ndarray) -> tuple:
        """创建时间序列样本（GPU优化版）"""
        num_samples = data.shape[0] - self.hparams['seq_len'] + 1
        sequences = torch.zeros((num_samples, self.hparams['seq_len'], data.shape[1]),
                                dtype=torch.float32, device=self.device)
        masks = torch.zeros_like(sequences)

        for i in range(num_samples):
            seq = data[i:i + self.hparams['seq_len']]
            mask = (~np.isnan(seq)).astype(float)
            seq = np.nan_to_num(seq, nan=0)

            sequences[i] = torch.as_tensor(seq, device=self.device)
            masks[i] = torch.as_tensor(mask, device=self.device)

        return sequences, masks

    def _build_model(self) -> nn.Module:
        """构建GRU模型结构"""
        return GRU_Model(
            input_dim=self.hparams['input_dim'],
            hidden_dim=self.hparams['hidden_dim'],
            num_layers=self.hparams['num_layers']
        ).to(self.device)

    def fit(self, df: pd.DataFrame) -> None:
        """训练过程（GPU加速）"""
        # 数据预处理
        numeric_df = self._validate_input(df)
        self.hparams['input_dim'] = numeric_df.shape[1]
        scaled_data = self.scaler.fit_transform(numeric_df)

        # 生成训练序列
        sequences, masks = self._create_sequences(scaled_data)
        dataset = TensorDataset(sequences, masks)
        loader = DataLoader(dataset,
                            batch_size=self.hparams['batch_size'],
                            shuffle=True,
                            pin_memory=True,
                            num_workers=4)

        # 初始化模型
        self.model = self._build_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # 训练循环
        self.model.train()
        for epoch in range(self.hparams['epochs']):
            total_loss = 0.0
            for batch_seq, batch_mask in loader:
                optimizer.zero_grad()

                # 前向传播
                outputs = self.model(batch_seq)

                # 计算缺失位置的损失
                loss = criterion(outputs * (1 - batch_mask), batch_seq * (1 - batch_mask))

                # 反向传播
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.hparams['epochs']} | Loss: {total_loss / len(loader):.10f}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行填补（支持完整序列处理）"""
        if self.model is None:
            raise RuntimeError("需要先调用fit方法训练模型")

        numeric_df = self._validate_input(df)
        original_index = numeric_df.index
        original_cols = numeric_df.columns

        # 数据预处理
        scaled_data = self.scaler.transform(numeric_df.fillna(0))
        full_seq = torch.as_tensor(scaled_data, dtype=torch.float32, device=self.device)

        # 滑动窗口填补
        self.model.eval()
        with torch.no_grad():
            filled_data = full_seq.clone()
            counts = torch.zeros_like(filled_data)

            for i in range(len(full_seq) - self.hparams['seq_len'] + 1):
                window = full_seq[i:i + self.hparams['seq_len']].unsqueeze(0)
                pred = self.model(window)[0]  # 获取预测序列
                mask = torch.isnan(full_seq[i:i + self.hparams['seq_len']]).float()

                # 累积填补结果
                filled_data[i:i + self.hparams['seq_len']] += pred * mask
                counts[i:i + self.hparams['seq_len']] += mask

            # 平均多窗口预测结果
            filled_data = filled_data / counts.clamp(min=1)

        # 逆标准化
        filled_np = filled_data.cpu().numpy()
        filled_df = pd.DataFrame(
            self.scaler.inverse_transform(filled_np),
            index=original_index,
            columns=original_cols
        )

        # 保留原始非数值列
        df[filled_df.columns] = filled_df
        return df
class GRU_Model(nn.Module):
    """优化的GRU网络结构"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 约束输出范围在[0,1]
        )

    def forward(self, x):
        # x形状: (batch_size, seq_len, input_dim)
        out, _ = self.gru(x)
        # 输出整个序列的预测
        return self.fc(out)


class CNN_Imputer:
    """基于CNN的缺失值填补器（支持GPU加速）"""

    def __init__(self, window_size=5, num_filters=64, epochs=100, batch_size=512):
        """
        参数说明：
            window_size: 滑动窗口大小（建议3-15）
            num_filters: 卷积核数量
            epochs: 训练轮次
            batch_size: GPU批处理大小
        """
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hparams = {
            'window_size': window_size,
            'num_filters': num_filters,
            'epochs': epochs,
            'batch_size': batch_size,
            'input_dim': None
        }
        self.model = None
        print(f"使用设备：{self.device}")

    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """输入验证 """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("输入必须是pandas DataFrame")
        if df.empty:
            raise ValueError("输入DataFrame不能为空")
        if not df.isna().any().any():
            raise ValueError("输入数据必须包含缺失值")
        return df.select_dtypes(include=np.number)

    def _create_windows(self, data: np.ndarray) -> tuple:
        """创建滑动窗口样本（GPU优化版）"""
        num_samples = data.shape[0] - self.hparams['window_size'] + 1
        windows = torch.zeros((num_samples, 1, self.hparams['window_size'], data.shape[1]),
                              dtype=torch.float32, device=self.device)
        masks = torch.zeros_like(windows)

        for i in range(num_samples):
            window = data[i:i + self.hparams['window_size']]
            mask = (~np.isnan(window)).astype(float)
            window = np.nan_to_num(window, nan=0)

            # 将窗口转换为类似图像的2D格式 (通道, 高度, 宽度)
            windows[i] = torch.as_tensor(window.T, device=self.device).unsqueeze(0)
            masks[i] = torch.as_tensor(mask.T, device=self.device).unsqueeze(0)

        return windows, masks

    def _build_model(self) -> nn.Module:
        """构建CNN模型 [[7][8]]"""
        return nn.Sequential(
            # 输入形状: (batch, 1, window_size, num_features)
            nn.Conv2d(1, self.hparams['num_filters'], kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(self.hparams['num_filters'], self.hparams['num_filters'] // 2, kernel_size=(3, 3)),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(self.hparams['num_filters'] // 2 * (self.hparams['window_size'] // 2 - 2) * (
                        self.hparams['input_dim'] - 2),
                      self.hparams['window_size'] * self.hparams['input_dim']),
            nn.Sigmoid()
        ).to(self.device)

    def fit(self, df: pd.DataFrame) -> None:
        """训练过程（参考[[5][8]]）"""
        numeric_df = self._validate_input(df)
        self.hparams['input_dim'] = numeric_df.shape[1]
        scaled_data = self.scaler.fit_transform(numeric_df)

        # 生成窗口数据
        windows, masks = self._create_windows(scaled_data)
        dataset = TensorDataset(windows, masks)
        loader = DataLoader(dataset,
                            batch_size=self.hparams['batch_size'],
                            shuffle=True,
                            pin_memory=True,
                            num_workers=4)

        # 初始化模型
        self.model = self._build_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # 训练循环
        self.model.train()
        for epoch in range(self.hparams['epochs']):
            total_loss = 0.0
            for batch_windows, batch_masks in loader:
                optimizer.zero_grad()

                # 前向传播
                outputs = self.model(batch_windows)
                outputs = outputs.view(batch_windows.shape)

                # 计算缺失位置的损失
                loss = criterion(outputs * (1 - batch_masks), batch_windows * (1 - batch_masks))

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.hparams['epochs']} | Loss: {total_loss / len(loader):.10f}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行填补（参考[[1][6]]）"""
        if self.model is None:
            raise RuntimeError("需要先调用fit方法训练模型")

        numeric_df = self._validate_input(df)
        original_index = numeric_df.index
        original_cols = numeric_df.columns

        # 数据预处理
        scaled_data = self.scaler.transform(numeric_df.fillna(0))
        full_seq = torch.as_tensor(scaled_data, dtype=torch.float32, device=self.device)

        # 滑动窗口填补
        self.model.eval()
        with torch.no_grad():
            filled_data = full_seq.clone()
            counts = torch.zeros_like(filled_data)

            for i in range(len(scaled_data) - self.hparams['window_size'] + 1):
                window = scaled_data[i:i + self.hparams['window_size']].T
                input_tensor = torch.as_tensor(window, device=self.device).unsqueeze(0).unsqueeze(0)

                pred = self.model(input_tensor).view(self.hparams['window_size'], -1).T
                mask = np.isnan(numeric_df.iloc[i:i + self.hparams['window_size']]).values

                filled_data[i:i + self.hparams['window_size']] += torch.as_tensor(pred * mask, device=self.device)
                counts[i:i + self.hparams['window_size']] += torch.as_tensor(mask, device=self.device)

            # 平均多窗口预测结果
            filled_data = filled_data / counts.clamp(min=1)

        # 逆标准化
        filled_np = filled_data.cpu().numpy()
        filled_df = pd.DataFrame(
            self.scaler.inverse_transform(filled_np),
            index=original_index,
            columns=original_cols
        )

        # 保留原始非数值列
        df[filled_df.columns] = filled_df
        return df


