import torch.nn as nn
import torch.optim as optim
def Train_DL_Model(model, data, mask, epochs=100, lr=0.01):
    """
    训练 MLP 模型。 ，model指的是使用的模型；data指的是输入的数据，mask代表标志矩阵；epoch最大循环次数；lr学习率
    """
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  #使用MSE作为损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 训练模型
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # 前向传播
        outputs = model(data)
        # 计算损失（仅在已知值上计算）
        loss = criterion(outputs * mask, data * mask)
        # 反向传播和优化
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")   #打印相关信息
    return model  #返回训练好的模型
