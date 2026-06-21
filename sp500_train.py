import torch
import torch.nn as nn
import numpy as np
import os
import signal
import sys
from sklearn.preprocessing import MinMaxScaler

# ===================== 教学配置 =====================
# 可选值: 'RNN', 'GRU', 'LSTM'
MODEL_TYPE = 'LSTM'
# 时序滑窗长度：用过去N个分钟收盘价预测下1分钟收盘价
LOOKBACK = 20
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.1  # 正则化：防止拟合金融噪声
TOTAL_EPOCHS = 300
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-5  # 权重衰减：抑制过拟合
TEST_RATIO = 0.2  # 测试集比例
EARLY_STOP_PATIENCE = 30  # 早停：验证损失不下降则停止

# 动态生成文件名
CHECKPOINT_PATH = f'sp500_{MODEL_TYPE.lower()}_checkpoint.pth'
FINAL_WEIGHTS_PATH = f'sp500_{MODEL_TYPE.lower()}_weights.pth'
SCALER_PATH = f'sp500_scaler.npy'  # 保存归一化参数，推理端必须使用

# ===================== 1. 信号捕获与断点续训（保留原机制） =====================
def receive_signal(signum, frame):
    print(f"\n[警告] 收到资源回收信号 (Signal: {signum})! 正在紧急保存进度...")
    global model, optimizer, epoch, train_loss, val_loss
    save_checkpoint(epoch, model, optimizer, train_loss, val_loss, path=CHECKPOINT_PATH)
    print(f"[退出] {MODEL_TYPE} 模型进度已安全保存，程序优雅退出。")
    sys.exit(0)

signal.signal(signal.SIGTERM, receive_signal)
signal.signal(signal.SIGINT, receive_signal)

def save_checkpoint(epoch, model, optimizer, train_loss, val_loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    torch.save(checkpoint, path)
    print(f"-> 检查点已保存至: {path}")

# ===================== 2. 核心模型：三模型统一实现 =====================
class Sp500Predictor(nn.Module):
    def __init__(self, cell_type='LSTM', input_size=1, hidden_size=32, 
                 num_layers=1, output_size=1, dropout=0.0):
        super(Sp500Predictor, self).__init__()
        self.cell_type = cell_type.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if self.cell_type == 'RNN':
            self.rnn_core = nn.RNN(input_size, hidden_size, num_layers,
                                   batch_first=True, dropout=dropout if num_layers>1 else 0)
        elif self.cell_type == 'GRU':
            self.rnn_core = nn.GRU(input_size, hidden_size, num_layers,
                                   batch_first=True, dropout=dropout if num_layers>1 else 0)
        elif self.cell_type == 'LSTM':
            self.rnn_core = nn.LSTM(input_size, hidden_size, num_layers,
                                    batch_first=True, dropout=dropout if num_layers>1 else 0)
        else:
            raise ValueError("未知网络类型！请选择 'RNN', 'GRU' 或 'LSTM'")
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        if self.cell_type == 'LSTM':
            # LSTM特有细胞状态c0
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            out, (hn, cn) = self.rnn_core(x, (h0, c0))
        else:
            # RNN/GRU仅需隐状态h0
            out, hn = self.rnn_core(x, h0)
        
        out = self.dropout(out)
        # 取序列最后一个时刻的输出做预测
        return self.fc(out[:, -1, :])

# ===================== 3. 金融数据生成与预处理 =====================
def generate_sp500_mock_data(n_minutes=5000, seed=42):
    """
    生成模拟标普500分钟线：几何布朗运动+趋势+噪声，还原金融数据
    低信噪比、随机游走特性，可替换为真实CSV数据
    """
    np.random.seed(seed)
    # 基础漂移与波动率（对标标普500分钟级特性）
    mu = 0.00002
    sigma = 0.0015
    dt = 1
    
    # 几何布朗运动生成价格序列
    returns = np.random.normal(mu*dt, sigma*np.sqrt(dt), n_minutes)
    price = 100 * np.exp(np.cumsum(returns))
    # 叠加低频趋势与高频噪声，进一步降低信噪比
    trend = 0.00008 * np.arange(n_minutes)
    noise = np.random.normal(0, 0.0005, n_minutes)
    price = price * (1 + trend) + noise
    
    return price.reshape(-1, 1)

def create_sliding_windows(data, lookback):
    """滑窗构造时序样本：[样本数, 序列长度, 特征数]"""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def load_and_process_data(lookback, test_ratio):
    # 可替换为：pd.read_csv("SP500_minute.csv")['close'].values.reshape(-1,1)
    raw_data = generate_sp500_mock_data()
    
    # 归一化：金融数据必须做缩放，避免梯度爆炸
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(raw_data)
    
    # 保存scaler参数，推理端必须复用
    np.save(SCALER_PATH, [scaler.data_min_, scaler.data_max_])
    
    # 滑窗构造样本
    X, y = create_sliding_windows(scaled_data, lookback)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # 时序数据必须按时间划分，禁止随机打乱
    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test, scaler

# ===================== 4. 主训练逻辑 =====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"======== 量化实验：当前训练【{MODEL_TYPE}】 标普500分钟线预测模型 ========")
    print(f"运行设备: {device}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 加载数据
    X_train, X_test, y_train, y_test, scaler = load_and_process_data(LOOKBACK, TEST_RATIO)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
    print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

    # 实例化模型
    model = Sp500Predictor(
        cell_type=MODEL_TYPE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    # 打印参数量（对应问题7）
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    train_loss = torch.tensor(0.0)
    val_loss = torch.tensor(0.0)

    # 断点续训
    if os.path.exists(CHECKPOINT_PATH):
        print(f"发现【{MODEL_TYPE}】历史训练记录，正在恢复...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"成功恢复！从第 {start_epoch} 个 Epoch 继续。")

    try:
        for epoch in range(start_epoch, TOTAL_EPOCHS):
            # 训练阶段
            model.train()
            outputs = model(X_train)
            train_loss = criterion(outputs, y_train)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # 验证阶段
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_loss = criterion(val_outputs, y_test)

            # 日志与检查点
            if (epoch + 1) % 10 == 0:
                print(f'[{MODEL_TYPE}] Epoch [{epoch+1}/{TOTAL_EPOCHS}] '
                      f'Train Loss: {train_loss.item():.6f} | Val Loss: {val_loss.item():.6f}')
                save_checkpoint(epoch, model, optimizer, train_loss, val_loss, CHECKPOINT_PATH)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 早停机制：防止拟合训练集噪声
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                torch.save(model.state_dict(), FINAL_WEIGHTS_PATH)
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"\n[早停] 验证损失连续{EARLY_STOP_PATIENCE}轮未下降，停止训练。")
                    print(f"最优验证损失: {best_val_loss:.6f}，最佳权重已保存至 {FINAL_WEIGHTS_PATH}")
                    break

        else:
            print(f"\n[完成] {MODEL_TYPE} 模型训练完成！")
            torch.save(model.state_dict(), FINAL_WEIGHTS_PATH)
            print(f"部署权重已保存至: {FINAL_WEIGHTS_PATH}")

        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)

    except Exception as e:
        print(f"训练发生意外: {e}")
        save_checkpoint(epoch, model, optimizer, train_loss, val_loss, CHECKPOINT_PATH)