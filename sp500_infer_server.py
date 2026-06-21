import torch
import torch.nn as nn
import numpy as np
import asyncio
import json
import time
import random

# ===================== 配置 =====================
MODEL_TYPE = 'LSTM'
WEIGHTS_PATH = f'sp500_{MODEL_TYPE.lower()}_weights.pth'
SCALER_PATH = 'sp500_scaler.npy'
LOOKBACK = 20
HOST = "127.0.0.1"
PORT = 8765

# ===================== 模型结构（与训练端完全一致） =====================
class Sp500Predictor(nn.Module):
    def __init__(self, cell_type='LSTM', input_size=1, hidden_size=32, 
                 num_layers=1, output_size=1, dropout=0.0):
        super(Sp500Predictor, self).__init__()
        self.cell_type = cell_type.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if self.cell_type == 'RNN':
            self.rnn_core = nn.RNN(input_size, hidden_size, num_layers,
                                   batch_first=True)
        elif self.cell_type == 'GRU':
            self.rnn_core = nn.GRU(input_size, hidden_size, num_layers,
                                   batch_first=True)
        elif self.cell_type == 'LSTM':
            self.rnn_core = nn.LSTM(input_size, hidden_size, num_layers,
                                    batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        if self.cell_type == 'LSTM':
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            out, (hn, cn) = self.rnn_core(x, (h0, c0))
        else:
            out, hn = self.rnn_core(x, h0)
        
        return self.fc(out[:, -1, :])

# ===================== 生成测试集流式数据（修复标量问题） =====================
def generate_test_stream():
    # 生成与训练端同源的测试数据
    np.random.seed(42)
    n_total = 5000
    mu = 0.00002
    sigma = 0.0015
    returns = np.random.normal(mu, sigma*np.sqrt(1), n_total)
    price = 100 * np.exp(np.cumsum(returns))
    trend = 0.00008 * np.arange(n_total)
    noise = np.random.normal(0, 0.0005, n_total)
    price = price * (1 + trend) + noise
    price = price.reshape(-1, 1)

    # 加载训练集保存的极值，提取为标量（修复核心）
    scaler_params = np.load(SCALER_PATH)
    data_min = scaler_params[0][0]  # 从一维数组中取出单个数值
    data_max = scaler_params[1][0]  # 从一维数组中取出单个数值
    scaled_data = (price - data_min) / (data_max - data_min)

    # 取后20%作为测试集
    split = int(len(scaled_data) * 0.8)
    test_data = scaled_data[split:]
    # 返回归一化后的数据 + 极值标量用于反归一化
    return test_data, data_min, data_max

# ===================== WebSocket 流式推理 =====================
async def stream_data(websocket):
    print(f"\n[终端接入] 客户端已连接。开始下发 {MODEL_TYPE} 行情预测数据...")
    
    # 加载模型
    model = Sp500Predictor(cell_type=MODEL_TYPE)
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
        model.eval()
        print(f"[系统] 成功加载权重文件: {WEIGHTS_PATH}")
    except FileNotFoundError:
        print(f"[警告] 未找到 {WEIGHTS_PATH}，将使用初始权重演示。")

    test_data, data_min, data_max = generate_test_stream()
    window_buffer = list(test_data[:LOOKBACK, 0])

    try:
        with torch.no_grad():
            for i in range(LOOKBACK, len(test_data) - 1):
                start_time = time.time()

                # 构造输入窗口
                input_seq = torch.tensor(window_buffer, dtype=torch.float32).view(1, LOOKBACK, 1)
                pred_scaled = model(input_seq).item()
                actual_scaled = test_data[i+1, 0]

                # 手动反归一化为真实价格（全标量计算，无数组维度问题）
                pred_price = pred_scaled * (data_max - data_min) + data_min
                actual_price = actual_scaled * (data_max - data_min) + data_min

                # 滑动窗口更新
                window_buffer.pop(0)
                window_buffer.append(actual_scaled)

                # 模拟计算+网络延迟
                calc_time = (time.time() - start_time) * 1000
                simulated_latency = calc_time + random.uniform(2.0, 8.0)

                payload = {
                    "timestamp": time.time() * 1000,
                    "model_type": MODEL_TYPE,
                    "ch1_actual": float(actual_price),
                    "ch2_predict": float(pred_price),
                    "error_abs": abs(float(actual_price) - float(pred_price)),
                    "latency_ms": round(simulated_latency, 2)
                }
                await websocket.send(json.dumps(payload))
                await asyncio.sleep(0.05)

    except Exception as e:
        print(f"[断开] 客户端连接中断或发生异常: {e}")

async def main():
    import websockets
    async with websockets.serve(stream_data, HOST, PORT):
        print("===============================================")
        print(f" [SYS] 量化行情预测边缘终端已启动")
        print(f" [SYS] 当前预测模型: {MODEL_TYPE} 神经网络")
        print(f" [SYS] 监听端口: ws://{HOST}:{PORT}")
        print("===============================================")
        print("等待前端监控面板接入...")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())