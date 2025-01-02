# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from dataset import NBA_Dataset
from model import Net
from tqdm import tqdm

window = 7
workers = 0
test_dataset = NBA_Dataset("val_data", "column_mean_and_std.csv", window=window)
print("資料集大小:", len(test_dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(test_dataset[0][0].shape[0]).to(device)
model.load_state_dict(torch.load("modelacc.pth",weights_only=True))
print("輸入大小:", test_dataset[0][0].shape[0])
print("模型大小:", sum(p.numel() for p in model.parameters() if p.requires_grad))


if __name__ == '__main__':
    win_probabilitys_times = [0] * 10
    win_probabilitys_count = [0] * 10
    test_acc = 0
    model.eval()
    with torch.no_grad():
        for data, label in tqdm(test_dataset):
            data, label = data.to(device).unsqueeze(0), label.to(device).unsqueeze(0)
            win_probability = model(data)
            is_right = ((win_probability > 0.5).int() == (label > 0.5).int()).sum().item()

            win_probabilitys_times[int(win_probability*10)]+=1
            win_probabilitys_count[int(win_probability*10)]+=is_right

            test_acc += is_right

    test_acc /= len(test_dataset)
    print(f"測試集正確率為 {test_acc:.4f}")


for i in range(10):
    temp = win_probabilitys_count[i]/(win_probabilitys_times[i]+1e-10)
    print(f"{i*10}% ~ {(i+1)*10}% 的正確率為 {temp:.4f} 場數為 {win_probabilitys_times[i]}")