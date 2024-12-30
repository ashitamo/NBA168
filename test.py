# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from dataset import NBA_Dataset
from model import Net
from tqdm import tqdm

window = 11
workers = 0
BATCH_SIZE = 32
test_dataset = NBA_Dataset("test_data", "column_mean_and_std.csv", window=window)
print("資料集大小:", len(test_dataset))

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(test_dataset[0][0].shape[0]).to(device)
model.load_state_dict(torch.load("modelloss.pth",weights_only=True))
print("輸入大小:", test_dataset[0][0].shape[0])
print("模型大小:", sum(p.numel() for p in model.parameters() if p.requires_grad))


if __name__ == '__main__':


    model.eval()
    criterion = nn.BCELoss()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for data, label in tqdm(test_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            test_loss += loss.item()
            test_acc += ((output > 0.5).int() == (label > 0.5).int()).sum().item()

    print(f"Test Loss: {test_loss / len(test_loader)}")
    print(f"Test Accuracy: {test_acc / len(test_dataset)}")


