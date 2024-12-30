# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from dataset import NBA_Dataset
from model import Net
from tqdm import tqdm

window = 7
workers = 8
EPOCH = 50
BATCH_SIZE = 512
LR = 0.001

train_dataset = NBA_Dataset("train_data", "column_mean_and_std.csv", window=window)
val_dataset = NBA_Dataset("val_data", "column_mean_and_std.csv", window=window)
print("資料集大小:", len(train_dataset), len(val_dataset))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(train_dataset[0][0].shape[0]).to(device)
# model.load_state_dict(torch.load("modelloss.pth"))
print("輸入大小:", train_dataset[0][0].shape[0])
print("模型大小:", sum(p.numel() for p in model.parameters() if p.requires_grad))

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR,weight_decay=5e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=LR,weight_decay=5e-3,momentum=0.9)

if __name__ == '__main__':

    maxacc = 0
    minloss = 1000
    for epoch in range(EPOCH):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        model.train()
        for data, label in tqdm(train_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            train_loss += loss.item()
            train_acc += ((output > 0.5).eq(label > 0.5)).sum().item()


        model.eval()
        with torch.no_grad():
            for data, label in tqdm(val_loader):
                data, label = data.to(device), label.to(device)
                output = model(data)
                loss = criterion(output, label)
                val_loss += loss.item()
                val_acc += ((output > 0.5).eq(label > 0.5)).sum().item()

        print(f"Epoch: {epoch}, Train Loss: {train_loss / len(train_loader):.6f}, Train Acc: {train_acc / len(train_dataset):.6f}, Val Loss: {val_loss / len(val_loader):.6f}, Val Acc: {val_acc / len(val_dataset):.6f}")
        if val_acc / len(val_dataset) > maxacc:
            maxacc = val_acc / len(val_dataset)
            print("save model with acc:", maxacc)
            torch.save(model.state_dict(), "modelacc.pth")
        if val_loss / len(val_loader) < minloss:
            minloss = val_loss / len(val_loader)
            print("save model with loss:", minloss)
            torch.save(model.state_dict(), "modelloss.pth")

