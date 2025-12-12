import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.deepSSE import BaseModel
from utils.dataset import RealMANDataset
from utils.util import create_worker_seed_train


class DOALowSNRNet(BaseModel):
    def __init__(self, num_out_grids=180):
        """CNN based DOA estimation network

        Args:
            num_out_grids (int): number of angle girds used

        References:
            Papageorgiou, Georgios K., Mathini Sellathurai, and Yonina C. Eldar.
            “Deep Networks for Direction-of-Arrival Estimation in Low SNR.”
            IEEE Transactions on Signal Processing 69 (2021): 3714-29.
            https://doi.org/10.1109/TSP.2021.3089927.
        """
        super().__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=256,
                kernel_size=(3, 3),
                stride=2,
                padding=0,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(2, 2),
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(2, 2),
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(2, 2),
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(nn.LazyLinear(out_features=4096), nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=2048), nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024), nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=num_out_grids),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x


def lowSNRNet_train(num_epochs,model,optimizer,train_dataset,val_loader, criterion, device,writer,folder_path,batch_size,num_workers,scheduler,early_stop):
    # 训练循环
    best_F1 = -1.0
    not_improve_epoch = 0
    if isinstance(train_dataset, RealMANDataset):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    for epoch in range(num_epochs):
        if not isinstance(train_dataset, RealMANDataset):
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers,worker_init_fn=create_worker_seed_train(torch.initial_seed()+epoch * num_workers))

        model.train()
        train_loss = 0.0

        # 训练阶段
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch in pbar:
            cov = batch['cov'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(cov)
            loss = criterion(outputs, labels.float())

            if torch.isnan(loss).item():
                return -1

            loss.backward()
            optimizer.step()


            train_loss += loss.item() * labels.size(0)
            pbar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})

        scheduler.step()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_correct_positive = 0
        val_total_positive = 0
        val_correct_pre = 0
        val_total_pre = 0

        with torch.no_grad():
            for batch in tqdm(val_loader):
                cov = batch['cov'].to(device)
                labels = batch['label'].to(device)

                outputs = model(cov)
                loss = criterion(outputs, labels.float())

                val_loss += loss.item() * labels.size(0)

                # 计算准确率
                probabilities = outputs
                predictions = (probabilities > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0) * labels.size(1)

                # 计算召回率
                positive_mask = (labels == 1)  # 布尔掩码，标记正样本位置
                if positive_mask.any():  # 如果存在正样本
                    # 提取正样本的预测结果和标签
                    positive_predictions = predictions[positive_mask]
                    positive_labels = labels[positive_mask]

                    # 统计正确数
                    val_correct_positive += (positive_predictions == positive_labels).sum().item()
                    # 统计真实正样本总数
                    val_total_positive += positive_labels.size(0)

                # 计算精确率
                positive_mask = (predictions == 1)
                if positive_mask.any():
                    positive_predictions = predictions[positive_mask]
                    positive_labels = labels[positive_mask]
                    # 统计正确数
                    val_correct_pre += (positive_predictions == positive_labels).sum().item()
                    val_total_pre += positive_predictions.size(0)

            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)
            val_acc = correct / total
            val_recall = val_correct_positive / val_total_positive if val_total_positive > 0 else 0.0
            val_precision = val_correct_pre / val_total_pre if val_total_pre > 0 else 0.0
            val_F1 = 2 * val_precision * val_recall / (val_precision + val_recall) if (
                                                                                                  val_precision + val_recall) > 0 else 0.0

            writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('Recall/val', val_recall, epoch)
            writer.add_scalar('Precision/val', val_precision, epoch)
            writer.add_scalar('F1/val', val_F1, epoch)

            # 保存最佳模型
            if val_F1 > best_F1:
                best_F1 = val_F1
                torch.save(model.state_dict(), folder_path / 'best_model.pth')
                not_improve_epoch = 0
            else:
                not_improve_epoch += 1
                if not_improve_epoch >= early_stop:
                    print('Early stopped')
                    break

            print(
                f'Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2%}, Val Recall={val_recall:.2%}, Val Precision={val_precision:.2%}, Val F1={val_F1:.2%}')
    return 0


def lowSNRNet_val(model, val_loader, device, validate_config,gen_data_config):
    model.eval()
    doas_list = []
    label_list = []
    doas_known_d_list = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            cov = batch['cov'].to(device)
            labels = batch['label'].numpy()

            outputs = model(cov).detach().cpu().numpy()

            for index in range(batch['label'].shape[0]):
                probability = outputs[index]

                plt.figure(figsize=(12, 6))
                plt.plot(probability, label='result_lowSNRNet_energy', color='blue')
                thetas = np.where(labels[index] == 1)[0]
                plt.scatter(thetas, np.ones(thetas.shape[0]) * max(probability), color='green')
                plt.title('lowSNRNet_energy')
                plt.xlabel('degree')
                plt.ylabel('energy')
                plt.grid(True)
                plt.show()

                # 过滤空间噪声 即低于阈值0.5认为不存在目标
                doas = np.where(probability > validate_config['td'])[0]
                doas_list.append((doas * gen_data_config['deg_p'] - gen_data_config['deg_u']) * np.pi / 180)
                label_list.append((np.where(labels[index] == 1)[0] * gen_data_config['deg_p'] - gen_data_config['deg_u']) * np.pi / 180)

                # top d 个
                d = len(np.where(labels[index] == 1)[0])
                doas = np.argsort(probability)[-d:]
                doas_known_d_list.append((doas * gen_data_config['deg_p'] - gen_data_config['deg_u']) * np.pi / 180)


    return doas_list, label_list, doas_known_d_list