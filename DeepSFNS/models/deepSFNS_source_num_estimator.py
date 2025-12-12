import numpy as np
import torch
from pycparser.ply.ctokens import t_OREQUAL
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.daMUSIC import DeepAugmentMusic
from models.deepSFNS import DeepSFNS
from utils.dataset import RealMANDataset
from utils.util import create_worker_seed_train, cal_eigenvalue


class DeepSFNSSourceNumEstimator(nn.Module):
    def __init__(self,gen_data_config,model_config,train_config,device):
        super().__init__()
        self.M = gen_data_config['m']
        self.SourceNumEstimator = nn.Sequential(
            nn.Linear(
                in_features=len(list(np.arange(gen_data_config['deg_l'], gen_data_config['deg_u'], gen_data_config['deg_p']))),
                out_features=2 * self.M * self.M,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=2 * self.M * self.M, out_features=2 * self.M * self.M
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=2 * self.M * self.M, out_features=2 * self.M * self.M
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=2 * self.M * self.M, out_features=self.M - 1
            ),
        )
        self.deepSFNS_pretrain = DeepSFNS(gen_data_config, model_config, train_config).to(device)
        self.deepSFNS_pretrain.load_state_dict(torch.load('./best_model.pth', map_location=device))
        self.deepSFNS_pretrain.eval()


    def forward(self, A, Y):

        with torch.no_grad():
            P = self.deepSFNS_pretrain(A, Y)

        return self.SourceNumEstimator(P)

def deepSFNSSourceNumEstimator_train(num_epochs,model,optimizer,train_dataset,val_loader, criterion, device,writer,folder_path,batch_size,num_workers,scheduler,early_stop):
    # 训练循环
    best_accuracy = -1.0
    not_improve_epoch  = 0
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
            signal = batch['signal'].to(device)
            labels = (torch.sum(batch['label'] == 1,dim=1)-1).to(device)
            steeringMatrix_A = batch['steeringMatrix_A'].to(device)

            optimizer.zero_grad()
            outputs = model(steeringMatrix_A, signal)
            loss = criterion(outputs, labels)
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
        with torch.no_grad():
            for batch in tqdm(val_loader):
                signal = batch['signal'].to(device)
                labels = (torch.sum(batch['label'] == 1, dim=1) - 1).to(device)
                steeringMatrix_A = batch['steeringMatrix_A'].to(device)

                outputs = model(steeringMatrix_A, signal)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)

                # 计算准确率
                _, predicted = torch.max(outputs, 1)  # 获取预测类别
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)
            val_accuracy = 100. * correct / total

            writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)

            # 保存最佳模型
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), folder_path / 'best_model.pth')
                not_improve_epoch = 0
            else:
                not_improve_epoch += 1
                if not_improve_epoch >= early_stop:
                    print('Early stopped')
                    break

            print(
                f'Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val accuracy={val_accuracy:.4f}')


def deepSFNSSourceNumEstimator_val(model, val_loader, device, validate_config,gen_data_config):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            signal = batch['signal'].to(device)
            labels = (torch.sum(batch['label'] == 1, dim=1) - 1).to(device)
            steeringMatrix_A = batch['steeringMatrix_A'].to(device)

            outputs = model(steeringMatrix_A, signal)

            # 计算准确率
            _, predicted = torch.max(outputs, 1)  # 获取预测类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_accuracy = 100. * correct / total

    return val_accuracy
