import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import RealMANDataset
from utils.util import create_worker_seed_train


class SourceNumEstimator(nn.Module):
    def __init__(self,M):
        super().__init__()
        self.M = M
        self.SourceNumEstimator = nn.Sequential(
            nn.Linear(
                in_features=self.M,
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

    def forward(self, X):
        return self.SourceNumEstimator(X)


def sourceNumEstimator_train(num_epochs,model,optimizer,train_dataset,val_loader, criterion, device,writer,folder_path,batch_size,num_workers,scheduler,early_stop):
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
            signal = batch['eigenvalue'].to(device)
            labels = (torch.sum(batch['label'] == 1,dim=1)-1).to(device)

            optimizer.zero_grad()
            outputs = model(signal)
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
                signal = batch['eigenvalue'].to(device)
                labels = (torch.sum(batch['label'] == 1, dim=1) - 1).to(device)

                outputs = model(signal)
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


def sourceNumEstimator_val(model, val_loader, device, validate_config,gen_data_config):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            signal = batch['eigenvalue'].to(device)
            labels = (torch.sum(batch['label'] == 1, dim=1) - 1).to(device)

            outputs = model(signal)

            # 计算准确率
            _, predicted = torch.max(outputs, 1)  # 获取预测类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_accuracy = 100. * correct / total

    return val_accuracy
