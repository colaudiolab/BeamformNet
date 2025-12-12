import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.signal import find_peaks

from utils.dataset import RealMANDataset
from utils.metrics import RMSPE_padding_zeros
from utils.util import create_worker_seed_train, create_worker_seed_val


class RNNBlock(nn.Module):
    def __init__(self,in_dim,mid_dim,out_dim,is_bidirectional = False):
        #in_dim 2*m
        #mid_dim n
        #out_dim 2*m
        super().__init__()
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.bn = nn.BatchNorm1d(self.in_dim)
        self.gru = nn.GRU(input_size = self.in_dim, hidden_size = self.out_dim,num_layers = 2, batch_first=True,bidirectional=is_bidirectional)

    def forward(self,x):
        # x (batch,2*m,n)

        x = self.bn(x) # out x (batch,2*m,n)
        x = x.permute(0, 2, 1) # out x (batch,n,2*m)
        x = self.gru(x)[0] # out x (batch,n,2*m)
        x = x.permute(0, 2, 1) #out x (batch,2*m,n)

        return x


class DeepSFNS(nn.Module):
    def __init__(self,gen_data_config, model_config, train_config):
        #in_dim 2*m
        #mid_dim n
        #out_dim 2*m
        super().__init__()
        self.gen_data_config = gen_data_config
        self.model_config = model_config
        self.train_config = train_config
        self.m = gen_data_config['m']
        degree_lower = gen_data_config['deg_l']
        degree_upper = gen_data_config['deg_u']
        degree_precision = gen_data_config['deg_p']
        self.n = len(list(np.arange(degree_lower, degree_upper, degree_precision)))
        self.snapshots = gen_data_config['spshots']
        self.k = model_config['em_d']
        self.is_bidirectional = model_config['is_bid']
        self.Ann1 = RNNBlock(2*self.m, self.n, self.k,self.is_bidirectional)
        self.Ynn2 = RNNBlock(2*self.m, self.snapshots, self.k,self.is_bidirectional)
        if self.is_bidirectional:
            self.k = 2 * self.k
        self.Bnn3 = RNNBlock(self.k, self.n, self.k)
        self.fc = nn.Sequential(
            nn.Linear(self.n, 2*self.n),
            nn.BatchNorm1d(2*self.n),
            nn.ReLU(),
            nn.Linear(2*self.n, self.n),
        )
        self.W_v = nn.Linear(self.n, self.n)
        self.W_k = nn.Linear(self.n, self.n)
        self.W_q = nn.Linear(self.snapshots, self.n)
        self.W_b = nn.Linear(self.k, 2 * self.m)
        self.scale = self.n ** 0.5

    def forward(self,A,Y):
        # A (batch,m,n) complex
        # Y (batch,m,snapshots) complex
        A_real = A.real
        A_imag = A.imag
        A = torch.cat((A_real, A_imag), dim=1) #out A (batch,2*m,n)

        Y_real = Y.real
        Y_imag = Y.imag
        Y = torch.cat((Y_real, Y_imag), dim=1)#out Y (batch,2*m,snapshots)



        A = A.float()
        Y = Y.float()

        A = self.Ann1(A) #out A (batch,k,n)
        Y = self.Ynn2(Y) #out Y (batch,k,snapshots)

        A_v = self.W_v(A) #out A_v (batch,k,n)
        A_k = self.W_k(A) #out A_k (batch,k,n)
        Y_q = self.W_q(Y) #out Y_q (batch,k,n)

        # 计算注意力分数
        scores = torch.matmul(Y_q, A_k.transpose(-2, -1)) / self.scale #out scores (batch,k,k)
        attn_weights = torch.softmax(scores, dim=-1)

        # 加权求和
        B = torch.matmul(attn_weights, A_v)  # out B (batch_size, k, n)

        B = self.Bnn3(B) #out B (batch_size, k, n)
        B = B.permute(0, 2, 1) #out B (batch_size, n, k)
        B = self.W_b(B) #out B (batch_size, n, 2 * m)

        B_real = B[:,:,:self.m]
        B_imag = B[:,:,self.m:]

        Y_origin = torch.complex(Y_real.float(), Y_imag.float())
        B = torch.complex(B_real, B_imag)

        P = B @ Y_origin # out P (batch_size, n, snapshots)
        P = torch.abs(P)**2
        P = torch.mean(P, dim=2)#out P (batch_size, n)

        # P = self.fc(P)  # out P (batch_size, n)

        P = torch.tanh(P)

        return P


def deepSFNS_train(num_epochs,model,optimizer,train_dataset,val_loader, criterion, device,writer,folder_path,batch_size,num_workers,scheduler,early_stop):
    # 训练循环
    best_F1 = -1.0
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
            labels = batch['label'].to(device)
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
        val_correct_positive = 0
        val_total_positive = 0
        val_correct_pre = 0
        val_total_pre = 0

        with torch.no_grad():
            for batch in tqdm(val_loader):
                signal = batch['signal'].to(device)
                labels = batch['label'].to(device)
                steeringMatrix_A = batch['steeringMatrix_A'].to(device)

                outputs = model(steeringMatrix_A, signal)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)

                # 计算准确率
                # probabilities = F.softmax(outputs, dim=1)
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

def deepSFNS_val(model, val_loader, device, validate_config,gen_data_config):
    model.eval()
    doas_list = []
    label_list = []
    doas_known_d_list = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            signal = batch['signal'].to(device)
            labels = batch['label'].numpy()
            steeringMatrix_A = batch['steeringMatrix_A'].to(device)

            outputs = model(steeringMatrix_A, signal)

            spectrums = torch.atanh(torch.clamp(outputs, -1+1e-6, 1-1e-6)).detach().cpu().numpy() #反atanh得到空间能量谱
            outputs = outputs.detach().cpu().numpy()
            for index in range(batch['label'].shape[0]):
                d = len(np.where(labels[index] == 1)[0])
                spectrum = spectrums[index]
                spectrum_copy = np.copy(spectrum)

                #过滤空间噪声 即低于阈值0.5认为不存在目标
                spectrum[np.where(outputs[index] <= validate_config['td'])[0]] = 0

                plt.figure(figsize=(12, 6))
                plt.plot(spectrum, label='result_deepSFNS_energy', color='blue')
                thetas = np.where(labels[index] == 1)[0]
                plt.scatter(thetas, np.ones(thetas.shape[0]) * max(spectrum), color='green')
                plt.title('deepSFNS_energy')
                plt.xlabel('degree')
                plt.ylabel('energy')
                plt.grid(True)
                plt.show()


                # 滤波寻找峰值
                doas_peak = find_peaks(spectrum)[0]
                doas_list.append((doas_peak * gen_data_config['deg_p'] - gen_data_config['deg_u']) * np.pi / 180)
                label_list.append((np.where(labels[index] == 1)[0] * gen_data_config['deg_p'] - gen_data_config['deg_u']) * np.pi / 180)

                #先从滤波得到的角度找DOA，多出的截断（优先取高能量），少的用top d个填充
                doas_peak = doas_peak[np.argsort(spectrum[doas_peak])][::-1]
                doas_great = np.argsort(spectrum_copy)[::-1]
                doas_known_d = np.concatenate([doas_peak, doas_great[~np.isin(doas_great, doas_peak)]])[:d]
                doas_known_d_list.append((doas_known_d * gen_data_config['deg_p'] - gen_data_config['deg_u']) * np.pi / 180)


    return doas_list, label_list, doas_known_d_list
