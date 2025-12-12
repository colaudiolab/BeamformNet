import copy
import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import RealMANDataset
from utils.util import cal_cov, create_worker_seed_train


class PositionEncoding2D(nn.Module):
    def __init__(
        self,
        num_pos_feats=64,
        temperature=10000,
        normalize=False,
        scale=None,
        maxh=16,
        maxw=16,
    ):
        """This is a more standard version of the position embedding, very
        similar to the one used by the Attention is all you need paper,
        generalized to work on images.

        demision of output: (batch_size, num_pos_feats * 2, maxH, maxW)
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.maxh = maxh
        self.maxw = maxw
        pe = self._gen_pos_buffer()
        self.register_buffer("pe", pe)

    def _gen_pos_buffer(self):
        _eyes = torch.ones((1, self.maxh, self.maxw))
        y_embed = _eyes.cumsum(1, dtype=torch.float32)
        x_embed = _eyes.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(self, inp):
        """Generate positional encoding without added to original embedding."""
        x = inp
        return self.pe.repeat((x.size(0), 1, 1, 1))


def _get_activation(activation: str = "relu"):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "prelu":
        return nn.PReLU()
    raise ValueError(f"Activation {activation} not supported")


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module | None = None,
        activation: str = "relu",
    ):
        """Residual Block

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride of the first convolutional layer
            expansion: multiplicative factor for the subsequent conv2d layer's
                output
            downsample: downsample layer
        """
        super().__init__()

        # Multiplicative factor for the subsequent conv2d layer's output
        # channels. It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        # 第一层卷积
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = _get_activation(activation)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class BaseModel(ABC, nn.Module):
    """Base class for all models"""

    @abstractmethod
    def forward(self, x):
        """Forward pass logic

        Args:
            model_input: model input

        Returns:
            model output
        """
        raise NotImplementedError()

    def __str__(self):
        """Model prints with number of trainable parameters"""
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())

        return (
            super().__str__()
            + "\nParameters: {}".format(
                sum([np.prod(p.size()) for p in trainable_params])
            )
            + "\nTrainable parameters: {}".format(
                sum([np.prod(p.size()) for p in self.parameters()])
            )
        )




class BaseModule(ABC, nn.Module):
    def __str__(self):
        """Model prints with number of trainable parameters"""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    @classmethod
    def build_model(cls, **kwargs):
        params = {
            k: v
            for k, v in kwargs.items()
            if k in cls.__init__.__code__.co_varnames
        }
        return cls(**params)


class SpatialFeatureExtractor(BaseModule):
    def __init__(
        self,
        img_channels,
        layers=[2, 2, 2, 2],
        in_channels=32,
        out_channels=32,
        backbone_activation="relu",
    ):
        """ResNet like architecture for spatial feature extraction."""
        super().__init__()

        # layers = [2, 2, 2, 2] is resnet18
        self.expansion = 1

        self.activation = backbone_activation

        self.in_channels = in_channels
        self.out_channels = out_channels

        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.input_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=img_channels,
                out_channels=self.in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(self.in_channels),
            _get_activation(self.activation),
        )

        self.res_layers = nn.ModuleList()
        for num_layer in layers:
            self.res_layers.append(
                self._make_layer(self.out_channels, num_layer)
            )

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(
            ResidualBlock(
                self.in_channels,
                out_channels,
                stride=stride,
                activation=self.activation,
            )
        )

        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(
                ResidualBlock(
                    self.in_channels, out_channels, activation=self.activation
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layers(x)

        for res_layer in self.res_layers:
            x = res_layer(x)

        return x


class AngleFeatureProjector(BaseModule):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.angle_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        return self.angle_proj(x)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.functional.relu
    if activation == "gelu":
        return nn.functional.gelu
    if activation == "glu":
        return nn.functional.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class CALayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ):
        """Cross attention layer, attention + FFN like Transformer's decoder."""
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        query,
        feature,
        pos,
    ):
        # -- cross-attention ---------------------------------------------------
        tgt1, _ = self.multihead_attn(
            query=query,
            key=feature + pos,
            value=feature,
        )

        # -- FFN ---------------------------------------------------------------
        tgt = query + self.dropout1(tgt1)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        return tgt


class CABlock(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        query,
        feature,
        pos: Optional[Tensor] = None,
    ):
        output = query

        for layer in self.layers:
            output = layer(
                output,
                feature,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class AngularGridSearch(BaseModule):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_ca_layers=1,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()

        ca_layer = CALayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

        norm = nn.LayerNorm(d_model)
        self.ca_block = CABlock(ca_layer, num_ca_layers, norm)

        self.d_model = d_model

    def forward(self, feature, query, pos_embed):
        # feature: (HW, bs, d)
        # query: (num_class, bs, d)
        # pos_embed: (HW, bs, d)

        out = self.ca_block(
            query,  # (num_clas, bs, d)
            feature,  # feature from SFE, (hw, bs, d)
            pos=pos_embed,  # pos encoding of feature
        )

        # hs: (bs, K, d)
        return out.transpose(0, 1)


class FeatureWiseLinear(nn.Module):
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))

    def forward(self, x):
        # x: (bs, num_class, d)
        # self.W: (1, num_class, d)
        x = (self.W * x).sum(-1)
        # point-wise mul and sum in the last dim, x will be (bs, num_class)

        if self.bias:
            x = x + self.b
            # broadcast when adding bias, x will be (bs, num_class)

        return x


class TransForm(BaseModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.input_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.flatten(2).permute(2, 0, 1)

        return x


class DeepSSE(BaseModel):
    def __init__(self, steering_vetor, num_class, num_antenna, antenna_spacing, **kwargs):
        """Deep Learning based Spatial Spectrum Estimator."""
        super().__init__()
        self.sfe = SpatialFeatureExtractor.build_model(**kwargs)

        self.ags = AngularGridSearch.build_model(**kwargs)

        self.pos_embed = PositionEncoding2D(
            num_pos_feats=self.ags.d_model // 2,
            maxh=num_antenna,
            maxw=num_antenna,
        )

        hidden_dim = self.ags.d_model

        self.transform = TransForm(self.sfe.out_channels, hidden_dim)

        self.angle_projector = AngleFeatureProjector(
            3 * num_antenna, hidden_dim
        )

        self.fc = FeatureWiseLinear(num_class, hidden_dim, bias=True)

        self._get_steering_vectors(steering_vetor)

    def _get_steering_vectors(self, steering_vetor):
        steering_vetor = torch.from_numpy(steering_vetor)
        # grids = torch.linspace(-90, 90 - 180 / num_class, num_class)
        # antenna_position = (
        #     (torch.arange(0, num_antenna, 1) * antenna_spacing)
        #     .view(-1, 1)
        #     .to(torch.float)
        # )
        # delay = antenna_position @ torch.sin(grids).view(1, -1)
        #
        # steering_vetor = torch.exp(-2j * math.pi * delay)
        steering_vetor = torch.cat(
            (steering_vetor.real, steering_vetor.imag, steering_vetor.angle()),
            dim=0,
        )

        steering_vetor = steering_vetor.float()

        self.register_buffer("angle_embed", steering_vetor.transpose(0, 1))

    def forward(self, x):
        # inp: (bs, C, H_0, W_0)
        bs, _, _, _ = x.shape

        spatial_feature = self.sfe(x)  # (bs, d_0, H, W)

        angle_embed = self.angle_projector(self.angle_embed)  # (num_class, d)
        angle_embed = angle_embed.unsqueeze(1).repeat(
            1, bs, 1
        )  # (num_class, bs, d)

        pos_embed = self.pos_embed(
            spatial_feature
        )  # pos_embed of spatial feature: (bs, 2*d//2, H, W)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # (HW, bs, d)

        out = self.ags(
            feature=self.transform(spatial_feature),
            query=angle_embed,
            pos_embed=pos_embed,
        )

        out = self.fc(out)

        return torch.sigmoid(out)


def deepSSE_train(num_epochs,model,optimizer,train_dataset,val_loader, criterion, device,writer,folder_path,batch_size,num_workers,scheduler,early_stop):
    # 训练循环
    best_F1 = 0.0
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
            loss = criterion(outputs, labels)

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
                loss = criterion(outputs, labels)

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


def deepSSE_val(model, val_loader, device, validate_config,gen_data_config):
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

                # plt.figure(figsize=(12, 6))
                # plt.plot(probability, label='result_deepSSE_energy', color='blue')
                # thetas = np.where(labels[index] == 1)[0]
                # plt.scatter(thetas, np.ones(thetas.shape[0]) * max(probability), color='green')
                # plt.title('deepSSE_energy')
                # plt.xlabel('degree')
                # plt.ylabel('energy')
                # plt.grid(True)
                # plt.show()

                # 过滤空间噪声 即低于阈值0.5认为不存在目标
                doas = np.where(probability > validate_config['td'])[0]
                doas_list.append((doas * gen_data_config['deg_p'] - gen_data_config['deg_u']) * np.pi / 180)
                label_list.append((np.where(labels[index] == 1)[0] * gen_data_config['deg_p'] - gen_data_config['deg_u']) * np.pi / 180)

                # top d 个
                d = len(np.where(labels[index] == 1)[0])
                doas = np.argsort(probability)[-d:]
                doas_known_d_list.append((doas * gen_data_config['deg_p'] - gen_data_config['deg_u']) * np.pi / 180)


    return doas_list, label_list, doas_known_d_list