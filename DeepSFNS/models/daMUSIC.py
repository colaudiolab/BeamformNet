import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.deepSSE import BaseModel
from utils.dataset import RealMANDataset
from utils.util import create_worker_seed_train


class DeepAugmentMusic(BaseModel):
    def __init__(self, num_antennas, antenna_spacing, degree_lower, degree_upper, grid):
        """
        Reference:
            Merkofer, Julian P., Guy Revach, Nir Shlezinger, Tirza Routtenberg,
            and Ruud J. G. van Sloun. “DA-MUSIC: Data-Driven DoA Estimation via
            Deep Augmented MUSIC Algorithm.” arXiv, January 11, 2023.
            https://doi.org/10.48550/arXiv.2109.10581.
        """
        super().__init__()

        self._num_antennas = num_antennas
        self._angle_grids = torch.arange(degree_lower, degree_upper, grid)

        # ━━ 1. get covariance matrix ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.norm = nn.BatchNorm1d(2 * num_antennas)  # norm in feature
        self.gru = nn.GRU(
            input_size=2 * num_antennas, hidden_size=2 * num_antennas
        )
        self.linear = nn.Linear(
            in_features=2 * num_antennas,
            out_features=2 * num_antennas * num_antennas,
        )

        # ━━ 2. get the weight of eigenvectors ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.eig_vec_pro = nn.Sequential(
            nn.Linear(
                in_features=2 * num_antennas, out_features=2 * num_antennas
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=2 * num_antennas, out_features=2 * num_antennas
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=2 * num_antennas, out_features=2 * num_antennas
            ),
            nn.ReLU(),
            nn.Linear(in_features=2 * num_antennas, out_features=num_antennas),
            nn.Sigmoid(),
        )

        # ━━ 3. peak finder ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.peak_finder = nn.Sequential(
            nn.Linear(
                in_features=self._angle_grids.shape[0],
                out_features=2 * num_antennas,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=2 * num_antennas, out_features=2 * num_antennas
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=2 * num_antennas, out_features=2 * num_antennas
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=2 * num_antennas, out_features=num_antennas - 1
            ),
        )

        self._get_steering_vectors(num_antennas, antenna_spacing)

    def _get_steering_vectors(self, num_antenna, antenna_spacing):
        antenna_position = (
            (torch.arange(0, num_antenna, 1) * antenna_spacing)
            .view(-1, 1)
            .to(torch.float)
        )
        delay = antenna_position @ torch.sin(self._angle_grids).view(1, -1)

        steering_vectors = torch.exp(-2j * np.pi * delay)

        self.register_buffer("steering_vectors", steering_vectors)

    def _get_cov(self, x):
        # x: (batch_size, 2 * num_antennas, num_snapshots)
        # `2 * num_antennas` is the number of feature
        x = self.norm(x)  # norm in feature
        x = torch.permute(
            x, (2, 0, 1)
        )  # (num_snapshots, batch_size, 2 * num_antennas)
        _, x = self.gru(x)  # (batch_size, 1, 2 * num_antennas)
        x = self.linear(x)  # (batch_size, 1, 2 * num_antennas * num_antennas)
        x = x.reshape(-1, 2 * self._num_antennas, self._num_antennas)

        x = torch.complex(
            x[:, : self._num_antennas, :], x[:, self._num_antennas :, :]
        )

        return x

    def _get_noise_space(self, eig_val, eig_vec):
        prob = self.eig_vec_pro(torch.cat((eig_val.real, eig_val.imag), dim=1))
        prob = torch.diag_embed(prob)

        # NOTE: the `eig_vec` is parted into real and imaginary parts to prevent
        #       the gradient from being calculated on the phase of the complex
        # see this for reason: https://pytorch.org/docs/stable/generated/torch.svd.html
        noise_space = torch.complex(
            torch.bmm(prob, eig_vec.real), torch.bmm(prob, eig_vec.imag)
        )

        return noise_space

    def _cal_spectrum(self, noise_space):
        device = noise_space.device
        v = noise_space.transpose(1, 2).conj() @ self.steering_vectors.to(
            device
        )

        spectrum = 1 / torch.linalg.norm(v, axis=1) ** 2

        return spectrum.to(torch.float32)

    def get_eig_val(self, x):
        cov = self._get_cov(x)
        eig_val, _ = torch.linalg.eig(cov)

        return eig_val

    def forward(self, x):
        x_real = x.real
        x_imag = x.imag
        x = torch.cat((x_real, x_imag),dim=1).float()

        cov = self._get_cov(x)

        eig_val, eig_vec = torch.linalg.eig(cov)

        noise_space = self._get_noise_space(eig_val, eig_vec)

        spectrum = self._cal_spectrum(noise_space)

        estimates = self.peak_finder(spectrum)

        return estimates


def daMUSIC_train(num_epochs,model,optimizer,train_dataset,val_loader, criterion, device,writer,folder_path,batch_size,num_workers,scheduler,early_stop):
    # 训练循环
    best_RMSPE = 9999.99
    not_improv_epoch = 0
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

            optimizer.zero_grad()
            outputs = model(signal)

            loss = criterion(outputs, labels)
            try:
                loss.backward()
            except RuntimeError:
                print("linalg error")
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            pbar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})

        scheduler.step()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader):
                signal = batch['signal'].to(device)
                labels = batch['label'].to(device)

                outputs = model(signal)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)



            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)
            val_RMSPE = val_loss

            writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            writer.add_scalar('RMSPE/val', val_RMSPE, epoch)

            # 保存最佳模型
            if val_RMSPE < best_RMSPE:
                best_RMSPE = val_RMSPE
                torch.save(model.state_dict(), folder_path / 'best_model.pth')
                not_improv_epoch = 0
            else:
                not_improv_epoch += 1
                if not_improv_epoch >= early_stop:
                    print('Early stopped')
                    break

            print(f'Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val RMSPE={val_RMSPE:.6f}')


def daMUSIC_val(model, val_loader, device,gen_data_config):
    model.eval()
    doas_list = []
    label_list = []
    doas_known_d_list = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            signal = batch['signal'].to(device)
            labels = batch['label'].numpy()

            outputs = model(signal)

            for index in range(batch['label'].shape[0]):
                label = (np.where(labels[index] == 1)[0] * gen_data_config['deg_p'] - gen_data_config['deg_u']) * np.pi / 180
                d = label.shape[0]
                doa = outputs[index][:d].detach().cpu().numpy()
                doas_list.append(doa)
                doas_known_d_list.append(doa)
                label_list.append(label)

    return doas_list, label_list, doas_known_d_list