import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import RealMANDataset
from utils.util import create_worker_seed_train


def find_roots_torch(coefficients: torch.Tensor):
    """Finds the roots of a polynomial defined by its coefficients.
    equivalent to src.utils.find_roots, but support Pytorch.

    Args:
        coefficients (torch.Tensor): List of polynomial coefficients in descending order of powers.

    Returns:
        torch.Tensor: An array containing the roots of the polynomial.

    Raises:
        None

    Examples:
        >>> coefficients = torch.tensor([1, -5, 6])  # x^2 - 5x + 6
        >>> find_roots(coefficients)
        tensor([3., 2.])

    """
    A = torch.diag(torch.ones(len(coefficients) - 2, dtype=coefficients.dtype), -1)
    A[0, :] = -coefficients[1:] / coefficients[0]
    roots = torch.linalg.eigvals(A)
    return roots


def sum_of_diags_torch(matrix: torch.Tensor):
    """Calculates the sum of diagonals in a square matrix.
    equivalent sum_of_diag, but support Pytorch.

    Args:
        matrix (torch.Tensor): Square matrix for which diagonals need to be summed.

    Returns:
        torch.Tensor: A list containing the sums of all diagonals in the matrix, from left to right.

    Raises:
        None

    Examples:
        >>> matrix = torch.tensor([[1, 2, 3],
                                    [4, 5, 6],
                                    [7, 8, 9]])
        >>> sum_of_diag(matrix)
            torch.tensor([7, 12, 15, 8, 3])
    """
    diag_sum = []
    diag_index = torch.linspace(
        -matrix.shape[0] + 1, matrix.shape[0] - 1, 2 * matrix.shape[0] - 1, dtype=int
    )
    for idx in diag_index:
        diag_sum.append(torch.sum(torch.diagonal(matrix, idx)))
    return torch.stack(diag_sum, dim=0)


def gram_diagonal_overload(Kx: torch.Tensor, eps: float, batch_size: int):
    """Multiply a matrix Kx with its Hermitian conjecture (gram matrix),
        and adds eps to the diagonal values of the matrix,
        ensuring a Hermitian and PSD (Positive Semi-Definite) matrix.

    Args:
    -----
        Kx (torch.Tensor): Complex matrix with shape [BS, N, N],
            where BS is the batch size and N is the matrix size.
        eps (float): Constant multiplier added to each diagonal element.
        batch_size(int): The number of batches

    Returns:
    --------
        torch.Tensor: Hermitian and PSD matrix with shape [BS, N, N].

    """
    # Insuring Tensor input
    if not isinstance(Kx, torch.Tensor):
        Kx = torch.tensor(Kx)

    Kx_list = []
    bs_kx = Kx
    for iter in range(batch_size):
        K = bs_kx[iter]
        # Hermitian conjecture
        Kx_garm = torch.matmul(torch.t(torch.conj(K)), K).to(Kx.device)
        # Diagonal loading
        eps_addition = (eps * torch.diag(torch.ones(Kx_garm.shape[0]))).to(Kx.device)
        Rz = Kx_garm + eps_addition
        Kx_list.append(Rz)
    Kx_Out = torch.stack(Kx_list, dim=0)
    return Kx_Out


def root_music(Rz: torch.Tensor, labels, batch_size: int):
    """Implementation of the model-based Root-MUSIC algorithm, support Pytorch, intended for
        MB-DL models. the model sets for nominal and ideal condition (Narrow-band, ULA, non-coherent)
        as it accepts the surrogate covariance matrix.
        it is equivalent tosrc.methods: RootMUSIC.narrowband() method.

    Args:
    -----
        Rz (torch.Tensor): Focused covariance matrix
        M (int): Number of sources
        batch_size: the number of batches

    Returns:
    --------
        doa_batches (torch.Tensor): The predicted doa, over all batches.
        doa_all_batches (torch.Tensor): All doa predicted, given all roots, over all batches.
        roots_to_return (torch.Tensor): The unsorted roots.
    """

    dist = 0.5
    f = 1
    doa_batches = []
    doa_all_batches = []
    Bs_Rz = Rz
    for iter in range(batch_size):
        M = len(np.where(labels[iter] == 1)[0])
        R = Bs_Rz[iter]
        # Extract eigenvalues and eigenvectors using EVD
        eigenvalues, eigenvectors = torch.linalg.eig(R)
        # Assign noise subspace as the eigenvectors associated with M greatest eigenvalues
        Un = eigenvectors[:, torch.argsort(torch.abs(eigenvalues)).flip(0)][:, M:]
        # Generate hermitian noise subspace matrix
        F = torch.matmul(Un, torch.t(torch.conj(Un)))
        # Calculates the sum of F matrix diagonals
        diag_sum = sum_of_diags_torch(F)
        # Calculates the roots of the polynomial defined by F matrix diagonals
        roots = find_roots_torch(diag_sum)
        # Calculate the phase component of the roots
        roots_angels_all = torch.angle(roots)
        # Calculate doa
        doa_pred_all = torch.arcsin((1 / (2 * np.pi * dist * f)) * roots_angels_all)
        doa_all_batches.append(doa_pred_all)
        roots_to_return = roots
        # Take only roots which inside the unit circle
        roots = roots[
            sorted(range(roots.shape[0]), key=lambda k: abs(abs(roots[k]) - 1))
        ]
        mask = (torch.abs(roots) - 1) < 0
        roots = roots[mask][:M]
        # Calculate the phase component of the roots
        roots_angels = torch.angle(roots)
        # Calculate doa
        doa_pred = torch.arcsin((1 / (2 * np.pi * dist * f)) * roots_angels)
        doa_batches.append(doa_pred)

    return (
        doa_batches,
        doa_all_batches,
        roots_to_return,
    )


def esprit(Rz: torch.Tensor, M: int, batch_size: int):
    """Implementation of the model-based Esprit algorithm, support Pytorch, intended for
        MB-DL models. the model sets for nominal and ideal condition (Narrow-band, ULA, non-coherent)
        as it accepts the surrogate covariance matrix.
        it is equivalent to src.methods: RootMUSIC.narrowband() method.

    Args:
    -----
        Rz (torch.Tensor): Focused covariance matrix
        M (int): Number of sources
        batch_size: the number of batches

    Returns:
    --------
        doa_batches (torch.Tensor): The predicted doa, over all batches.
    """

    doa_batches = []

    Bs_Rz = Rz
    for iter in range(batch_size):
        R = Bs_Rz[iter]
        # Extract eigenvalues and eigenvectors using EVD
        eigenvalues, eigenvectors = torch.linalg.eig(R)

        # Get signal subspace
        Us = eigenvectors[:, torch.argsort(torch.abs(eigenvalues)).flip(0)][:, :M]
        # Separate the signal subspace into 2 overlapping subspaces
        Us_upper, Us_lower = (
            Us[0 : R.shape[0] - 1],
            Us[1 : R.shape[0]],
        )
        # Generate Phi matrix
        phi = torch.linalg.pinv(Us_upper) @ Us_lower
        # Find eigenvalues and eigenvectors (EVD) of Phi
        phi_eigenvalues, _ = torch.linalg.eig(phi)
        # Calculate the phase component of the roots
        eigenvalues_angels = torch.angle(phi_eigenvalues)
        # Calculate the DoA out of the phase component
        doa_predictions = -1 * torch.arcsin((1 / np.pi) * eigenvalues_angels)
        doa_batches.append(doa_predictions)

    return torch.stack(doa_batches, dim=0)


class SubspaceNet(nn.Module):
    """SubspaceNet is model-based deep learning model for generalizing DOA estimation problem,
        over subspace methods.

    Attributes:
    -----------
        M (int): Number of sources.
        tau (int): Number of auto-correlation lags.
        conv1 (nn.Conv2d): Convolution layer 1.
        conv2 (nn.Conv2d): Convolution layer 2.
        conv3 (nn.Conv2d): Convolution layer 3.
        deconv1 (nn.ConvTranspose2d): De-convolution layer 1.
        deconv2 (nn.ConvTranspose2d): De-convolution layer 2.
        deconv3 (nn.ConvTranspose2d): De-convolution layer 3.
        DropOut (nn.Dropout): Dropout layer.
        ReLU (nn.ReLU): ReLU activation function.

    Methods:
    --------
        anti_rectifier(X): Applies the anti-rectifier operation to the input tensor.
        forward(Rx_tau): Performs the forward pass of the SubspaceNet.
        gram_diagonal_overload(Kx, eps): Applies Gram operation and diagonal loading to a complex matrix.

    """

    def __init__(self, tau: int, diff_method: str = "root_music"):
        """Initializes the SubspaceNet model.

        Args:
        -----
            tau (int): Number of auto-correlation lags.
            M (int): Number of sources.

        """
        super(SubspaceNet, self).__init__()
        self.tau = tau
        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(64, 16, kernel_size=2)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=2)
        self.DropOut = nn.Dropout(0.2)
        self.ReLU = nn.ReLU()
        # Set the subspace method for training
        self.set_diff_method(diff_method)

    def set_diff_method(self, diff_method: str):
        """Sets the differentiable subspace method for training subspaceNet.
            Options: "root_music", "esprit"

        Args:
        -----
            diff_method (str): differentiable subspace method.

        Raises:
        -------
            Exception: Method diff_method is not defined for SubspaceNet
        """
        if diff_method.startswith("root_music"):
            self.diff_method = root_music
        elif diff_method.startswith("esprit"):
            self.diff_method = esprit
        else:
            raise Exception(
                f"SubspaceNet.set_diff_method: Method {diff_method} is not defined for SubspaceNet"
            )

    def anti_rectifier(self, X):
        """Applies the anti-rectifier operation to the input tensor.

        Args:
        -----
            X (torch.Tensor): Input tensor.

        Returns:
        --------
            torch.Tensor: Output tensor after applying the anti-rectifier operation.

        """
        return torch.cat((self.ReLU(X), self.ReLU(-X)), 1)

    def forward(self, Rx_tau: torch.Tensor, labels):
        """
        Performs the forward pass of the SubspaceNet.

        Args:
        -----
            Rx_tau (torch.Tensor): Input tensor of shape [Batch size, tau, 2N, N].

        Returns:
        --------
            doa_prediction (torch.Tensor): The predicted direction-of-arrival (DOA) for each batch sample.
            doa_all_predictions (torch.Tensor): All DOA predictions for each root, over all batches.
            roots_to_return (torch.Tensor): The unsorted roots.
            Rz (torch.Tensor): Surrogate covariance matrix.

        """
        # Rx_tau shape: [Batch size, tau, 2N, N]
        self.N = Rx_tau.shape[-1]
        self.batch_size = Rx_tau.shape[0]
        ## Architecture flow ##
        # CNN block #1
        x = self.conv1(Rx_tau)
        x = self.anti_rectifier(x)
        # CNN block #2
        x = self.conv2(x)
        x = self.anti_rectifier(x)
        # CNN block #3
        x = self.conv3(x)
        x = self.anti_rectifier(x)
        # DCNN block #1
        x = self.deconv2(x)
        x = self.anti_rectifier(x)
        # DCNN block #2
        x = self.deconv3(x)
        x = self.anti_rectifier(x)
        # DCNN block #3
        x = self.DropOut(x)
        Rx = self.deconv4(x)
        # Reshape Output shape: [Batch size, 2N, N]
        Rx_View = Rx.view(Rx.size(0), Rx.size(2), Rx.size(3))
        # Real and Imaginary Reconstruction
        Rx_real = Rx_View[:, : self.N, :]  # Shape: [Batch size, N, N])
        Rx_imag = Rx_View[:, self.N :, :]  # Shape: [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape: [Batch size, N, N])
        # Apply Gram operation diagonal loading
        Rz = gram_diagonal_overload(
            Kx=Kx_tag, eps=1, batch_size=self.batch_size
        )  # Shape: [Batch size, N, N]
        # Feed surrogate covariance to the differentiable subspace algorithm
        method_output = self.diff_method(Rz, labels, self.batch_size)
        if isinstance(method_output, tuple):
            # Root MUSIC output
            doa_prediction, doa_all_predictions, roots = method_output
        else:
            # Esprit output
            doa_prediction = method_output
            doa_all_predictions, roots = None, None
        return doa_prediction, doa_all_predictions, roots, Rz


def subspaceNet_train(num_epochs,model,optimizer,train_dataset,val_loader, criterion, device,writer,folder_path,batch_size,num_workers,scheduler,early_stop):
    # 训练循环
    best_RMSPE = 9999.99
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
            signal = batch['Rx_tou'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs,_,_,_ = model(signal,batch['label'])

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
                signal = batch['Rx_tou'].to(device)
                labels = batch['label'].to(device)

                outputs,_,_,_ = model(signal,batch['label'])
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
                not_improve_epoch = 0
            else:
                not_improve_epoch += 1
                if not_improve_epoch >= early_stop:
                    print('Early stopped')
                    break

            print(f'Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val RMSPE={val_RMSPE:.6f}')


def subspaceNet_val(model, val_loader, device,gen_data_config):
    model.eval()
    doas_list = []
    label_list = []
    doas_known_d_list = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            signal = batch['Rx_tou'].to(device)
            labels = batch['label'].numpy()

            outputs_doa_known_d, outputs_doa,_,_ = model(signal,batch['label'])

            for index in range(batch['label'].shape[0]):
                label = (np.where(labels[index] == 1)[0] * gen_data_config['deg_p'] - gen_data_config['deg_u']) * np.pi / 180
                doa_known_d = outputs_doa_known_d[index].detach().cpu().numpy()
                doas_list.append(outputs_doa[index])
                doas_known_d_list.append(doa_known_d)
                label_list.append(label)

    return doas_list, label_list, doas_known_d_list