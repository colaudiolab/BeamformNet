import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.gen_data import get_arrangement
from models.IQResNet import IQResNet, IQResNet_train
from models.daMUSIC import DeepAugmentMusic, daMUSIC_train
from models.daMUSIC_source_num_estimator import DaMUSICSourceNumEstimator, daMUSICSourceNumEstimator_train
from models.deepSFNS import DeepSFNS, deepSFNS_train
from models.deepSFNS_source_num_estimator import DeepSFNSSourceNumEstimator, deepSFNSSourceNumEstimator_train
from models.deepSSE import DeepSSE, deepSSE_train
from models.lowSNRNet import DOALowSNRNet, lowSNRNet_train
from models.source_num_estimator import SourceNumEstimator, sourceNumEstimator_train
from models.subspaceNet import SubspaceNet, subspaceNet_train
from utils.dataset import DOADataset, RealMANDataset
from utils.loss import AsymmetricLoss, RMSPELoss
from utils.util import create_worker_seed_val



def train_main(gen_data_config, train_config,model_config,folder_path,RealMAN_train_list = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Run training...")

    if gen_data_config['ar_tp'] == 'ULA' or gen_data_config['ar_tp'] == 'RealMAN':
        mod = torch.pi
    else:
        mod = 2 * torch.pi

    arrangement = get_arrangement(gen_data_config)

    data_num = gen_data_config['da_n']
    train_datanum = int(data_num*0.9)

    is_cov = False
    is_Rx_tou = False
    is_eigenvalue = False
    if train_config['model'] == 'deepSSE' or train_config['model'] == 'lowSNRNet':
        is_cov = True
    if train_config['model'] == 'subspaceNet':
        is_Rx_tou = True
    if train_config['model'] == 'SourceNumEstimator':
        is_eigenvalue = True

    train_dataset = DOADataset(arrangement = arrangement, num_samples = train_datanum, gen_data_config=gen_data_config,is_cov=is_cov,is_Rx_tou=is_Rx_tou,is_eigenvalue=is_eigenvalue)
    val_dataset = DOADataset(arrangement = arrangement, num_samples = data_num - train_datanum , gen_data_config=gen_data_config,is_cov=is_cov,is_Rx_tou=is_Rx_tou,is_eigenvalue=is_eigenvalue)

    if RealMAN_train_list is not None:
        train_dataset = RealMANDataset(data_list = RealMAN_train_list[:train_datanum],arrangement=arrangement, gen_data_config=gen_data_config,
                                   is_cov=is_cov, is_Rx_tou=is_Rx_tou)
        val_dataset = RealMANDataset(data_list = RealMAN_train_list[train_datanum:],arrangement=arrangement,
                                 gen_data_config=gen_data_config, is_cov=is_cov, is_Rx_tou=is_Rx_tou)


    batch_size = train_config['bs']
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4,worker_init_fn=worker_seed_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,num_workers=4,worker_init_fn=create_worker_seed_val(20251024))

    num_epochs = train_config['n_eps']
    writer = SummaryWriter(log_dir=folder_path / 'logs')

    #加载loss
    if train_config['loss'] == 'ASL':
        criterion = AsymmetricLoss()
    elif train_config['loss'] == 'RMSPE':
        criterion = RMSPELoss(mod,gen_data_config['deg_l'],gen_data_config['deg_u'],gen_data_config['deg_p'])
    elif train_config['loss'] == 'BCE':
        criterion = nn.BCELoss()
    elif train_config['loss'] == 'Softmax':
        criterion = nn.CrossEntropyLoss()
    else:
        return -1

    num_workers = 4
    early_stop = 20
    step_size = 100
    gamma = 0.5

    if train_config['model'] == 'deepSFNS':
        model = DeepSFNS(gen_data_config, model_config, train_config).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")

        optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        deepSFNS_train(num_epochs,model,optimizer,train_dataset,val_loader, criterion, device,writer,folder_path,batch_size,num_workers,scheduler,early_stop)
    elif train_config['model'] == 'deepSSE':
        deepsse_config = {
            "steering_vetor": train_dataset.steeringMatrix_A,
            "num_class": len(list(np.arange(gen_data_config['deg_l'], gen_data_config['deg_u'], gen_data_config['deg_p']))),
            "num_antenna": gen_data_config['m'],
            "antenna_spacing": (gen_data_config['c'] / gen_data_config['f']) / 2,
            "img_channels": 3,
            "in_channels": 32,
            "out_channels": 32,
            "layers": [2, 2],
            "activation": "gelu",
            "d_model": 128,
            "nhead": 8,
            "num_ca_layers": 2,
            "dim_feedforward": 512,
            "dropout": 0.05
        }

        return_code = -1
        while return_code == -1:
            model = DeepSSE(**deepsse_config).to(device)
            optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            return_code = deepSSE_train(num_epochs,model,optimizer,train_dataset,val_loader, criterion, device,writer,folder_path,batch_size,num_workers,scheduler,early_stop)
    elif train_config['model'] == 'daMUSIC':
        model = DeepAugmentMusic(gen_data_config['m'], gen_data_config['c'] / (2 * gen_data_config['f']), gen_data_config['deg_l'], gen_data_config['deg_u'], gen_data_config['deg_p']).to(device)

        # total_params = sum(p.numel() for p in model.parameters())
        # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #
        # print(f"总参数量: {total_params:,}")
        # print(f"可训练参数量: {trainable_params:,}")

        optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        daMUSIC_train(num_epochs,model,optimizer,train_dataset,val_loader, criterion, device,writer,folder_path,batch_size,num_workers,scheduler,early_stop)
    elif train_config['model'] == 'subspaceNet':
        model = SubspaceNet(tau = 8).to(device)
        optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        subspaceNet_train(num_epochs,model,optimizer,train_dataset,val_loader, criterion, device,writer,folder_path,batch_size,num_workers,scheduler,early_stop)
    elif train_config['model'] == 'lowSNRNet':
        model = DOALowSNRNet(len(list(np.arange(gen_data_config['deg_l'], gen_data_config['deg_u'], gen_data_config['deg_p'])))).to(device)
        optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        lowSNRNet_train(num_epochs,model,optimizer,train_dataset,val_loader, criterion, device,writer,folder_path,batch_size,num_workers,scheduler,early_stop)
    elif train_config['model'] == 'IQResNet':
        model = IQResNet(len(list(np.arange(gen_data_config['deg_l'], gen_data_config['deg_u'], gen_data_config['deg_p']))),gen_data_config['m']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        IQResNet_train(num_epochs,model,optimizer,train_dataset,val_loader, criterion, device,writer,folder_path,batch_size,num_workers,scheduler,early_stop)
    elif train_config['model'] == 'SourceNumEstimator':
        model = SourceNumEstimator(gen_data_config['m']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        sourceNumEstimator_train(num_epochs,model,optimizer,train_dataset,val_loader, criterion, device,writer,folder_path,batch_size,num_workers,scheduler,early_stop)
    elif train_config['model'] == 'DaMUSICSourceNumEstimator':
        model = DaMUSICSourceNumEstimator(gen_data_config,device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        daMUSICSourceNumEstimator_train(num_epochs,model,optimizer,train_dataset,val_loader, criterion, device,writer,folder_path,batch_size,num_workers,scheduler,early_stop)
    elif train_config['model'] == 'DeepSFNSSourceNumEstimator':
        model = DeepSFNSSourceNumEstimator(gen_data_config, model_config, train_config,device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        deepSFNSSourceNumEstimator_train(num_epochs,model,optimizer,train_dataset,val_loader, criterion, device,writer,folder_path,batch_size,num_workers,scheduler,early_stop)
    elif train_config['model'] == 'MUSIC' or train_config['model'] == 'B_MUSIC':
        print()

    # save_rng_state(folder_path / 'rng_state.pkl',rng)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Script')
    parser.add_argument('--yaml_config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    yaml_config_path = Path(args.yaml_config_path)

    with open(yaml_config_path, 'r') as file:
        train_config = yaml.safe_load(file)  # 推荐使用 safe_load 避免安全风险

    train_main(train_config,None,None)






