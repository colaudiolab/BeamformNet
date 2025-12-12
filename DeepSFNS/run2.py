import argparse
import os
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

seed = 12
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

import yaml
from train import train_main
from validate import val_main




# rng = np.random.default_rng(seed=42)  #默认使用 PCG64生成器2^128次才会序列重复

def main(root_path, gen_data_config, train_config, model_config,validate_config,only_run_validate,RealMAN_data_path = None):
    # 创建文件夹
    folder_name = ""
    for key, value in gen_data_config.items():
        # 添加键值对到文件名
        folder_name += f"{key}_{value}_"
    for key, value in train_config.items():
        # 添加键值对到文件名
        folder_name += f"{key}_{value}_"

    # for key, value in model_config.items():
    #     # 添加键值对到文件名
    #     folder_name += f"{key}_{value}_"
    # for key, value in validate_config.items():
    #     # 添加键值对到文件名
    #     folder_name += f"{key}_{value}_"

    folder_path = root_path / folder_name.rstrip("_")
    os.makedirs(folder_path,exist_ok=True)


    # 将各个config保存为yaml文件备份
    with open(folder_path / "gen_data_config.yaml", "w", encoding="utf-8") as file:
        yaml.dump(
            gen_data_config,
            file,
            allow_unicode=True,  # 支持中文
            default_flow_style=False,  # 使用块样式（更易读）
            indent=4,  # 缩进4个空格
            sort_keys=False  # 保持键的原始顺序
        )

    with open(folder_path / "train_config.yaml", "w", encoding="utf-8") as file:
        yaml.dump(
            train_config,
            file,
            allow_unicode=True,  # 支持中文
            default_flow_style=False,  # 使用块样式（更易读）
            indent=4,  # 缩进4个空格
            sort_keys=False  # 保持键的原始顺序
        )

    with open(folder_path / "model_config.yaml", "w", encoding="utf-8") as file:
        yaml.dump(
            model_config,
            file,
            allow_unicode=True,  # 支持中文
            default_flow_style=False,  # 使用块样式（更易读）
            indent=4,  # 缩进4个空格
            sort_keys=False  # 保持键的原始顺序
        )

    with open(folder_path / "validate_config.yaml", "w", encoding="utf-8") as file:
        yaml.dump(
            validate_config,
            file,
            allow_unicode=True,  # 支持中文
            default_flow_style=False,  # 使用块样式（更易读）
            indent=4,  # 缩进4个空格
            sort_keys=False  # 保持键的原始顺序
        )

    RealMAN_train_list = None
    RealMAN_test_list = None
    if RealMAN_data_path is not None:
        print('Loading RealMAN data...')
        RealMAN_data_path = Path(RealMAN_data_path)
        RealMAN_file_list = [str(file.resolve()) for file in RealMAN_data_path.rglob("*.npy")]
        RealMAN_train_list, RealMAN_test_list = train_test_split(
            RealMAN_file_list,
            test_size=0.1,  # 测试集比例 10%
            train_size=0.9,  # 训练集比例 90%
            random_state=42,  # 随机种子
            shuffle=True  # 是否打乱数据
        )

    if only_run_validate == False:
        # 训练模型 并保存best
        train_main(gen_data_config, train_config, model_config, folder_path,RealMAN_train_list)

    #验证模型
    val_main(gen_data_config, train_config,model_config,validate_config,folder_path,RealMAN_test_list)



    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Script')
    parser.add_argument('--root_path', type=str, help='Path for the root folder')
    parser.add_argument('--gen_data_config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--train_config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--model_config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--validate_config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--only_run_validate', action='store_true', default=False)
    parser.add_argument('--RealMAN_data_path', default=None, type=str, help='Path to the RealMAN dataset folder')
    args = parser.parse_args()

    root_path = Path(args.root_path)
    gen_data_config_path = Path(args.gen_data_config_path)
    train_config_path = Path(args.train_config_path)
    model_config_path = Path(args.model_config_path)
    validate_config_path = Path(args.validate_config_path)
    only_run_validate = args.only_run_validate
    RealMAN_data_path = args.RealMAN_data_path

    with open(gen_data_config_path, 'r') as file:
        gen_data_config = yaml.safe_load(file)

    with open(train_config_path, 'r') as file:
        train_config = yaml.safe_load(file)

    with open(model_config_path, 'r') as file:
        model_config = yaml.safe_load(file)

    with open(validate_config_path, 'r') as file:
        validate_config = yaml.safe_load(file)

    main(root_path, gen_data_config, train_config, model_config,validate_config,only_run_validate,RealMAN_data_path)