"""Training script
This is the training script for superpoint detector and descriptor.

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

#启动模型训练流程：main函数解析命令行->执行train_joint/train_base
import argparse
import yaml
import os
import logging

import torch
import torch.optim
import torch.utils.data

from tensorboardX import SummaryWriter

# from utils.utils import tensor2array, save_checkpoint, load_checkpoint, save_path_formatter
from utils.utils import getWriterPath
from settings import EXPER_PATH

## loaders: data, model, pretrained model
from utils.loader import dataLoader, modelLoader, pretrainedLoader
from utils.logging import *
# from models.model_wrap import SuperPointFrontend_torch, PointTracker

###### util functions ######
# 打印数据集大小信息
def datasize(train_loader, config, tag='train'):
    logging.info('== %s split size %d in %d batches'%\
    (tag, len(train_loader)*config['model']['batch_size'], len(train_loader)))
    pass

from utils.loader import get_save_path

###### util functions end ######


###### train script ######
# 基础训练函数，直接调用联合训练函数
#这是训练 MagicPoint（关键点检测器）时使用的入口
def train_base(config, output_dir, args):
    return train_joint(config, output_dir, args)
    pass

# def train_joint_dsac():
#     pass

# 联合训练函数，最主要的训练函数
def train_joint(config, output_dir, args):
    assert 'train_iter' in config  # 确保配置中包含训练迭代次数

    # 。配置初始化
    torch.set_default_tensor_type(torch.FloatTensor)  # 设置默认张量类型
    task = config['data']['dataset']  # 获取数据集名称
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置训练设备
    logging.info('train on device: %s', device)
    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)  # 保存配置到文件到configyml
    
    #日志和路径设置
    # 初始化TensorBoard日志记录器，用于在训练期间记录和可视化损失、指标等
    writer = SummaryWriter(getWriterPath(task=args.command, 
        exper_name=args.exper_name, date=True))
    
    # 获取保存路径
    save_path = get_save_path(output_dir)

    # ！！！数据加载
    data = dataLoader(config, dataset=task, warp_input=True)  # 加载训练和验证数据
    train_loader, val_loader = data['train_loader'], data['val_loader']

    # 打印数据集大小信息
    datasize(train_loader, config, tag='train')
    datasize(val_loader, config, tag='val')
    

    # ！！！初始化训练代理
    from utils.loader import get_module
    train_model_frontend = get_module('', config['front_end_model'])  # 动态加载前端模型：加载一个包含 SuperPoint 模型及其所有训练逻辑的类
    train_agent = train_model_frontend(config, save_path=save_path, device=device) #实例化训练核心/实例化训练代理

    # 设置TensorBoard日志记录器
    train_agent.writer = writer

    # 将数据加载到训练代理中
    train_agent.train_loader = train_loader
    train_agent.val_loader = val_loader

    # 加载模型并初始化，应该是加载预训练模型
    train_agent.loadModel()  # 加载模型
    train_agent.dataParallel()  # 启用数据并行

    try:
        # 开始训练
        train_agent.train()
    except KeyboardInterrupt:
        print ("press ctrl + c, save model!")  # 捕获中断信号并保存模型
        train_agent.saveModel()
        pass

if __name__ == '__main__':
    # 全局变量初始化,init log
    torch.set_default_tensor_type(torch.FloatTensor)
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    # 命令行参数解析
    parser = argparse.ArgumentParser() #参数解析器初始化
    subparsers = parser.add_subparsers(dest='command')  # 创建子命令解析器

    # 基础训练命令
    p_train = subparsers.add_parser('train_base')
    p_train.add_argument('config', type=str)  # 配置文件路径
    p_train.add_argument('exper_name', type=str)  # 实验名称
    p_train.add_argument('--eval', action='store_true')  # 是否启用评估模式
    p_train.add_argument('--debug', action='store_true', default=False,
                         help='turn on debuging mode')  # 是否启用调试模式
    p_train.set_defaults(func=train_base)  # 设置默认函数

    # 联合训练命令
    p_train = subparsers.add_parser('train_joint')
    p_train.add_argument('config', type=str)  # 配置文件路径
    p_train.add_argument('exper_name', type=str)  # 实验名称
    p_train.add_argument('--eval', action='store_true')  # 是否启用评估模式
    p_train.add_argument('--debug', action='store_true', default=False,
                         help='turn on debuging mode')  # 是否启用调试模式
    p_train.set_defaults(func=train_joint)  # 设置默认函数

    args = parser.parse_args()  # 解析命令行参数

    if args.debug:
        logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG)  # 调试模式日志配置

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)  ############ 加载配置文件
    
    ##### 创建输出目录
    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    os.makedirs(output_dir, exist_ok=True)

    logging.info('Running command {}'.format(args.command.upper()))  # 打印运行命令
    args.func(config, output_dir, args)  # 执行训练函数


