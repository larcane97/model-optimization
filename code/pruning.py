# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

'''
NNI example for supported basic pruning algorithms.
In this example, we show the end-to-end pruning process: pre-training -> pruning -> fine-tuning.
Note that pruners use masks to simulate the real pruning. In order to obtain a real compressed model, model speed up is required.
You can also try auto_pruners_torch.py to see the usage of some automatic pruning algorithms.
'''
import logging
from src.model import Model

from src.dataloader import create_dataloader
from src.utils.common import read_yaml
import argparse
import os
import sys
import torch
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler

from nni.compression.pytorch.utils.counter import count_flops_params

import nni
from nni.compression.pytorch import ModelSpeedup
from nni.algorithms.compression.v2.pytorch.pruning import SlimPruner
from nni.algorithms.compression.pytorch.pruning import (
    LevelPruner,
    # SlimPruner,
    FPGMPruner,
    TaylorFOWeightFilterPruner,
    L1FilterPruner,
    L2FilterPruner,
    AGPPruner,
    ActivationMeanRankFilterPruner,
    ActivationAPoZRankFilterPruner
)

_logger = logging.getLogger('pruning process')
_logger.setLevel(logging.INFO)

str2pruner = {
    'level': LevelPruner,
    'l1filter': L1FilterPruner,
    'l2filter': L2FilterPruner,
    'slim': SlimPruner,
    'agp': AGPPruner,
    'fpgm': FPGMPruner,
    'mean_activation': ActivationMeanRankFilterPruner,
    'apoz': ActivationAPoZRankFilterPruner,
    'taylorfo': TaylorFOWeightFilterPruner
}

def get_dummy_input(device):
    dummy_input = torch.randn((16,3,224,224)).to(device)
    return dummy_input


def get_data(data_path):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {
    }
    data_config = read_yaml(data_path)
    train_loader,test_loader,_ = create_dataloader(data_config)
    criterion = torch.nn.CrossEntropyLoss()
    return train_loader, test_loader, criterion

def get_model_optimizer_scheduler(model_path,data_path,device):
    data_config = read_yaml(data_path)
    model_instance = Model(model_path, verbose=True)
    model = model_instance.model.to(device)
    EPS = 1e-8
    BETAS = (0.9, 0.999)
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.05
    MIN_LR = data_config["INIT_LR"]/100
    WARMUP_LR = MIN_LR/10
    WARMUP_EPOCHS = 10
    
    # optimizer = torch.optim.SGD(model_instance.model.parameters(), lr=data_config["INIT_LR"], momentum=0.9)
    optimizer = optim.AdamW(model.parameters(), eps=EPS, betas=BETAS,lr=data_config["INIT_LR"], weight_decay=WEIGHT_DECAY)
    scheduler = CosineLRScheduler(
        optimizer, 
        t_initial=data_config['EPOCHS'], 
        warmup_t=WARMUP_EPOCHS, 
        warmup_lr_init=WARMUP_LR,
        lr_min = MIN_LR)

    return model, optimizer, scheduler

def train(args, model, device, train_loader, criterion, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output.squeeze_()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(args, model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100 * correct / len(test_loader.dataset)

    print('Test Loss: {}  Accuracy: {}%\n'.format(
        test_loss, acc))
    return acc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.experiment_data_dir, exist_ok=True)

    # prepare model and data
    train_loader, test_loader, criterion = get_data(args.data_dir)

    model, optimizer, scheduler = get_model_optimizer_scheduler(args.model_dir, args.data_dir,device)

    dummy_input = get_dummy_input(device)
    flops, params, _ = count_flops_params(model, dummy_input)
    print(f"FLOPs: {flops}, params: {params}")

    print(f'start {args.pruner} pruning...')

    def trainer(model, optimizer, criterion):
        return train(args, model, device, train_loader, criterion, optimizer)

    pruner_cls = str2pruner[args.pruner]

    kw_args = {}
    config_list = [{
        'sparsity': args.sparsity,
        'op_types': ['Conv2d']
    }]

    if args.pruner == 'level':
        config_list = [{
            'sparsity': args.sparsity,
            'op_types': ['default']
        }]

    else:
        if args.global_sort:
            print('Enable the global_sort mode')
            # only taylor pruner supports global sort mode currently
            kw_args['global_sort'] = True
        if args.dependency_aware:
            dummy_input = get_dummy_input(device)
            print('Enable the dependency_aware mode')
            # note that, not all pruners support the dependency_aware mode
            kw_args['dependency_aware'] = True
            kw_args['dummy_input'] = dummy_input
        if args.pruner not in ('l1filter', 'l2filter', 'fpgm'):
            # set only work for training aware pruners
            kw_args['trainer'] = trainer
            kw_args['optimizer'] = optimizer
            kw_args['criterion'] = criterion

        if args.pruner in ('mean_activation', 'apoz', 'taylorfo'):
            kw_args['sparsifying_training_batches'] = 1

        if args.pruner == 'slim':
            kw_args['training_epochs'] = 1

        if args.pruner == 'agp':
            kw_args['pruning_algorithm'] = 'l1'
            kw_args['num_iterations'] = 2
            kw_args['epochs_per_iteration'] = 1

        # Reproduced result in paper 'PRUNING FILTERS FOR EFFICIENT CONVNETS',
        # Conv_1, Conv_8, Conv_9, Conv_10, Conv_11, Conv_12 are pruned with 50% sparsity, as 'VGG-16-pruned-A'
        # If you want to skip some layer, you can use 'exclude' like follow.
        if args.pruner == 'slim':
            config_list = [{ 'total_sparsity': 0.5, 'op_types': ['BatchNorm2d'],'max_sparsity_per_layer':0.8}]

            pruner = SlimPruner(model, config_list, trainer, optimizer, criterion, training_epochs=1, scale=0.0001, mode='global')
        elif args.model == 'resnet18':
            config_list = [{
                'sparsity': args.sparsity,
                'op_types': ['Conv2d']
            }, {
                'exclude': True,
                'op_names': ['layer1.0.conv1', 'layer1.0.conv2']
            }]
        elif args.pruner == 'apoz':
            from nni.algorithms.compression.v2.pytorch.pruning import ActivationAPoZRankPruner
            config_list = [{ 'sparsity': 0.8, 'op_types': ['Conv2d'] },{'op_types':['Linear'],'exclude':True}]
            pruner = ActivationAPoZRankPruner(model, config_list, trainer, optimizer, criterion, training_batches=1)

    # pruner = pruner_cls(model, config_list, **kw_args)
    # Pruner.compress() returns the masked model
    _,masks = pruner.compress()
    # print(type(model))
    # torch.save(model,os.path.join(args.experiment_data_dir,'pruned_model'))
    pruner.show_pruned_weights()

    # export the pruned model masks for model speedup
    model_path = os.path.join(args.experiment_data_dir, 'pruned_{}_{}_{}.pth'.format(
        args.model, args.dataset, args.pruner))
    mask_path = os.path.join(args.experiment_data_dir, 'mask_{}_{}_{}.pth'.format(
        args.model, args.dataset, args.pruner))

    pruner.export_model(model_path=model_path, mask_path=mask_path)

    if args.test_only:
        test(args, model, device, criterion, test_loader)
    
    if args.speed_up:
        # Unwrap all modules to normal state
        pruner._unwrap_model()
        model.eval()
        ModelSpeedup(model, dummy_input=torch.rand([64, 3, 224, 224]).to(device), masks_file=masks).speedup_model()
        # m_speedup = ModelSpeedup(model, dummy_input, mask_path, device)
        # m_speedup.speedup_model()

    print('start finetuning...')

    # Optimizer used in the pruner might be patched, so recommend to new an optimizer for fine-tuning stage.
    

    best_top1 = 0
    save_path = os.path.join(args.experiment_data_dir, f'finetuned.pth')
    for epoch in range(args.fine_tune_epochs):
        print('# Epoch {} #'.format(epoch))
        train(args, model, device, train_loader, criterion, optimizer)
        scheduler.step()
        top1 = test(args, model, device, criterion, test_loader)
        if top1 > best_top1:
            best_top1 = top1
            torch.save(model.state_dict(), save_path)

    flops, params, results = count_flops_params(model, dummy_input)
    print(f'Finetuned model FLOPs {flops/1e6:.2f} M, #Params: {params/1e6:.2f}M, Accuracy: {best_top1: .2f}')

    if args.nni:
        nni.report_final_result(best_top1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Example for model comporession')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='taco',
                        help='dataset to use, only taco')
    parser.add_argument('--data_dir', type=str, default='/opt/ml/code/exp/shufflenetV2_0.5_kd_nst-swinL/data.yml',
                        help='dataset directory')
    parser.add_argument('--model_dir', type=str, default='/opt/ml/code/exp/shufflenetV2_0.5_kd_nst-swinL/model.yml',
                        help='model directory')
    parser.add_argument('--model', type=str, default='shufflnetV2_0.5',
                        help='model to use')
    parser.add_argument('--pretrained-model-dir', type=str, default=None,
                        help='path to pretrained model')
    parser.add_argument('--pretrain-epochs', type=int, default=160,
                        help='number of epochs to pretrain the model')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=200,
                        help='input batch size for testing')
    parser.add_argument('--experiment-data-dir', type=str, default='./experiment_data',
                        help='For saving output checkpoints')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--multi-gpu', action='store_true', default=False,
                        help='run on mulitple gpus')
    parser.add_argument('--test-only', action='store_true', default=False,
                        help='run test only')

    # pruner
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='target overall target sparsity')
    parser.add_argument('--dependency-aware', action='store_true', default=False,
                        help='toggle dependency aware mode')
    parser.add_argument('--global-sort', action='store_true', default=False,
                        help='toggle global sort mode')
    parser.add_argument('--pruner', type=str, default='l1filter',
                        choices=['level', 'l1filter', 'l2filter', 'slim', 'agp',
                                 'fpgm', 'mean_activation', 'apoz', 'taylorfo'],
                        help='pruner to use')

    # speed-up
    parser.add_argument('--speed-up', action='store_true', default=False,
                        help='Whether to speed-up the pruned model')

    # fine-tuning
    parser.add_argument('--fine-tune-epochs', type=int, default=160,
                        help='epochs to fine tune')

    parser.add_argument('--nni', action='store_true', default=False,
                        help="whether to tune the pruners using NNi tuners")

    args = parser.parse_args()

    if args.nni:
        params = nni.get_next_parameter()
        print(params)
        args.sparsity = params['sparsity']
        args.pruner = params['pruner']
        args.model = params['model']

    main(args)



# # Copyright (c) Microsoft Corporation.
# # Licensed under the MIT license.

# '''
# NNI example for supported slim pruning algorithms.
# In this example, we show the end-to-end pruning process: pre-training -> pruning -> speedup -> fine-tuning.
# Note that pruners use masks to simulate the real pruning. In order to obtain a real compressed model, model speed up is required.
# '''
# import argparse
# from src.model import Model
# from src.utils.common import read_yaml
# from src.dataloader import create_dataloader
# import torch.optim as optim
# from timm.scheduler.cosine_lr import CosineLRScheduler

# import torch
# from torchvision import datasets, transforms
# from torch.optim.lr_scheduler import MultiStepLR

# from nni.compression.pytorch import ModelSpeedup
# # from examples.model_compress.models.cifar10.vgg import VGG
# from nni.compression.pytorch.utils.counter import count_flops_params
# from nni.algorithms.compression.v2.pytorch.pruning.basic_pruner import SlimPruner

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
# g_epoch = 0

# # --model_dir exp/efficient-b0_kd_NST-E7/ --weight_name best.ts
# data_path = '/opt/ml/code/exp/mbv3-small_kd_NST-swinL/data.yml'
# model_path = '/opt/ml/code/exp/mbv3-small_kd_NST-swinL/model.yml'

# data_config = read_yaml(data_path)
# train_loader,test_loader,_ = create_dataloader(data_config)

# def trainer(model, optimizer, criterion):
#     global g_epoch
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx and batch_idx % 100 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 g_epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
#     g_epoch += 1

# def evaluator(model):
#     model.eval()
#     correct = 0.0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#     acc = 100 * correct / len(test_loader.dataset)
#     print('Accuracy: {}%\n'.format(acc))
#     return acc

# def optimizer_scheduler_generator(model):
#     data_config = read_yaml(data_path)
#     EPS = 1e-8
#     BETAS = (0.9, 0.999)
#     MOMENTUM = 0.9
#     WEIGHT_DECAY = 0.05
#     MIN_LR = data_config["INIT_LR"]/100
#     WARMUP_LR = MIN_LR/10
#     WARMUP_EPOCHS = 10
    
#     # optimizer = torch.optim.SGD(model_instance.model.parameters(), lr=data_config["INIT_LR"], momentum=0.9)
#     optimizer = optim.AdamW(model.parameters(), eps=EPS, betas=BETAS,lr=data_config["INIT_LR"], weight_decay=WEIGHT_DECAY)
#     scheduler = CosineLRScheduler(
#         optimizer, 
#         t_initial=data_config['EPOCHS'], 
#         warmup_t=WARMUP_EPOCHS, 
#         warmup_lr_init=WARMUP_LR,
#         lr_min = MIN_LR)
#     return optimizer,scheduler


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='PyTorch Example for model comporession')
#     parser.add_argument('--pretrain-epochs', type=int, default=1,#20,
#                         help='number of epochs to pretrain the model')
#     parser.add_argument('--fine-tune-epochs', type=int, default=1,#20,
#                         help='number of epochs to fine tune the model')
#     args = parser.parse_args()

#     print('\n' + '=' * 50 + ' START TO TRAIN THE MODEL ' + '=' * 50)
#     # model_instance = Model(model_path, verbose=True)
#     # model = model_instance.model.to(device)
#     model = torch.load('/opt/ml/pytorch-tensor-decompositions/decomposed_model').to(device)
#     optimizer, scheduler = optimizer_scheduler_generator(model)
#     criterion = torch.nn.CrossEntropyLoss()
#     pre_best_acc = 0.0
#     best_state_dict = None

#     for i in range(args.pretrain_epochs):
#         trainer(model, optimizer, criterion)
#         scheduler.step(epoch=1)
#         acc = evaluator(model)
#         if acc > pre_best_acc:
#             pre_best_acc = acc
#             best_state_dict = model.state_dict()
#     print("Best accuracy: {}".format(pre_best_acc))
#     model.load_state_dict(best_state_dict)
#     pre_flops, pre_params, _ = count_flops_params(model, torch.randn([128, 3, 224, 224]).to(device))
#     g_epoch = 0

#     # Start to prune and speedup
#     print('\n' + '=' * 50 + ' START TO PRUNE THE BEST ACCURACY PRETRAINED MODEL ' + '=' * 50)
#     config_list = [{
#         'total_sparsity': 0.5,
#         'op_types': ['BatchNorm2d'],
#         'max_sparsity_per_layer': 0.9
#     }]

#     optimizer, _ = optimizer_scheduler_generator(model)
#     pruner = SlimPruner(model, config_list, trainer, optimizer, criterion, training_epochs=1, scale=0.0001, mode='global')
#     _, masks = pruner.compress()
#     pruner.show_pruned_weights()
#     pruner._unwrap_model()
#     ModelSpeedup(model, dummy_input=torch.rand([10, 3, 224, 224]).to(device), masks_file=masks).speedup_model()
#     print('\n' + '=' * 50 + ' EVALUATE THE MODEL AFTER SPEEDUP ' + '=' * 50)
#     evaluator(model)

#     # Optimizer used in the pruner might be patched, so recommend to new an optimizer for fine-tuning stage.
#     print('\n' + '=' * 50 + ' START TO FINE TUNE THE MODEL ' + '=' * 50)
#     optimizer, scheduler = optimizer_scheduler_generator(model)
#     best_acc = 0.0
#     g_epoch = 0
#     for i in range(args.fine_tune_epochs):
#         trainer(model, optimizer, criterion)
#         scheduler.step(epoch=i)
#         best_acc = max(evaluator(model), best_acc)
#     flops, params, results = count_flops_params(model, torch.randn([128, 3, 224, 224]).to(device))
#     print(f'Pretrained model FLOPs {pre_flops/1e6:.2f} M, #Params: {pre_params/1e6:.2f}M, Accuracy: {pre_best_acc: .2f}%')
#     print(f'Finetuned model FLOPs {flops/1e6:.2f} M, #Params: {params/1e6:.2f}M, Accuracy: {best_acc: .2f}%')