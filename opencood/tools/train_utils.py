# -*- coding: utf-8 -*-
# License: TDG-Attribution-NonCommercial-NoDistrib


import glob
import importlib
import yaml
import os
import re
from datetime import datetime
import shutil
import torch
import torch.optim as optim

def backup_script(full_path, folders_to_save=["models", "data_utils", "utils", "loss"]):
    target_folder = os.path.join(full_path, 'scripts')
    if not os.path.exists(target_folder):
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
    
    current_path = os.path.dirname(__file__)  # __file__ refer to this file, then the dirname is "?/tools"

    for folder_name in folders_to_save:
        ttarget_folder = os.path.join(target_folder, folder_name)
        source_folder = os.path.join(current_path, f'../{folder_name}')
        shutil.copytree(source_folder, ttarget_folder)


def load_saved_model(saved_path, model):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """

    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    initial_epoch = findLastCheckpoint(saved_path)

    # print(initial_epoch)
    # print('resuming by loading epoch %d' % initial_epoch)
    if initial_epoch > 0:

        model_file = os.path.join(saved_path,
                                  'net_epoch%d.pth' % initial_epoch) \
            if initial_epoch != 10000 else os.path.join(saved_path,
                                                        'latest.pth')

        print('resuming by loading epoch %d' % initial_epoch,model_file)
        checkpoint = torch.load(
            model_file,
            map_location='cpu')

        model_dict = model.state_dict()

        print("now model len：", len(model_dict.keys()))
        print("checkpoint len：", len(checkpoint.keys()))


        state_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}

        # old code may has problem, its name can't match

        # for k, v in checkpoint.items():
        #     k_list = k.split('.')
        #     if k_list[0] == "vsa":
        #         k_list[0] = "rmpa"
        #         new_k = ".".join(k_list)
        #         state_dict[new_k] = v


        # for k, v in checkpoint.items():
        #     k_list = k.split('.')
        #     if k_list[0] == "cls_head" or k_list[0] == "reg_head" or k_list[0] == "dir_head":
        #         print(111,k_list)
        #         k_list.insert(0,"head")
        #         if k_list[1] == "cls_head":
        #             k_list[1] = "conv_cls"
        #         if k_list[1] == "reg_head":
        #             k_list[1] = "conv_box"
        #         if k_list[1] == "dir_head":
        #             k_list[1] = "conv_dir"
        #         new_k = ".".join(k_list)
        #         print("checkpoint  " + new_k, v.shape)
        #         print("now model  " + new_k, model_dict[new_k].shape)
        #         state_dict[new_k] = v

        # for k, v in checkpoint.items():
        #     k_list = k.split('.')
        #     if k_list[1] == "conv_cls" or k_list[1] == "conv_box" or k_list[1] == "conv_dir":
        #         print(222)
        #         if k_list[1] == "conv_cls":
        #             k_list[1]="cls_head"
        #
        #         if k_list[1] == "conv_box":
        #             k_list[1] = "reg_head"
        #
        #         if k_list[1] == "conv_dir":
        #             k_list[1] = "dir_head"
        #
        #         del k_list[0]
        #         new_k = ".".join(k_list)
        #         state_dict[new_k] = v


        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

        show_checkdata = torch.tensor(0.0)
        for k, v in model_dict.items():
            if show_checkdata.device != v.device:
                show_checkdata.to(v.device)
            show_checkdata = show_checkdata + v.sum()
        print("model data = ", show_checkdata)


        del checkpoint
    return initial_epoch, model


def load_saved_model_new(saved_path, model):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    checkpoint = torch.load(
        saved_path,
        map_location='cpu')
    
    model_dict = model.state_dict()

    # 打印checkpoint中的所有键和形状
    # print("==== Checkpoint Parameters ====")
    # for k, v in checkpoint.items():
    #     print(f"Checkpoint Parameter: {k}, Shape: {v.shape if hasattr(v, 'shape') else 'No shape'}")
    # # 打印model_dict中的所有键和形状
    # print("\n==== Model State Dict Parameters ====")
    # for k, v in model_dict.items():
    #     print(f"Model Parameter: {k}, Shape: {v.shape}")
    # # 找出checkpoint中存在但model_dict中不存在的键
    # print("\n==== Parameters in Checkpoint but not in Model ====")
    # for k in checkpoint.keys():
    #     if k not in model_dict.keys():
    #         print(f"Missing in model: {k}")
    # 找出model_dict中存在但checkpoint中不存在的键
    print("\n==== Parameters in Model but not in Checkpoint ====")
    for k in model_dict.keys():
        if k not in checkpoint.keys():
            print(f"Missing in checkpoint: {k}")
            
    # 检查哪些参数被加载了
    print("\n==== Parameters Successfully Loaded ====")
    loaded_params = []
    for k in model_dict.keys():
        if k in checkpoint.keys():
            print(f"Loaded: {k}, Shape: {model_dict[k].shape}")
    
    state_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
    print(f"\nNumber of parameters successfully loaded: {len(state_dict)}/{len(model_dict)}")

    # old code may has problem, its name can't match
    # for k, v in checkpoint.items():
    #     k_list = k.split('.')
    #     if k_list[0] == "vsa":
    #         k_list[0] = "rmpa"
    #         new_k = ".".join(k_list)
    #         state_dict[new_k] = v


    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    show_checkdata = torch.tensor(0.0)
    for k, v in model_dict.items():
        if show_checkdata.device != v.device:
            show_checkdata.to(v.device)
        show_checkdata = show_checkdata + v.sum()
    print("model data = ", show_checkdata)


    del checkpoint
    return model


def setup_train(hypes):
    """
    Create folder for saved model based on current timestep and model name

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for training:
    """
    model_name = hypes['name']
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name

    current_path = os.path.dirname(__file__)

    current_path = os.path.join(current_path, '../logs')

    full_path = os.path.join(current_path, folder_name)

    if not os.path.exists(full_path):
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
                backup_script(full_path)
            except FileExistsError:
                pass
        save_name = os.path.join(full_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)

        

    return full_path


def create_model(hypes):
    """
    Import the module "models/[model_name].py

    Parameters
    __________
    hypes : dict
        Dictionary containing parameters.

    Returns
    -------
    model : opencood,object
        Model object.
    """
    backbone_name = hypes['model']['core_method']
    backbone_config = hypes['model']['args']

    model_filename = "opencood.models." + backbone_name
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name.replace('_', '')

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print('backbone not found in models folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (model_filename,
                                                       target_model_name))
        exit(0)
    instance = model(backbone_config)
    return instance


def create_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    loss_func_name = hypes['loss']['core_method']
    loss_func_config = hypes['loss']['args']

    loss_filename = "opencood.loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace('_', '')

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion


def setup_optimizer(hypes, model):
    """
    Create optimizer corresponding to the yaml file

    Parameters
    ----------
    hypes : dict
        The training configurations.
    model : opencood model
        The pytorch model
    """
    method_dict = hypes['optimizer']
    optimizer_method = getattr(optim, method_dict['core_method'], None)
    if not optimizer_method:
        raise ValueError('{} is not supported'.format(method_dict['name']))
    if 'args' in method_dict:
        return optimizer_method(model.parameters(),
                                lr=method_dict['lr'],
                                **method_dict['args'])
    else:
        return optimizer_method(model.parameters(),
                                lr=method_dict['lr'])


def setup_lr_schedular(hypes, optimizer, init_epoch=None, data_loader=None):
    """
    Set up the learning rate schedular.

    Parameters
    ----------
    hypes : dict
        The training configurations.

    optimizer : torch.optimizer
    """
    lr_schedule_config = hypes['lr_scheduler']
    last_epoch = init_epoch if init_epoch is not None else 0
    

    if lr_schedule_config['core_method'] == 'step':
        from torch.optim.lr_scheduler import StepLR
        step_size = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule_config['core_method'] == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = MultiStepLR(optimizer,
                                milestones=milestones,
                                gamma=gamma)
    elif lr_schedule_config['core_method'] == 'cosine_warmup':
        # 使用transformers库中的余弦退火预热调度器
        from diffusers.optimization import get_cosine_schedule_with_warmup
        
        # 从配置获取参数
        num_warmup_steps = lr_schedule_config.get('num_warmup_steps', 500)
        num_training_steps = lr_schedule_config.get('num_training_steps', None)
        
        # 如果未指定总训练步数，则尝试计算
        if num_training_steps is None:
            num_training_steps = len(data_loader) * 10  # 默认10个epoch
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    else:
        from torch.optim.lr_scheduler import ExponentialLR
        gamma = lr_schedule_config['gamma']
        scheduler = ExponentialLR(optimizer, gamma)

    for _ in range(last_epoch):
        scheduler.step()

    return scheduler


def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) \
                or isinstance(inputs, str) or not hasattr(inputs, 'to'):
            return inputs
        return inputs.to(device, non_blocking=True)
