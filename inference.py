import argparse
import os
import time
from utils import misc
import torch
from torch.autograd import Variable
from models import dataset
from utils.common_utils import process_config
from datetime import datetime
from utee import hook
from utils.group_utils import (compute_strip_group, enable_calibrate, 
                                disable_calibrate, enable_hessian_quantizer, model_strip_group)


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/app/configs/exp_for_cifar/inference.json', required=False)
    args = parser.parse_args()

    print('Using config!')
    configs = process_config(args.config)

    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    configs.logdir = os.path.join(configs.logdir, configs.net, configs.dataset)
    inference_log_dir = os.path.join(configs.logdir, 'best.pth')

    # seed
    configs.cuda = torch.cuda.is_available()
    torch.manual_seed(configs.seed)

    if configs.cuda:
        torch.cuda.manual_seed(configs.seed)
        device = 'cuda'
    else:
        device = 'cpu'

    # data loader and model
    assert configs.dataset in ['cifar10', 'cifar100', 'imagenet'], configs.dataset
    if configs.dataset == 'cifar10':
        train_loader, test_loader = dataset.get_cifar10(batch_size=configs.batch_size, num_workers=1)
        num_classes = 100
    elif configs.dataset == 'cifar100':
        train_loader, test_loader = dataset.get_cifar100(batch_size=configs.batch_size, num_workers=1)
        num_classes = 100
    elif configs.dataset == 'imagenet':
        train_loader, test_loader = dataset.get_imagenet(batch_size=configs.batch_size, num_workers=1)
        num_classes = 1000
    else:
        raise ValueError("Unknown dataset type")

    if configs.net == 'VGG16':
        from models.VGG import vgg16
        model = vgg16(args=configs, num_classes=num_classes, init_weights=False)
        model.load_state_dict(torch.load(inference_log_dir))
    elif configs.net == 'ResNet18':
        from models.ResNet import resnet18
        model = resnet18(args=configs, num_classes=num_classes)
        model.load_state_dict(torch.load(inference_log_dir))
    elif configs.net == 'ResNet50':
        from models.ResNet import resnet50
        model = resnet50(args=configs, num_classes=num_classes)
        model.load_state_dict(torch.load(inference_log_dir))
    else:
        raise ValueError("Unknown model type")

    t_begin = time.time()

    criterion = torch.nn.CrossEntropyLoss()

    if configs.quantization == "hessian":
        model = enable_calibrate(model)
        importances, strip_importances_per_layer = compute_strip_group(model, test_loader, criterion, device)
        model = disable_calibrate(model)
        
        model = enable_hessian_quantizer(model)
        model = model_strip_group(model, importances, strip_importances_per_layer, configs.bits, configs.ratio)
    test_loss = 0
    correct = 0

    model.to(device)
    model.eval()
    for i, (data, target) in enumerate(test_loader):
        # if i==0:
        #     hook_handle_list = hook.hardware_evaluation(
        #         model, configs.wl_weight, configs.wl_activate, 
        #         configs.subArray, configs.parallelRead, configs.net
        #     )
        #     model = hook.enable_hook(model)
        indx_target = target.clone()
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss_i = criterion(output, target)
            test_loss += test_loss_i.data
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(indx_target).sum()
        # if i==0:
        #     hook.remove_hook_list(hook_handle_list)
        #     model = hook.disable_hook(model)

    test_loss = test_loss / len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)

    accuracy = acc.cpu().data.numpy()


    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset), acc))
