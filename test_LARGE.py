import argparse
import os
from statistics import mean

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

from torchvision import transforms
from mmcv.runner import load_checkpoint
import csv
from train_MAML import BBCEWithLogitLoss, _iou_loss, save
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


def save_fig(tensor, name):
    numpy_array = (tensor * 255).byte().numpy()
    numpy_array = numpy_array.transpose(1, 2, 0)
    if numpy_array.shape[-1] == 1:
        numpy_array = np.squeeze(numpy_array)
        appendix = '.png'
    else:
        # numpy_array = np.squeeze(numpy_array[..., 0])
        appendix = '.jpg'
    print(numpy_array.shape)
    img = Image.fromarray(numpy_array)
    img.save(os.path.join('/root/SAM_Adapter_MAML/save/imgs', name + appendix))


def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds


def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_psnr(test_batch, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    global config
    model.eval()
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.cal_dice_iou_ber
        metric1, metric2, metric3, metric4 = 'dice', 'iou', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()

    inp = test_batch['inp'].to(device)
    gt = test_batch['gt'].to(device)

    pred = torch.sigmoid(model(inp, gt, num_points=30))

    # model.cpu()
    # torch.cuda.empty_cache()
    # del model
    # print('here 1 97')
    result1, result2, result3, result4 = metric_fn(pred, gt)
    val_metric1.add(result1.item(), inp.shape[0])
    val_metric2.add(result2.item(), inp.shape[0])
    val_metric3.add(result3.item(), inp.shape[0])
    val_metric4.add(result4.item(), inp.shape[0])
    pred_cpu = pred.cpu()
    del pred
    # print(f'device:{pred_cpu.device}')
    torch.cuda.empty_cache()

    for i, mask in enumerate(pred_cpu):
        save_fig(mask, f'pred_{i + 1}')

    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item()


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    # log('{} dataset: size={}'.format(tag, len(dataset)))
    # for k, v in dataset[0].items():
    # log('  {}: shape={}'.format(k, tuple(v.shape)))

    sampler = torch.utils.data.Sampler(dataset)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        shuffle=False, num_workers=8, pin_memory=True)
    return loader


def make_data_loaders(config):
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    test_loader = make_data_loader(config.get('test_dataset'), tag='val')
    return train_loader, val_loader, test_loader


def train(train_batch, model, loss_mode, epoch):
    global config
    model.train()
    inp = train_batch['inp'].to(device)  # input
    gt = train_batch['gt'].to(device)  # ground truth
    model.optimizer.zero_grad()
    pred = model(inp, gt, num_points=3)
    if loss_mode == 'bce':
        criterionBCE = torch.nn.BCEWithLogitsLoss()
    elif loss_mode == 'bbce':
        criterionBCE = BBCEWithLogitLoss()
    elif loss_mode == 'iou':
        criterionBCE = torch.nn.BCEWithLogitsLoss()

    batch_loss = criterionBCE(pred, gt)
    # print('这里')
    if loss_mode == 'iou':
        batch_loss += _iou_loss(pred, gt)

    if (epoch + 1) % 5 == 0 or epoch == 0:  # 保存图片用
        pred = torch.sigmoid(pred)
        pred_c = pred.cpu()
        save_fig(pred_c[0], f'train_pred_{epoch + 1}')

    batch_loss.backward()
    model.optimizer.step()  # TODO: 学习率得设置成fixed，应该要给optimizer改了

    return batch_loss.item()


def evaluation(eval_loader, model):
    model.eval()
    global config
    with torch.no_grad():
        metric1, metric2, metric3, metric4 = eval_psnr(eval_loader, model,
                                                       data_norm=config.get('data_norm'),
                                                       eval_type=config.get('eval_type'),
                                                       eval_bsize=config.get('eval_bsize'),
                                                       verbose=True)
    return metric1, metric2, metric3, metric4


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default="/root/SAM_Adapter_MAML/configs/test_LARGE.yaml")  # /root/SAM_Adapter_MAML/configs/test.yaml
    parser.add_argument('--model',
                        default='/root/autodl-tmp/best_dice_50imgs_12chunk_Lkidney.pth')  # 权重保存路径/root/autodl-tmp/model_epoch_best_ber.pth
    parser.add_argument('--prompt', default='none')
    args = parser.parse_args()

    writer = SummaryWriter(log_dir='./tensorboard')

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # spec = config['test_dataset']
    # test_dataset = datasets.make(spec['dataset'])
    # test_dataset = datasets.make(spec['wrapper'], args={'dataset': test_dataset})
    # test_loader = DataLoader(test_dataset, batch_size=spec['batch_size'], shuffle=False,
    #                     num_workers=8)
    data_path = config['data_path']
    task = 'left_kidney_normal'
    config['train_dataset']['dataset']['args']['root_path_1'] = os.path.join(data_path, task, 'support', 'img')
    config['train_dataset']['dataset']['args']['root_path_2'] = os.path.join(data_path, task, 'support', 'mask')
    config['val_dataset']['dataset']['args']['root_path_1'] = os.path.join(data_path, task, 'query', 'img')
    config['val_dataset']['dataset']['args']['root_path_2'] = os.path.join(data_path, task, 'query', 'mask')
    config['test_dataset']['dataset']['args']['root_path_1'] = os.path.join(data_path, task, 'query', 'img')
    config['test_dataset']['dataset']['args']['root_path_2'] = os.path.join(data_path, task, 'query', 'mask')
    print('data path: ' + config['train_dataset']['dataset']['args']['root_path_1'])
    train_loader, test_loader, eval_loader = make_data_loaders(config)  # 这里的eval是真正的evaluation_loader

    dice_recort = []
    total_data_len = len(os.listdir(os.path.join(data_path, task, 'query', 'img')))
    for i, (train_batch, test_batch) in enumerate(zip(train_loader, test_loader)):
        torch.cuda.empty_cache()
        model = models.make(config['model']).to(device)
        sam_checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(sam_checkpoint, strict=False)  # 这样load出来的model没有require_grad的属性记录
        for name, para in model.named_parameters():
            if "image_encoder" in name and "prompt_generator" not in name:
                para.requires_grad_(False)
        optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
        max_epoch = config['epoch_max']
        lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
        model.optimizer = optimizer

        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
        save_path = os.path.join('./save', save_name)  # ./save/save_name

        log = utils.set_save_path(save_path, remove=False)

        epochs = 30  # config['train_dataset']['epochs']
        loss_mode = config['model']['args']['loss']
        min_val_v = 0
        for epoch in range(epochs):  # 每个one-shot样本更新几步
            print(f'epoch: {epoch + 1}')
            loss_log = train(train_batch, model, loss_mode, epoch)  # 训练
            print(f'loss:{loss_log}')
            lr_scheduler.step()  # 余弦退火学习率
        metric1, metric2, metric3, metric4 = eval_psnr(test_batch, model,
                                                       data_norm=config.get('data_norm'),
                                                       eval_type=config.get('eval_type'),
                                                       eval_bsize=config.get('eval_bsize'),
                                                       verbose=True)  # evaluation

        dice_recort.append(metric1)
        print(f'dice : {metric1}')
        writer.add_scalars('percentage', {'val': ((i + 1) * 5 / total_data_len) * 100}, i+1)
        writer.add_scalars('DSC', {'val': mean(dice_recort)}, i + 1)
        # if metric1 < 0.4 or metric1 > 0.7:
        save_fig(train_batch['inp'][0], 'train_img')
        save_fig(train_batch['gt'][0], 'train_mask')
        for j in range(5):
            print(j)
            save_fig(test_batch['inp'][j], f'test_img_{j + 1}')
            save_fig(test_batch['gt'][j], f'test_mask_{j + 1}')
        # save_fig(pred[j], f'pred_{j + 1}')

    DSC = mean(dice_recort)
    print(f'mean dice score: {DSC}')
