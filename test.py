import argparse
import os
from statistics import mean
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


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
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

    pbar = tqdm(loader, leave=True, desc='val')

    with torch.no_grad():
        for i, batch in enumerate(pbar):
            if batch['inp'].shape[0] != config['val_dataset']['batch_size']:
                break
            inp = batch['inp'].to(device)
            gt = batch['gt'].to(device)

            pred = torch.sigmoid(model(inp, gt, num_points=1))

            result1, result2, result3, result4 = metric_fn(pred, gt)
            val_metric1.add(result1.item(), inp.shape[0])
            val_metric2.add(result2.item(), inp.shape[0])
            val_metric3.add(result3.item(), inp.shape[0])
            val_metric4.add(result4.item(), inp.shape[0])

            if verbose:
                pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))
                pbar.set_description('val {} {:.4f}'.format(metric2, val_metric2.item()))
                pbar.set_description('val {} {:.4f}'.format(metric3, val_metric3.item()))
                pbar.set_description('val {} {:.4f}'.format(metric4, val_metric4.item()))

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
                        shuffle=True, num_workers=8, pin_memory=True)
    return loader


def make_data_loaders(config):
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    test_loader = make_data_loader(config.get('test_dataset'), tag='val')
    return train_loader, val_loader, test_loader


def train(loader, model, loss_mode):
    global config
    model.train()
    bar = tqdm(total=min(len(loader) * config['train_dataset']['batch_size'], config['train_img_number_restrict']),
               leave=True, desc='train')
    loss = []
    for i, batch in enumerate(loader):
        if ((i + 1) * config['train_dataset']['batch_size'] > config['train_img_number_restrict']
                or batch['inp'].shape[0] != config['val_dataset']['batch_size']):
            break
        inp = batch['inp'].to(device)  # input
        gt = batch['gt'].to(device)  # ground truth
        model.optimizer.zero_grad()
        pred = model(inp, gt, num_points=20)
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

        batch_loss.backward()
        model.optimizer.step()  # TODO: 学习率得设置成fixed，应该要给optimizer改了
        loss.append(batch_loss.item())
        if bar is not None:
            bar.update(config['train_dataset']['batch_size'])
    if bar is not None:
        bar.close()

    return mean(loss)


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
    parser.add_argument('--config', default="/root/SAM_Adapter_MAML/configs/test.yaml")  # /root/SAM_Adapter_MAML/configs/test.yaml
    parser.add_argument('--model', default='/root/autodl-tmp/best_dice_50imgs_12chunk_spleen.pth')  # 权重保存路径/root/autodl-tmp/model_epoch_best_ber.pth
    parser.add_argument('--prompt', default='none')
    args = parser.parse_args()

    writer = SummaryWriter(log_dir='./tensorboard')

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # spec = config['test_dataset']
    # test_dataset = datasets.make(spec['dataset'])
    # test_dataset = datasets.make(spec['wrapper'], args={'dataset':
    # test_dataset})
    # test_loader = DataLoader(test_dataset, batch_size=spec['batch_size'], shuffle=False,
    #                     num_workers=8)
    data_path = config['data_path']
    task = config['test_task']
    config['train_dataset']['dataset']['args']['root_path_1'] = os.path.join(data_path, task, 'train', 'img')
    config['train_dataset']['dataset']['args']['root_path_2'] = os.path.join(data_path, task, 'train', 'mask')
    config['val_dataset']['dataset']['args']['root_path_1'] = os.path.join(data_path, task, 'eval', 'img')
    config['val_dataset']['dataset']['args']['root_path_2'] = os.path.join(data_path, task, 'eval', 'mask')
    config['test_dataset']['dataset']['args']['root_path_1'] = os.path.join(data_path, task, 'eval', 'img')
    config['test_dataset']['dataset']['args']['root_path_2'] = os.path.join(data_path, task, 'eval', 'mask')
    print('data path: ' + config['train_dataset']['dataset']['args']['root_path_1'])
    train_loader, test_loader, eval_loader = make_data_loaders(config)  # 这里的eval是真正的evaluation_loader

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

    epochs = config['train_dataset']['epochs']
    loss_mode = config['model']['args']['loss']
    min_val_v = 0
    for epoch in range(epochs):
        print(f'epoch: {epoch + 1}')
        loss_log = train(train_loader, model, loss_mode)  # 训练
        lr_scheduler.step()  # 余弦退火学习率
        metric1, metric2, metric3, metric4 = eval_psnr(eval_loader, model,
                                                       data_norm=config.get('data_norm'),
                                                       eval_type=config.get('eval_type'),
                                                       eval_bsize=config.get('eval_bsize'),
                                                       verbose=True)  # evaluation
        log_info = ['epoch {}/{}'.format(epoch + 1, epochs)]
        log_info.append('dice' + ':' + str(metric1))
        log_info.append('iou' + ':' + str(metric2))
        log_info.append('ber' + ':' + str(metric3))
        log(', '.join(log_info))

        if metric1 > min_val_v:
            min_val_v = metric1
            save(config, model, save_path, 'best_dice')  # TODO:没debug
        writer.add_scalars('Test Loss', {'loss': loss_log}, epoch + 1)
        writer.add_scalars('metric', {'dice': metric1, 'iou': metric2, 'ber': metric3},
                           epoch + 1)
        writer.flush()

    sam_checkpoint = torch.load(os.path.join(save_path, "model_epoch_best_dice.pth"), map_location=device)  # TODO:没debug
    model.load_state_dict(sam_checkpoint, strict=False)  # 这样load出来的model没有require_grad的属性记录
    model.eval()
    metric1, metric2, metric3, metric4 = eval_psnr(test_loader, model,
                                                   data_norm=config.get('data_norm'),
                                                   eval_type=config.get('eval_type'),
                                                   eval_bsize=config.get('eval_bsize'),
                                                   verbose=True)
    print('metric1: {:.4f}'.format(metric1))
    print('metric2: {:.4f}'.format(metric2))
    print('metric3: {:.4f}'.format(metric3))
    print('metric4: {:.4f}'.format(metric4))

    # save(config, model, save_path, round(metric3, 3))

    epoch = config['resume']
    des = 'best ber test for first meta-learning'
    with open('./save/test_log.csv', 'a', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([epoch, des, metric1, metric2, metric3, metric4])
        f.close()
