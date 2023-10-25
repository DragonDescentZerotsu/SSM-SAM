import argparse
import os
import csv

import yaml
from tqdm import tqdm  # 进度条显示
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import datasets
import models
import utils
from statistics import mean
import torch
import torch.nn as nn
import copy
import random
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.distributed as dist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''

    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss


def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    # log('{} dataset: size={}'.format(tag, len(dataset)))
    # for k, v in dataset[0].items():
    #     log('  {}: shape={}'.format(k, tuple(v.shape)))

    sampler = torch.utils.data.Sampler(dataset)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        shuffle=False, num_workers=6, pin_memory=True)
    return loader


def make_data_loaders():
    support_train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    query_train_loader = make_data_loader(config.get('val_dataset'), tag='val')
    support_eval_loader = make_data_loader(config.get('test_dataset'), tag='val')
    query_eval_loader = make_data_loader(config.get('query_eval_dataset'), tag='val')
    return support_train_loader, query_train_loader, support_eval_loader, query_eval_loader


def make_data_loaders_dict(all_task_list):
    """
    返回与task对应的data_loader字典
    """
    global config
    support_train_loader_dict = {}
    query_train_loader_dict = {}
    support_eval_loader_dict = {}
    query_eval_loader_dict = {}
    data_path = config['data_path']
    for task in all_task_list:
        config['train_dataset']['dataset']['args']['root_path_1'] = os.path.join(data_path, task, 'support_train',
                                                                                 'img')
        config['train_dataset']['dataset']['args']['root_path_2'] = os.path.join(data_path, task, 'support_train',
                                                                                 'mask')
        config['val_dataset']['dataset']['args']['root_path_1'] = os.path.join(data_path, task, 'query_train', 'img')
        config['val_dataset']['dataset']['args']['root_path_2'] = os.path.join(data_path, task, 'query_train', 'mask')
        config['test_dataset']['dataset']['args']['root_path_1'] = os.path.join(data_path, task, 'support_eval', 'img')
        config['test_dataset']['dataset']['args']['root_path_2'] = os.path.join(data_path, task, 'support_eval', 'mask')
        config['query_eval_dataset']['dataset']['args']['root_path_1'] = os.path.join(data_path, task, 'query_eval',
                                                                                      'img')
        config['query_eval_dataset']['dataset']['args']['root_path_2'] = os.path.join(data_path, task, 'query_eval',
                                                                                      'mask')
        support_train_loader, query_train_loader, support_eval_loader, query_eval_loader = make_data_loaders()
        support_train_loader_dict[task] = support_train_loader
        query_train_loader_dict[task] = query_train_loader
        support_eval_loader_dict[task] = support_eval_loader
        query_eval_loader_dict[task] = query_eval_loader
        print(config['train_dataset']['dataset']['args']['root_path_1'])  # 为了调试
    return support_train_loader_dict, query_train_loader_dict, support_eval_loader_dict, query_eval_loader_dict


def eval_psnr(support_eval_loader, query_eval_loader, model, eval_type=None):
    model.eval()

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

    pbar = tqdm(total=min(len(query_eval_loader) * config['test_dataset']['batch_size'], config['out_eval_INR']), leave=True,
                desc='evaluation')

    pred_list = []
    gt_list = []
    with torch.no_grad():
        for i, (support_batch, query_batch) in enumerate(zip(support_eval_loader, query_eval_loader)):
            if (query_batch['inp'].shape[0] != config['train_dataset']['batch_size'] or (i + 1) * config['train_dataset'][
                'batch_size'] > config['out_eval_INR']):
                break

            inp = query_batch['inp'].to(device)
            batch_gt = query_batch['gt'].to(device)
            support_mask = support_batch['gt'].to(device)

            pred = torch.sigmoid(model(inp, batch_gt, support_mask, num_points=5))

            pred_list.append(pred)
            gt_list.append(batch_gt)
            if pbar is not None:
                pbar.update(config['test_dataset']['batch_size'])

        if pbar is not None:
            pbar.close()

    pred_list = torch.cat(pred_list, 1)
    gt_list = torch.cat(gt_list, 1)
    result1, result2, result3, result4 = metric_fn(pred_list, gt_list)

    return result1, result2, result3, result4, metric1, metric2, metric3, metric4


def prepare_training():
    if config.get('resume') is not None:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    # log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(all_task_list, support_train_loader_dict, query_train_loader_dict, support_eval_loader_dict,
          query_eval_loader_dict, model, loss_mode):
    model.train()
    inner_epochs = config['train_dataset']['inner_epochs']  # 每一个任务内有几次梯度更新
    task_batch_index = 0
    task_loss_list = []
    task_batch_size = config['train_dataset']['task_batch_size']  # 一次进来多少个task
    while task_batch_index < len(all_task_list):
        if task_batch_index + task_batch_size < len(all_task_list):
            train_task_list = all_task_list[task_batch_index:task_batch_index + task_batch_size]
        else:
            train_task_list = all_task_list[task_batch_index:]
        task_batch_index += task_batch_size
        in_batch_task_loss_list = []
        model.optimizer.zero_grad()  # 下面within-task testing之后要累加梯度

        for i, task in enumerate(train_task_list):
            model_inner = copy.deepcopy(model).to(device)  # 这样的话估计里面那个optimizer还是model的参数，不是inner_model的参数
            inner_optimizer = utils.make_optimizer(model_inner.parameters(),
                                                   config['inner_optimizer'])  # 给inner_model一个新的optimizer
            model_inner.optimizer = inner_optimizer
            print(device)
            model_inner = nn.DataParallel(model_inner).to(device)
            support_train_loader, query_train_loader, support_eval_loader, query_eval_loader = \
            support_train_loader_dict[task], query_train_loader_dict[task], support_eval_loader_dict[task], query_eval_loader_dict[task]
            pbar = tqdm(total=min(config['train_img_number_restrict'] * inner_epochs,
                                  config['train_dataset']['batch_size'] * len(support_train_loader) * inner_epochs), leave=True,
                        desc='{}/{} within task training: {}'.format(i + 1, len(train_task_list),
                                                                     task))  # 包裹在任何可迭代对象上即可显示进度条
            for inner_epoch in range(inner_epochs):
                for j, (support_batch, query_batch) in enumerate(zip(support_train_loader, query_train_loader)):
                    if ((j + 1) * config['train_dataset']['batch_size'] > config['train_img_number_restrict']
                            or query_batch['inp'].shape[0] != config['train_dataset']['batch_size']):
                        break  # 添加每个任务的训练图片张数限制
                    inp = query_batch['inp'].to(device)  # input
                    gt = query_batch['gt'].to(device)  # ground truth
                    support_mask = support_batch['gt'].to(device)
                    model_inner.module.optimizer.zero_grad()
                    pred = model_inner(inp, gt, support_mask, num_points=20)
                    if loss_mode == 'bce':
                        criterionBCE = torch.nn.BCEWithLogitsLoss()
                    elif loss_mode == 'bbce':
                        criterionBCE = BBCEWithLogitLoss()
                    elif loss_mode == 'iou':
                        criterionBCE = torch.nn.BCEWithLogitsLoss()
                    batch_loss = criterionBCE(pred, gt)
                    if loss_mode == 'iou':
                        batch_loss += _iou_loss(pred, gt)

                    batch_loss.backward()
                    model_inner.optimizer.step()

                    if pbar is not None:
                        pbar.update(config['train_dataset']['batch_size'])

            if pbar is not None:
                pbar.close()

            eval_loss_list = []
            # model_inner.eval()
            # cumulative_loss = torch.zeros(1).to(device)
            model_inner.optimizer.zero_grad()  # 前面within_task_train的梯度不能要，清零
            ebar = tqdm(
                total=min(config['eval_img_number_restrict'], config['val_dataset']['batch_size'] * len(query_eval_loader)),
                leave=True,
                desc='within task test')  # 包裹在任何可迭代对象上即可显示进度条
            for k, (support_batch, query_batch) in enumerate(zip(support_eval_loader, query_eval_loader)):

                if ((k + 1) * config['val_dataset']['batch_size'] > config['eval_img_number_restrict']
                        or query_batch['inp'].shape[0] != config['val_dataset']['batch_size']):
                    break  # 添加每个任务的训练图片张数限制
                # for k, v in batch.items():
                #     batch[k] = v.to(device)
                inp = query_batch['inp'].to(device)  # input
                gt = query_batch['gt'].to(device)  # ground truth
                support_mask = support_batch['gt'].to(device)
                # print('到这了')
                pred = model_inner(inp, gt, support_mask, num_points=5)
                if loss_mode == 'bce':
                    criterionBCE = torch.nn.BCEWithLogitsLoss()
                elif loss_mode == 'bbce':
                    criterionBCE = BBCEWithLogitLoss()
                elif loss_mode == 'iou':
                    criterionBCE = torch.nn.BCEWithLogitsLoss()
                batch_loss = criterionBCE(pred, gt)
                if loss_mode == 'iou':
                    batch_loss += _iou_loss(pred, gt)
                # (batch_loss / len(eval_loader)).backward()  # 这样做显存才不会超标，下面注释掉的那种是不行的
                batch_loss.backward()  # 想不通为啥上面要除以一个len(eval_loader)了，好像弄错了
                eval_loss_list.append(batch_loss.item())

                if ebar is not None:
                    ebar.update(config['train_dataset']['batch_size'])

            if ebar is not None:
                ebar.close()
                # cumulative_loss += batch_loss
            # cumulative_loss /= len(eval_loader)  # 得到一个任务的测试loss
            # model_inner.optimizer.zero_grad()
            # cumulative_loss.backward()  # 得到一个任务inner_model的梯度

            model_inner = model_inner.to('cpu')  # 反向传播完成之后到cpu上操作

            torch.cuda.empty_cache()  # 再清除一次显存
            for param_model, param_inner_model in zip(model.parameters(), model_inner.parameters()):
                if param_inner_model.grad is not None:
                    if param_model.grad is None:
                        param_model.grad = param_inner_model.grad / len(train_task_list)  # 还是None的时候不能相加先赋值
                    else:
                        param_model.grad += param_inner_model.grad / len(train_task_list)  # 需要对任务取平均并累计一个任务batch里的梯度

            del model_inner  # 获得梯度之后就不再需要这个inner模型了
            in_batch_task_loss_list.append(mean(eval_loss_list))

        in_batch_across_task_loss = mean(in_batch_task_loss_list)
        model.optimizer.step()

        task_loss_list.append(in_batch_across_task_loss)
    across_task_loss = mean(task_loss_list)

    return across_task_loss


def main(config_, save_path, args):
    global config, log, writer, log_info  # 声明为全局变量
    config = config_
    log = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    # train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model.optimizer = optimizer
    lr_scheduler = CosineAnnealingLR(model.optimizer, config['epoch_max'], eta_min=config.get('lr_min'))

    model = model.to('cpu')

    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model.load_state_dict(sam_checkpoint, strict=False)  # 载入权重

    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)

    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 0
    timer = utils.Timer()
    best_loss = 1e8

    all_task_list = os.listdir(config['data_path'])  # 所有的任务都列在这里
    eval_task = 'liver'  # TODO:做val_loader
    exclude_task_list = ['USF_shadow', 'SBU_shadow', 'ISTD_shadow', 'polyp', 'COD10K', 'cup', 'BraTS20', eval_task,
                         'ISBI', 'transparent', 'all_train_meta_data', '__MACOSX', 'large_test', '.DS_Store']
    all_train_task_list = list(set(all_task_list) - set(exclude_task_list))
    support_train_loader_dict, query_train_loader_dict, support_eval_loader_dict, query_eval_loader_dict = make_data_loaders_dict(
        all_train_task_list)

    config['test_dataset']['dataset']['args']['root_path_1'] = os.path.join(config['data_path'], eval_task, 'support_eval', 'img')
    config['test_dataset']['dataset']['args']['root_path_2'] = os.path.join(config['data_path'], eval_task, 'support_eval', 'mask')
    config['query_eval_dataset']['dataset']['args']['root_path_1'] = os.path.join(config['data_path'], eval_task, 'query_eval', 'img')
    config['query_eval_dataset']['dataset']['args']['root_path_2'] = os.path.join(config['data_path'], eval_task, 'query_eval', 'mask')
    support_eval_loader = make_data_loader(config.get('test_dataset'), tag='test')  # 给下面eval_psnr用
    query_eval_loader = make_data_loader(config.get('query_eval_dataset'), tag='test')  # 给下面eval_psnr用
    for epoch in range(epoch_start, epoch_max + 1):
        print(f"outer epoch {epoch} : ")
        t_epoch_start = timer.t()
        loss_mode = config['model']['args']['loss']
        print("loss mode: ", loss_mode)
        random.shuffle(all_train_task_list)
        train_loss_G = train(all_train_task_list, support_train_loader_dict, query_train_loader_dict,
                             support_eval_loader_dict, query_eval_loader_dict, model,
                             loss_mode)
        lr_scheduler.step()

        with open('./save/training_log.csv', 'a', newline='') as file_obj:
            writer.add_scalar('Loss across tasks', train_loss_G, epoch)
            writer.flush()
            csv_writer = csv.writer(file_obj)
            csv_writer.writerow([epoch, train_loss_G])
            file_obj.close()

        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        # log_info.append('train G: loss={:.4f}'.format(train_loss_G))
        # writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

        model_spec = config['model']
        model_spec['sd'] = model.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()

        save(config, model, save_path, 'last')
        if train_loss_G < best_loss:
            best_loss = train_loss_G
            print(f'save best loss: {best_loss}')
            save(config, model, save_path, 'best_loss')  # 训练的loss小就保存

        if epoch_val is not None:
            if epoch % epoch_val == 0:
                # torch.cuda.empty_cache()
                model = model.to(device)
                result1, result2, result3, result4, metric1, metric2, metric3, metric4 = eval_psnr(support_eval_loader,
                                                                                                   query_eval_loader,
                                                                                                   model,
                                                                                                   eval_type=config.get(
                                                                                                       'eval_type'))

                log_info.append('val: {}={:.4f}'.format(metric1, result1))
                writer.add_scalars(metric1, {'val': result1}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric2, result2))
                writer.add_scalars(metric2, {'val': result2}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric3, result3))
                writer.add_scalars(metric3, {'val': result3}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric4, result4))
                writer.add_scalars(metric4, {'val': result4}, epoch)

                if config['eval_type'] != 'ber':
                    if result1 > max_val_v:
                        max_val_v = result1
                        save(config, model, save_path, 'best_dice')
                else:
                    if result1 > max_val_v:
                        max_val_v = result1
                        print(f'save best dice: {max_val_v}')
                        save(config, model, save_path, 'best_dice')

                t = timer.t()
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                t_epoch = utils.time_text(t - t_epoch_start)
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

                log(', '.join(log_info))
                writer.flush()
        model = model.to('cpu')


def save(config, model, save_path, name):
    print("model name = ", config['model']['name'])
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(),
                   os.path.join(save_path, f"model_epoch_{name}.pth"))  # ./save/save_name/model_epoch_name.pth


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="/root/SAM_Adapter_MAML/configs/new_module.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]  # 没输入name时
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)  # ./save/save_name

    writer = SummaryWriter(log_dir='./tensorboard')
    main(config, save_path, args=args)
