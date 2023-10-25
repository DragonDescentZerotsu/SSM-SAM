import argparse
import os
import csv

import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import datasets
import models
import utils
from statistics import mean
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
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
    global config
    support_train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    query_train_loader = make_data_loader(config.get('val_dataset'), tag='val')
    support_eval_loader = make_data_loader(config.get('test_dataset'), tag='val')
    query_eval_loader = make_data_loader(config.get('query_eval_dataset'), tag='val')
    return support_train_loader, query_train_loader, support_eval_loader, query_eval_loader


def eval_psnr(support_eval_loader, query_eval_loader, model, eval_type=None):
    global config
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
    pbar = tqdm(total=len(query_eval_loader) * config['test_dataset']['batch_size'], leave=True, desc='evaluation')

    pred_list = []
    gt_list = []
    with torch.no_grad():
        for i, (support_batch, query_batch) in enumerate(zip(support_eval_loader, query_eval_loader)):
            if query_batch['inp'].shape[0] != config['train_dataset']['batch_size']:
                break
            # for k, v in batch.items():
            #     batch[k] = v.to(device)

            inp = query_batch['inp'].to(device)
            batch_gt = query_batch['gt'].to(device)
            gaussian_mask = support_batch['gt'].to(device)

            pred = torch.sigmoid(model(inp, batch_gt, gaussian_mask, num_points=20))

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
    global config
    if config.get('resume') is not None:
        model = models.make(config['model']).to(device)
        optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).to(device)
        optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    # log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(support_train_loader, query_train_loader, model, loss_mode):
    global config
    model.train()
    pbar = tqdm(
        total=len(query_train_loader) * config['train_dataset']['batch_size'],
        leave=True,
        desc='train original adapter SAM')

    loss_list = []
    for i, (support_batch, query_batch) in enumerate(zip(support_train_loader, query_train_loader)):
        if query_batch['inp'].shape[0] != config['val_dataset']['batch_size']:
            break
        inp = query_batch['inp'].to(device)
        gt = query_batch['gt'].to(device)
        gaussian_mask = support_batch['gt'].to(device)
        model.optimizer.zero_grad()
        pred = model(inp, gt, gaussian_mask, num_points=20)
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
        model.optimizer.step()
        loss_list.append(batch_loss)
        # print('loss: ', batch_loss.item())
        if pbar is not None:
            pbar.update(config['train_dataset']['batch_size'])

    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    return mean(loss)


def main(config_, save_path, args):
    global config, log, writer, log_info  # 声明为全局变量
    config = config_
    log = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    data_path = config['data_path']
    config['train_dataset']['dataset']['args']['root_path_1'] = os.path.join(data_path, 'all_train_meta_data/support_train/img')
    config['train_dataset']['dataset']['args']['root_path_2'] = os.path.join(data_path, 'all_train_meta_data/support_train/mask')
    config['val_dataset']['dataset']['args']['root_path_1'] = os.path.join(data_path, 'all_train_meta_data/query_train/img')
    config['val_dataset']['dataset']['args']['root_path_2'] = os.path.join(data_path, 'all_train_meta_data/query_train/mask')
    eval_task = config['test_task']
    config['test_dataset']['dataset']['args']['root_path_1'] = os.path.join(data_path, f'{eval_task}/support_eval/img')
    config['test_dataset']['dataset']['args']['root_path_2'] = os.path.join(data_path, f'{eval_task}/support_eval/mask')
    config['query_eval_dataset']['dataset']['args']['root_path_1'] = os.path.join(data_path, f'{eval_task}/query_eval/img')
    config['query_eval_dataset']['dataset']['args']['root_path_2'] = os.path.join(data_path, f'{eval_task}/query_eval/mask')
    support_train_loader, query_train_loader, support_eval_loader, query_eval_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model.optimizer = optimizer
    lr_scheduler = CosineAnnealingLR(model.optimizer, config['epoch_max'], eta_min=config.get('lr_min'))

    model = model.to(device)

    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model.load_state_dict(sam_checkpoint, strict=False)

    for name, para in model.named_parameters():  # 锁定权重
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)

    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 0
    timer = utils.Timer()
    best_loss = 1e9
    for epoch in range(epoch_start, epoch_max + 1):
        print(f"{epoch} : ")
        t_epoch_start = timer.t()
        loss_mode = config['model']['args']['loss']
        print("loss mode: ", loss_mode)
        train_loss_G = train(support_train_loader, query_train_loader, model, loss_mode)
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
            save(config, model, save_path, 'best_loss_OSAM')

        if epoch_val is not None:
            if epoch % epoch_val == 0:
                # torch.cuda.empty_cache()

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
                        save(config, model, save_path, 'best_dice_OSAM')

                t = timer.t()
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                t_epoch = utils.time_text(t - t_epoch_start)
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

                log(', '.join(log_info))
                writer.flush()


def save(config, model, save_path, name):
    # print("model name = ", config['model']['name'])
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="/root/SAM_Adapter_MAML/configs/FMAM-no-meta.yaml")
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
