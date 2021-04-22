import os
from os.path import join
from tqdm import tqdm
import time
import torch
import torch.nn as nn
from importlib import import_module
from .optimizer import Optimizer
from model import Model
from data import TransitionDataset, GestureDataset, Data
import random
import numpy as np
from utils.logger import Logger
from datetime import datetime


class Trainer(object):
    def __init__(self, para):
        self.para = para

    def run(self):
        # recoding parameters
        self.para.time = datetime.now()
        logger = Logger(self.para)
        logger.record_para()
        if not self.para.test_only:
            proc(self.para)
        # setting for test
        self.para.test_state = True
        test_cmhad(self.para, logger)


def proc(para):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # set random seed
    torch.manual_seed(para.seed)
    torch.cuda.manual_seed(para.seed)
    random.seed(para.seed)
    np.random.seed(para.seed)

    # create logger
    logger = Logger(para)

    # create model
    logger('building {} model ...'.format(para.model), prefix='\n')
    model = Model(para).model
    model.cuda()
    logger('model structure:', model, verbose=False)

    # create criterion according to the loss function
    loss_name = para.loss
    module = import_module('train.loss')
    criterion = getattr(module, loss_name)().cuda()

    # create optimizer
    opt = Optimizer(para, model)

    # data parallel
    model = nn.DataParallel(model)

    # create dataloader
    logger('loading {} dataloader ...'.format(para.dataset), prefix='\n')
    data = Data(para)
    train_loader = data.dataloader_train
    valid_loader = data.dataloader_valid

    # optionally resume from a checkpoint
    if para.resume:
        if os.path.isfile(para.resume_file):
            checkpoint = torch.load(para.resume_file, map_location=lambda storage, loc: storage.cuda(0))
            logger('loading checkpoint {} ...'.format(para.resume_file))
            logger.register_dict = checkpoint['register_dict']
            para.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            opt.optimizer.load_state_dict(checkpoint['optimizer'])
            opt.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            logger('no check point found at {}'.format(para.resume_file))

    # training and validation
    for epoch in range(para.start_epoch, para.end_epoch + 1):
        train(train_loader, model, criterion, opt, epoch, para, logger)
        valid(valid_loader, model, criterion, epoch, para, logger)

        # save checkpoint
        is_best = logger.is_best(epoch)
        checkpoint = {
            'epoch': epoch + 1,
            'model': para.model,
            'state_dict': model.state_dict(),
            'register_dict': logger.register_dict,
            'optimizer': opt.optimizer.state_dict(),
            'scheduler': opt.scheduler.state_dict()
        }
        logger.save(checkpoint, is_best)


def train(train_loader, model, criterion, opt, epoch, para, logger):
    model.train()
    logger('[Epoch {} / lr {:.2e}]'.format(
        epoch, opt.get_lr()
    ), prefix='\n')
    loss_meter = AverageMeter()
    batchtime_meter = AverageMeter()
    start = time.time()
    end = time.time()
    pbar = tqdm(total=len(train_loader), ncols=80)
    for inputs, labels in train_loader:
        # forward
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss_meter.update(loss.detach().item(), inputs.size(0))
        # backward and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()
        # measure elapsed time
        batchtime_meter.update(time.time() - end)
        end = time.time()
        pbar.update(para.batch_size)
    pbar.close()
    # record info
    logger.register(para.loss + '_train', epoch, loss_meter.avg)
    # show info
    logger('[train] epoch time: {:.2f}s, average batch time: {:.2f}s'.format(end - start, batchtime_meter.avg),
           timestamp=False)
    logger.report([[para.loss, 'min'], ], state='train', epoch=epoch)
    # adjust learning rate
    opt.lr_schedule()


def valid(valid_loader, model, criterion, epoch, para, logger):
    model.eval()
    loss_meter = AverageMeter()
    batchtime_meter = AverageMeter()
    start = time.time()
    end = time.time()
    pbar = tqdm(total=len(valid_loader), ncols=80)
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_meter.update(loss.detach().item(), inputs.size(0))
            batchtime_meter.update(time.time() - end)
            end = time.time()
            pbar.update(para.batch_size)
    pbar.close()
    # record info
    logger.register(para.loss + '_valid', epoch, loss_meter.avg)
    # show info
    logger('[valid] epoch time: {:.2f}s, average batch time: {:.2f}s'.format(end - start, batchtime_meter.avg),
           timestamp=False)
    logger.report([[para.loss, 'min'], ], state='valid', epoch=epoch)


def test_cmhad(para, logger):
    if para.dataset == 'CMHAD_Transition':
        dataset = TransitionDataset()
        threshold = 25
    elif para.dataset == 'CMHAD_Gesture':
        dataset = GestureDataset()
        threshold = 15
    model = Model(para).model.cuda()
    model.only_last_layer = True
    if para.test_only:
        checkpoint = torch.load(para.test_checkpoint)
    else:
        checkpoint = torch.load(join(logger.save_dir, 'checkpoint.pth.tar'))
    model = nn.parallel.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    positive, prediction, true_positive, false_positive = [], [], [], []
    trial = 10
    with torch.no_grad():
        for subject in range(1, 13):
            start_frame = 0
            end_flag = False
            x, y = dataset.load_full_trial(subject, trial)
            y_pred = [[] for i in range(len(y))]
            while True:
                inputs = x[start_frame:start_frame + dataset.seq_length].unsqueeze(dim=0)
                outputs = model(inputs).squeeze(dim=0).permute(1, 0)
                _, labels = torch.max(outputs, dim=1)
                for i in range(start_frame, start_frame + dataset.seq_length):
                    y_pred[i].append(int(labels[i - start_frame].cpu().numpy()))
                start_frame += dataset.seq_skip
                if end_flag:
                    break
                if start_frame + dataset.seq_length >= len(y):
                    start_frame -= start_frame + dataset.seq_length - len(y)
                    end_flag = True
            y = y.tolist()
            y_hat = [max(item, key=item.count) for item in y_pred]
            temp_positive, temp_prediction, temp_true_positive, temp_false_positive = collect_statistics(y, y_hat,
                                                                                                         threshold)
            print(
                'Subject #{}\n{:<20}{}\n{:<20}{}\n{:<20}{}\n{:<20}{}\n'.format(
                    subject,
                    'actions:',
                    [y[item] for item in temp_positive],
                    'predictions:',
                    [y_hat[item] for item in temp_prediction],
                    'true predictions:',
                    [y_hat[item] for item in temp_true_positive],
                    'false predictions:',
                    [y_hat[item] for item in temp_false_positive])
            )
            positive += temp_positive
            prediction += temp_prediction
            true_positive += temp_true_positive
            false_positive += temp_false_positive
        recall = len(true_positive) / len(positive)
        if recall > 1:  # in the case of multiple true_positive within the same action time-span
            recall = 1.
        precision = len(true_positive) / len(prediction)
        f1 = 2 * precision * recall / (precision + recall)
        msg = '{} -- recall: {}, precision: {}, f1: {}'.format(para.dataset, recall, precision, f1)
        print(msg)


def collect_statistics(y, y_hat, threshold=25):
    positive = []
    prediction = []
    true_positive = []
    false_positive = []
    # generate index lists of prediction, true positive and false positive
    start_flag = False
    start_idx = 0
    end_idx = 0
    for i in range(len(y_hat)):
        if y_hat[i] > 0 and start_flag is False:
            start_idx = i
            start_flag = True
        if start_flag:
            if y_hat[i] == y_hat[start_idx] and i != len(y_hat) - 1:
                end_idx = i
            else:
                start_flag = False
                if i == len(y_hat) - 1:
                    end_idx = i
                if end_idx - start_idx >= threshold:
                    idx = int((end_idx - start_idx) / 2) + start_idx
                    prediction.append(idx)
                    if y_hat[idx] == y[idx]:
                        true_positive.append(idx)
                    else:
                        false_positive.append(idx)
    # generate index lists of positive
    start_flag = False
    start_idx = 0
    end_idx = 0
    for i in range(len(y)):
        if y[i] > 0 and start_flag is False:
            start_idx = i
            start_flag = True
        if start_flag:
            if y[i] == y[start_idx] and i != len(y_hat) - 1:
                end_idx = i
            else:
                if i == len(y) - 1:
                    end_idx = i
                start_flag = False
                idx = int((end_idx - start_idx) / 2) + start_idx
                positive.append(idx)

    assert len(prediction) == len(true_positive) + len(false_positive)

    return positive, prediction, true_positive, false_positive


# computes and stores the average and current value
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
