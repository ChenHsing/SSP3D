# -*- coding: utf-8 -*-

import os
import logging
import random
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders_un
import utils.data_transforms
import utils.helpers

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from core.test_pix3d import test_net
from models.encoder import Encoder
from models.decoder_c import Decoder
from models.Attention import Attention
from models.discriminatorpix import Discriminator
from utils.average_meter import AverageMeter
from collections import OrderedDict
import pickle
import math


def readprior(shapenet_dir):
    shapenet = open(shapenet_dir, 'rb')
    shapenet = pickle.load(shapenet)
    return shapenet

def getprior(sample_names,cfg):
    prior_dic = readprior(cfg.DATASETS.PROTOTYPE_PATH)
    lis = []
    for name in sample_names:
        tmp = prior_dic[name]
        lis.append(tmp)
    ret = torch.stack(lis,dim=0)
    ret = torch.FloatTensor(ret.float())
    return ret


momentum = 0.9996
tot_step = 30

@torch.no_grad()
def _update_teacher_model(now_step, encoder, decoder, attention, encoder_tea, decoder_tea, attention_tea, keep_rate=0.996):

    now_momentum = 1 - (1 - momentum) * (math.cos(math.pi * now_step / tot_step) + 1) / 2
    keep_rate = now_momentum

    student_encoder = encoder.state_dict()
    student_decoder = decoder.state_dict()
    student_attention = attention.state_dict()
    new_teacher_encoder = OrderedDict()
    new_teacher_decoder = OrderedDict()
    new_teacher_attention = OrderedDict()
    for key, value in encoder_tea.state_dict().items():
        if key in student_encoder.keys():
            new_teacher_encoder[key] = (
                    student_encoder[key] *
                    (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))
    for key, value in decoder_tea.state_dict().items():
        if key in student_decoder.keys():
            new_teacher_decoder[key] = (
                    student_decoder[key] *
                    (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))
    for key, value in attention_tea.state_dict().items():
        if key in student_attention.keys():
            new_teacher_attention[key] = (
                    student_attention[key] *
                    (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))
    encoder_tea.load_state_dict(new_teacher_encoder)
    decoder_tea.load_state_dict(new_teacher_decoder)
    attention_tea.load_state_dict(new_teacher_attention)
    return encoder_tea, decoder_tea, attention_tea




def finetune_net(cfg):
    torch.backends.cudnn.benchmark = True

    # Set up data augmentation
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W

    strong_transforms = utils.data_transforms.Compose([
        utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
        utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.RandomFlip(),
        utils.data_transforms.RandomPermuteRGB(),
        utils.data_transforms.ToTensor(),
    ])
    slight_transforms = utils.data_transforms.Compose([
        utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])
    val_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up data loader
    train_dataset_loader = utils.data_loaders_un.DATASET_LOADER_MAPPING['Pix3D'](cfg)
    val_dataset_loader = utils.data_loaders_un.DATASET_LOADER_MAPPING['Pix3D'](cfg)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders_un.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, strong_transforms, slight_transforms),
                                                    batch_size=cfg.CONST.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKER,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
        utils.data_loaders_un.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms, val_transforms),
                                                  batch_size=1,
                                                  num_workers=cfg.CONST.NUM_WORKER,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Set up networks
    encoder_stu = Encoder()
    decoder_stu = Decoder()
    attention_stu = Attention(2048, 2)
    disc = Discriminator()
    encoder_tea = Encoder()
    decoder_tea = Decoder()
    attention_tea = Attention(2048, 2)

    logging.debug('Parameters in Encoder_stu: %d.' % (utils.helpers.count_parameters(encoder_stu)))
    logging.debug('Parameters in Decoder_stu: %d.' % (utils.helpers.count_parameters(decoder_stu)))
    logging.debug('Parameters in Encoder_att: %d.' % (utils.helpers.count_parameters(attention_stu)))

    logging.debug('Parameters in Encoder_tea: %d.' % (utils.helpers.count_parameters(encoder_tea)))
    logging.debug('Parameters in Decoder_tea: %d.' % (utils.helpers.count_parameters(decoder_tea)))
    logging.debug('Parameters in Decoder_att: %d.' % (utils.helpers.count_parameters(attention_tea)))

    # Initialize weights of networks
    encoder_stu.apply(utils.helpers.init_weights)
    decoder_stu.apply(utils.helpers.init_weights)
    attention_stu.apply(utils.helpers.init_weights)
    disc.apply(utils.helpers.init_weights)


    # Set up solver
    if cfg.TRAIN.POLICY == 'adam':
        encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder_stu.parameters()),
                                          lr=5e-4,
                                          betas=cfg.TRAIN.BETAS)
        decoder_solver = torch.optim.Adam(decoder_stu.parameters(),
                                          lr=5e-4,
                                          betas=cfg.TRAIN.BETAS)
        attention_solver = torch.optim.Adam(attention_stu.parameters(),
                                          lr=5e-4,
                                          betas=cfg.TRAIN.BETAS)

    elif cfg.TRAIN.POLICY == 'sgd':
        encoder_solver = torch.optim.SGD(filter(lambda p: p.requires_grad, encoder_stu.parameters()),
                                         lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM)
        decoder_solver = torch.optim.SGD(decoder_stu.parameters(),
                                         lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM)
        attention_solver = torch.optim.SGD(attention_stu.parameters(),
                                         lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

    # Set up learning rate scheduler to decay learning rates dynamically
    encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_solver,
                                                                milestones=cfg.TRAIN.ENCODER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)
    decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_solver,
                                                                milestones=cfg.TRAIN.DECODER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)
    attention_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(attention_solver,
                                                                milestones=cfg.TRAIN.DECODER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)

    if torch.cuda.is_available():
        encoder_stu = encoder_stu.cuda()
        decoder_stu = decoder_stu.cuda()
        attention_stu = attention_stu.cuda()
        disc = disc.cuda()
        encoder_tea = encoder_tea.cuda()
        decoder_tea = decoder_tea.cuda()
        attention_tea = attention_tea.cuda()


    # Set up loss functions
    bce_loss = torch.nn.BCELoss()
    l2_loss = torch.nn.MSELoss()


    # Load pretrained model if exists
    init_epoch = 0
    best_iou = -1
    best_epoch = -1
    if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        if torch.cuda.is_available():
            checkpoint = torch.load(cfg.CONST.WEIGHTS)
        else:
            checkpoint = torch.load(cfg.CONST.WEIGHTS, map_location=torch.device('cpu'))
        init_epoch = 0
        best_iou = checkpoint['best_iou']
        best_epoch = checkpoint['best_epoch']

        encoder_tea.load_state_dict(checkpoint['encoder_state_dict'])
        decoder_tea.load_state_dict(checkpoint['decoder_state_dict'])
        attention_tea.load_state_dict(checkpoint['attention_state_dict'])
        disc.load_state_dict(checkpoint['disc_state_dict'])
        encoder_stu.load_state_dict(checkpoint['encoder_state_dict'])
        decoder_stu.load_state_dict(checkpoint['decoder_state_dict'])
        attention_stu.load_state_dict(checkpoint['attention_state_dict'])

        logging.info('Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' %
                     (init_epoch, best_iou, best_epoch))

    # Summary writer for TensorBoard
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())
    cfg.DIR.LOGS = output_dir % 'logs'
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    total_step = 0

    # Training loop
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHS):
        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        batch_time = AverageMeter()
        data_time = AverageMeter()
        encoder_losses = AverageMeter()
        refiner_losses = AverageMeter()

        # switch models to training mode
        encoder_stu.train()
        decoder_stu.train()
        attention_stu.train()
        disc.eval()
        encoder_tea.eval()
        decoder_tea.eval()
        attention_tea.eval()

        batch_end_time = time()
        n_batches = len(train_data_loader)
        for batch_idx, (taxonomy_names, sample_names, rendering_images, unlabel_name,
                        rendering_unlabel_slight, rendering_unlabel_strong,
                        ground_truth_volumes) in enumerate(train_data_loader):
            # Measure data time
            data_time.update(time() - batch_end_time)
            prior = getprior(taxonomy_names, cfg)
            un_prior = getprior(unlabel_name, cfg)

            # Get data from data loader
            rendering_images = utils.helpers.var_or_cuda(rendering_images)
            rendering_unlabel_slight = utils.helpers.var_or_cuda(rendering_unlabel_slight)
            rendering_unlabel_strong = utils.helpers.var_or_cuda(rendering_unlabel_strong)
            ground_truth_volumes = utils.helpers.var_or_cuda(ground_truth_volumes)
            prior = utils.helpers.var_or_cuda(prior)
            un_prior = utils.helpers.var_or_cuda(un_prior)

            label_features = encoder_stu(rendering_images)
            prior_features = attention_stu(label_features, prior)
            _,generated_label = decoder_stu(label_features, prior_features)
            generated_label = torch.mean(generated_label,dim=1)
            loss_supervised = bce_loss(generated_label,ground_truth_volumes) * 10

            # Train the encoder, decoder
            unlabel_features = encoder_stu(rendering_unlabel_strong)
            unlabel_prior_features = attention_stu(unlabel_features, un_prior)
            _, generated_unlabel = decoder_stu(unlabel_features, unlabel_prior_features)
            generated_unlabel = torch.mean(generated_unlabel, dim=1)

            with torch.no_grad():
                unlabel_features_tea = encoder_tea(rendering_unlabel_slight)
                unlabel_prior_features_tea = attention_tea(unlabel_features_tea,un_prior)
                _, generated_unlabel_tea = decoder_tea(unlabel_features_tea,unlabel_prior_features_tea)
                generated_unlabel_tea = torch.mean(generated_unlabel_tea, dim=1)
                generated_unlabel_tea = torch.ge(generated_unlabel_tea, 0.4).float()
                score = disc(generated_unlabel_tea)
                score = torch.mean(score,dim=0)

            loss_unsupervised = l2_loss(generated_unlabel, generated_unlabel_tea) * 10 * score
            loss = loss_supervised + 5 * loss_unsupervised

            # Gradient decent
            encoder_stu.zero_grad()
            decoder_stu.zero_grad()
            attention_stu.zero_grad()

            loss.backward()
            encoder_solver.step()
            decoder_solver.step()
            attention_solver.step()

            # Append loss to average metrics
            encoder_losses.update(loss.item())

            # Append loss to TensorBoard
            total_step += 1
            encoder_tea, decoder_tea, attention_tea = _update_teacher_model(epoch_idx,encoder_stu, decoder_stu, attention_stu,
                                                             encoder_tea, decoder_tea, attention_tea, 0.99996)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()

        # Adjust learning rate
        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()
        attention_lr_scheduler.step()

        # Append epoch loss to TensorBoard
        train_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx + 1)
        train_writer.add_scalar('Refiner/EpochLoss', refiner_losses.avg, epoch_idx + 1)

        # Tick / tock
        epoch_end_time = time()
        logging.info('[Epoch %d/%d] EpochTime = %.3f (s) EDLoss = %.4f RLoss = %.4f' %
                     (epoch_idx + 1, cfg.TRAIN.NUM_EPOCHS, epoch_end_time - epoch_start_time, encoder_losses.avg,
                      refiner_losses.avg))
        iou = test_net(cfg, epoch_idx + 1, val_data_loader, val_writer, encoder_stu, decoder_stu, attention_stu)
        # Save weights to file
        if iou > best_iou:
            file_name = 'checkpoint-epoch-%03d.pth' % (epoch_idx + 1)
            if iou > best_iou:
                best_iou = iou
                best_epoch = epoch_idx
                file_name = 'checkpoint-best.pth'

            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            if not os.path.exists(cfg.DIR.CHECKPOINTS):
                os.makedirs(cfg.DIR.CHECKPOINTS)

            checkpoint = {
                'epoch_idx': epoch_idx,
                'best_iou': best_iou,
                'best_epoch': best_epoch,
                'encoder_state_dict': encoder_stu.state_dict(),
                'decoder_state_dict': decoder_stu.state_dict(),
                'attention_state_dict': attention_stu.state_dict()
            }

            torch.save(checkpoint, output_path)
            logging.info('Saved checkpoint to %s ...' % output_path)

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()
