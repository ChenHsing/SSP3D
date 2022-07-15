# -*- coding: utf-8 -*-

import os
import logging
import random
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.helpers

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from core.test_shapenet import test_net
from models.encoder import Encoder
from models.decoder_3d import Decoder
from models.Attention import Attention
from models.discriminator import Discriminator
from utils.average_meter import AverageMeter
import pickle


def readprior(shapenet_dir):
    shapenet = open(shapenet_dir, 'rb')
    shapenet = pickle.load(shapenet)
    return shapenet


def getprior(sample_names, cfg):
    prior_dic = readprior(cfg.DATASETS.PROTOTYPE_PATH)
    lis = []
    for name in sample_names:
        tmp = prior_dic[name]
        lis.append(tmp)
    ret = torch.stack(lis,dim=0)
    ret = torch.FloatTensor(ret.float())
    return ret


def train_net(cfg):
    torch.backends.cudnn.benchmark = True
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
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, strong_transforms),
                                                    batch_size=cfg.CONST.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKER,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
        utils.data_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms),
                                                  batch_size=16,
                                                  num_workers=cfg.CONST.NUM_WORKER,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Set up networks
    encoder = Encoder()
    decoder = Decoder()
    attention = Attention(hidden_size=2048, num_attention_heads=2)
    disc = Discriminator()

    logging.debug('Parameters in Encoder: %d.' % (utils.helpers.count_parameters(encoder)))
    logging.debug('Parameters in Decoder: %d.' % (utils.helpers.count_parameters(decoder)))
    logging.debug('Parameters in attention: %d.' % (utils.helpers.count_parameters(attention)))



    # Initialize weights of networks
    encoder.apply(utils.helpers.init_weights)
    decoder.apply(utils.helpers.init_weights)
    attention.apply(utils.helpers.init_weights)
    disc.apply(utils.helpers.init_weights)


    # Set up solver
    if cfg.TRAIN.POLICY == 'adam':
        encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()),
                                          lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS)
        decoder_solver = torch.optim.Adam(decoder.parameters(),
                                          lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS)
        attention_solver = torch.optim.Adam(attention.parameters(),
                                          lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS)
        disc_solver = torch.optim.Adam(disc.parameters(),
                                       lr=1e-3, betas=cfg.TRAIN.BETAS)

    elif cfg.TRAIN.POLICY == 'sgd':
        encoder_solver = torch.optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM)
        decoder_solver = torch.optim.SGD(decoder.parameters(),
                                         lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM)
        attention_solver = torch.optim.SGD(attention.parameters(),
                                         lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM)
        disc_solver = torch.optim.SGD(disc.parameters(),
                                      lr=1e-4, momentum=cfg.TRAIN.MOMENTUM)

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
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        attention = attention.cuda()
        disc = disc.cuda()

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Load pretrained model if exists
    init_epoch = 0
    best_iou = -1
    best_epoch = -1
    if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        init_epoch = checkpoint['epoch_idx']
        best_iou = checkpoint['best_iou']
        best_epoch = checkpoint['best_epoch']

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

        logging.info('Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' %
                     (init_epoch, best_iou, best_epoch))

    # Summary writer for TensorBoard
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())
    cfg.DIR.LOGS = output_dir % 'logs'
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

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
        encoder.train()
        decoder.train()
        attention.train()
        disc.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)
        for batch_idx, (taxonomy_names, sample_names, rendering_images,
                        ground_truth_volumes) in enumerate(train_data_loader):
            # Measure data time
            data_time.update(time() - batch_end_time)
            prior = getprior(taxonomy_names,cfg)

            # Get data from data loader
            rendering_images = utils.helpers.var_or_cuda(rendering_images)
            ground_truth_volumes = utils.helpers.var_or_cuda(ground_truth_volumes)
            prior = utils.helpers.var_or_cuda(prior)

            #train generator
            encoder_solver.zero_grad()
            decoder_solver.zero_grad()
            attention_solver.zero_grad()

            for param in disc.parameters():
                param.requires_grad = False

            # Train the encoder, decoder
            image_features = encoder(rendering_images)
            prior_features = attention(image_features, prior)
            raw_features, generated_volumes = decoder(image_features, prior_features)
            generated_volumes = torch.mean(generated_volumes, dim=1)
            encoder_loss = bce_loss(generated_volumes, ground_truth_volumes) * 10

            #disc loss
            ground_label = torch.ones(ground_truth_volumes.size(0), 1).cuda()
            predict_label = torch.zeros(generated_volumes.size(0), 1).cuda()
            # train discriminator
            pr_data = disc(generated_volumes)
            fake_loss = bce_loss(pr_data, ground_label) * 0.0001
            generate_loss = encoder_loss +fake_loss
            generate_loss.backward()

            encoder_solver.step()
            decoder_solver.step()
            attention_solver.step()

            ## train dis
            for param in disc.parameters():
                param.requires_grad = True

            with torch.no_grad():
                image_features = encoder(rendering_images)
                prior_features = attention(image_features, prior)
                raw_features, generated_volumes = decoder(image_features, prior_features)
                generated_volumes = torch.mean(generated_volumes, dim=1)

            gt_data = disc(ground_truth_volumes)
            pr_data = disc(generated_volumes.detach())

            true_loss = bce_loss(pr_data, predict_label) + bce_loss(gt_data, ground_label)
            true_loss = 0.0001 * true_loss

            # Gradient decent
            disc.zero_grad()
            true_loss.backward()
            disc_solver.step()

            # Append loss to average metrics
            encoder_losses.update(generate_loss.item())

            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('EncoderDecoder/BatchLoss', encoder_loss.item(), n_itr)

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

        # Update Rendering Views
        # Validate the training models
        iou = test_net(cfg, epoch_idx + 1, val_data_loader, val_writer, encoder, decoder,attention)

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
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'attention_state_dict': attention.state_dict(),
                'disc_state_dict': disc.state_dict()
            }

            torch.save(checkpoint, output_path)
            logging.info('Saved checkpoint to %s ...' % output_path)

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()
