# -*- coding: utf-8 -*-

import json
import numpy as np
import logging
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.helpers
import os
import matplotlib.pylab as plt
import matplotlib
matplotlib.use('Agg')
from models.encoder import Encoder
from models.decoder_c import Decoder
from models.Attention import Attention
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


def showvoxel(vox, name, classid, trueid):
    vox = vox.squeeze().__ge__(0.1)
    vox = vox.detach().cpu().numpy()


    fig1 = plt.figure(name, figsize=(20, 20))
    ax1 = fig1.gca(projection='3d')
    ax1.voxels(vox, edgecolor="#6e6e6e", facecolors='#F5F5F5')

    ax1.grid(False)  # 默认True，风格线。
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    plt.axis('off')
    if not os.path.exists('ours/' + classid + "/" + trueid + "/"):
        os.makedirs('ours/' + classid + "/" + trueid + "/")

    plt.savefig('ours/' + classid + "/" + trueid + "/" + name + '.png')
    plt.clf()


def showvoxel_back(vox, name, classid, trueid):
    vox = vox.squeeze().__ge__(0.1)
    vox = vox.detach().cpu().numpy()


    fig1 = plt.figure(name, figsize=(20, 20))
    ax1 = fig1.gca(projection='3d')
    ax1.view_init(elev=30., azim=60)
    ax1.voxels(vox, edgecolor="#6e6e6e", facecolors='#F5F5F5')

    ax1.grid(False)  # 默认True，风格线。
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    plt.axis('off')
    if not os.path.exists('ours/' + classid + "/" + trueid + "/"):
        os.makedirs('ours/' + classid + "/" + trueid + "/")

    plt.savefig('ours/' + classid + "/" + trueid + "/" + name + '_back.png')
    plt.clf()


def test_net(cfg,
             epoch_idx=-1,
             test_data_loader=None,
             test_writer=None,
             encoder=None,
             decoder=None,
             attention=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Load taxonomies of dataset
    taxonomies = []
    with open(cfg.DATASETS['Pix3D'.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
                                                       batch_size=1,
                                                       num_workers=cfg.CONST.NUM_WORKER,
                                                       pin_memory=True,
                                                       shuffle=False)

    # Set up networks
    if decoder is None or encoder is None:
        encoder = Encoder()
        decoder = Decoder()
        attention = Attention(2048, 2, 0.1)

        encoder = encoder.cuda()
        decoder = decoder.cuda()
        attention = attention.cuda()

        logging.info('Loading weights from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        attention.load_state_dict(checkpoint['attention_state_dict'])

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = dict()
    encoder_losses = AverageMeter()
    refiner_losses = AverageMeter()

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()

    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume) in enumerate(test_data_loader):
        taxonomy_name = taxonomy_id
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]

        with torch.no_grad():
            # Get data from data loader
            prior = getprior(taxonomy_name, cfg)

            rendering_images = utils.helpers.var_or_cuda(rendering_images)
            ground_truth_volume = utils.helpers.var_or_cuda(ground_truth_volume)
            prior = utils.helpers.var_or_cuda(prior)

            # Test the encoder, decoder, refiner and merger
            image_features = encoder(rendering_images)
            prior_features = attention(image_features,prior)
            raw_features, generated_volume = decoder(image_features,prior_features)
            generated_volume = torch.mean(generated_volume, dim=1)
            encoder_loss = bce_loss(generated_volume, ground_truth_volume) * 10
            # Append loss and accuracy to average metrics
            encoder_losses.update(encoder_loss.item())

            # IoU per sample
            sample_iou = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(generated_volume, th).float()
                intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                sample_iou.append((intersection / union).item())

            if sample_name in dic[taxonomy_id]:
                showvoxel(generated_volume,'out',taxonomy_id,sample_name)
                showvoxel_back(generated_volume,'outback',taxonomy_id,sample_name)
                showvoxel(ground_truth_volume,'gt',taxonomy_id,sample_name)
                showvoxel_back(ground_truth_volume,'gt_back',taxonomy_id,sample_name)

            # IoU per taxonomy
            if taxonomy_id not in test_iou:
                test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
            test_iou[taxonomy_id]['n_samples'] += 1
            test_iou[taxonomy_id]['iou'].append(sample_iou)

            # Print sample loss and IoU


    # Output testing results
    mean_iou = []
    for taxonomy_id in test_iou:
        test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
        mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
    mean_iou = np.sum(mean_iou, axis=0) / n_samples

    # Print header
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    print('Baseline', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()
    # Print body
    for taxonomy_id in test_iou:
        print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
        print('%d' % test_iou[taxonomy_id]['n_samples'], end='\t')
        if 'baseline' in taxonomies[taxonomy_id]:
            print('%.4f' % taxonomies[taxonomy_id]['baseline']['%d-view' % cfg.CONST.N_VIEWS_RENDERING], end='\t\t')
        else:
            print('N/a', end='\t\t')

        for ti in test_iou[taxonomy_id]['iou']:
            print('%.4f' % ti, end='\t')
        print()
    # Print mean IoU for each threshold
    print('Overall ', end='\t\t\t\t')
    for mi in mean_iou:
        print('%.4f' % mi, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    max_iou = np.max(mean_iou)
    if test_writer is not None:
        test_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx)
        test_writer.add_scalar('Refiner/EpochLoss', refiner_losses.avg, epoch_idx)
        test_writer.add_scalar('Refiner/IoU', max_iou, epoch_idx)

    return max_iou
