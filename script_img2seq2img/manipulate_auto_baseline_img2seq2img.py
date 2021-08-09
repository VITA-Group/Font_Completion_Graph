import os
import sys
import torch
import pickle
import collections
import numpy as np
import itertools as it
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from common.arguments import args, dataloader_configs, transform_eval
from common.Trainer_img2seq2img import TrainerIMG_img2seq2img as TrainerMGT
from common.ret_utils import get_scores, vis_retrieval
from utilities.Logger import Logger

from dataset.FontImgDatasetBase import FontDatasetIMG
from dataset.FontSeqDatasets import FontDatasetMGTFC as FontDataset
from models.img_encoder import ImgEncoderConv, Classifier_only
from models.img_decoder import ImgDecoder
from models.seq_encoder import SeqEncoderConv
from models.seq_decoder import SeqDecoder

args.feed_forward_dim = 128
args.feed_forward_dim_extra = 512
args.test_subset = 'gallery'
args.resume = '/home/alternative/font_exps/checkpoints/train_baseline_img2seq_rewind_condConv_XE_AE_20201116051730/train_baseline_img2seq_20.pth'
args.resume_extra = '/home/alternative/font_exps/checkpoints/train_baseline_seq2img_rewind_condConv_XE_AE_COS_20201113052727/train_baseline_seq2img_10.pth'

# -----------------------------------------------------------------------------------------------------
# Part 2. configurations
# Part 2-1. log configuration
args.exp = '_'.join([args.resume.split('/')[-2].replace('train', 'man').replace('img2seq', 'img2seq2img'),
                     args.resume.split('_')[-1][:-4],
                     args.resume_extra.split('/')[-2].split('_')[-1],
                     args.resume_extra.split('_')[-1][:-4],
                     args.test_subset])
print(args.exp)

exp_log_dir = os.path.join(args.save_path, "log", args.exp)
exp_manipulate_dir = os.path.join(args.save_path, "manipulate", args.exp)

os.makedirs(exp_log_dir, exist_ok=True)
os.makedirs(exp_manipulate_dir, exist_ok=True)
print('\n', exp_manipulate_dir, '\n')

logger = Logger(os.path.join(exp_log_dir, args.exp + ".log")).get_logger()
logger.info("argument parser settings: {}".format(args))

# Part 2-4. configurations for loss function, model, and optimizer
model_configs = collections.OrderedDict()
model_configs['output_dim'] = args.output_dim
model_configs['output_dim_extra'] = args.output_dim_extra
model_configs['nPoints'] = args.nPoints
model_configs['coord_input_dim'] = args.coord_input_dim
model_configs['quant_input_dim'] = args.quant_input_dim
model_configs['flag_input_dim'] = args.flag_input_dim
model_configs['input_embed_dim'] = args.input_embed_dim
model_configs['char_embed_dim'] = args.char_embed_dim
model_configs['input_embed_dim_extra'] = (
    args.input_embed_dim_extra if args.input_embed_dim_extra else args.input_embed_dim)

model_configs['n_heads'] = args.n_heads
model_configs['n_layers'] = args.n_layers
model_configs['feed_forward_dim'] = args.feed_forward_dim
model_configs['feed_forward_dim_extra'] = (
    args.feed_forward_dim_extra if args.feed_forward_dim_extra else args.feed_forward_dim)
model_configs['normalization'] = args.normalization
model_configs['dropout'] = args.dropout

img_encoder = ImgEncoderConv(n_classes=model_configs['output_dim'],
                             input_embed_dim=model_configs['input_embed_dim'],
                             feed_forward_dim=model_configs['feed_forward_dim'],
                             dropout=model_configs['dropout']).cuda()

seq_decoder = SeqDecoder(feed_forward_dim=model_configs['feed_forward_dim'],
                         nPoints=model_configs['nPoints'],
                         coord_input_dim=model_configs['coord_input_dim'],
                         quant_input_dim=model_configs['quant_input_dim'],
                         flag_input_dim=model_configs['flag_input_dim'],
                         char_embed_dim=model_configs['char_embed_dim'],
                         char_embedded=args.char_embedded,
                         conditional_conv=args.conditional_conv,
                         conditional_BN=args.conditional_BN,
                         mgt_decoder=args.mgt_decoder,
                         quant_decoder=args.quantizeXE_loss,
                         flag_decoder=args.flagXE_loss,
                         nChars=args.nChars).cuda()

seq_encoder = SeqEncoderConv(n_classes=model_configs['output_dim_extra'],
                             coord_input_dim=model_configs['coord_input_dim'],
                             quant_input_dim=model_configs['quant_input_dim'],
                             feat_dict_size=model_configs['nPoints'],
                             load_quantize=args.quantize,
                             n_layers=model_configs['n_layers'], n_heads=model_configs['n_heads'],
                             input_embed_dim=model_configs['input_embed_dim_extra'],
                             feed_forward_dim=model_configs['feed_forward_dim_extra'],
                             normalization=model_configs['normalization'],
                             dropout=model_configs['dropout']).cuda()

img_decoder = ImgDecoder(feed_forward_dim=model_configs['feed_forward_dim_extra'],
                         char_embed_dim=model_configs['char_embed_dim'],
                         char_embedded=args.char_embedded,
                         conditional_conv=args.conditional_conv,
                         conditional_BN=args.conditional_BN,
                         nChars=args.nChars).cuda()

img2seq_model = {'encoder': img_encoder, 'decoder': seq_decoder}
seq2img_model = {'encoder': seq_encoder, 'decoder': img_decoder}

if args.resume:
    checkpoint = torch.load(args.resume)
    img2seq_model['encoder'].load_state_dict(checkpoint["encoder"])
    img2seq_model['decoder'].load_state_dict(checkpoint["decoder"])

if args.resume_extra:
    checkpoint_extra = torch.load(args.resume_extra)
    seq2img_model['encoder'].load_state_dict(checkpoint_extra["encoder"])
    seq2img_model['decoder'].load_state_dict(checkpoint_extra["decoder"])

logger.info("model configuration settings: {}".format(model_configs))

# Part 2-3. dataloader instantiation
qdataset = FontDataset(args.test_subset, dataloader_configs[args.test_subset],
                       args.label_retrieval, args.char_idx, args.nPoints,
                       args.coord_input_dim, args.padding,
                       residual=args.residual, quantize=args.quantize,
                       shorten=args.shorten, miss_seq=args.miss_ratio,
                       load_img=True, data_transforms=transform_eval, pair=False, DA=False)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
logger.info("dataloader configuration settings: {}".format(dataloader_configs))


def plot_points(points, endpoints, fname):
    # color_palette = ['red', 'green', 'yellow', 'blue']

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 8)

    for n in range(len(endpoints) - 1):
        plt.scatter(points[endpoints[n]:endpoints[n + 1], 0],
                    points[endpoints[n]:endpoints[n + 1], 1] * -1.,
                    # color=color_palette[n % 4],
                    s=100)
        # plt.savefig(fname + '_%s_%s' % (endpoints[n], endpoints[n + 1]) + '.png', dpi=100, bbox_inches='tight')

    plt.xlim(-25, 275)
    plt.ylim(-275, 25)
    plt.axis('off')
    plt.savefig(fname + '.png', dpi=100, bbox_inches='tight')
    plt.close()


def manipulate_points(input, endpoints):
    output = input.copy()

    # p0 = input[0, :2]
    p1 = input[endpoints[1], :2]
    p14 = input[endpoints[14], :2]

    p3 = input[endpoints[3], :2]
    # p4 = input[endpoints[4], :2]
    p5 = input[endpoints[5], :2]

    l1 = [p14[1] * (1. - 1. / endpoints[1] * n) + p1[1] * 1. / endpoints[1] * n for n in range(endpoints[1])]
    output[0:endpoints[1], 0] = p1[0]
    output[0:endpoints[1], 1] = np.array(l1)
    output[0:endpoints[1], 2:] = np.array([0, -1])

    l2 = [p3[1] * (1. - 1. / endpoints[1] * n) + p5[1] * 1. / endpoints[1] * n for n in range(endpoints[1])]
    output[endpoints[3]:endpoints[4], 0] = p3[0]
    output[endpoints[3]:endpoints[4], 1] = np.array(l2)
    output[endpoints[3]:endpoints[4], 2:] = np.array([0, -1])

    l3 = [p3[0] * (1. - 1. / endpoints[1] * n) + p5[0] * 1. / endpoints[1] * n for n in range(endpoints[1])]
    output[endpoints[4]:endpoints[5], 0] = np.array(l3)
    output[endpoints[4]:endpoints[5], 1] = p5[1]
    output[endpoints[4]:endpoints[5], 2:] = np.array([-1, 0])

    l4 = [p1[0] * (1. - 1. / endpoints[1] * n) + p14[0] * 1. / endpoints[1] * n for n in range(endpoints[1])]
    output[endpoints[14]:endpoints[15], 0] = np.array(l4)
    output[endpoints[14]:endpoints[15], 1] = p14[1]
    output[endpoints[14]:endpoints[15], 2:] = np.array([1, 0])

    return output


# Part 5. 'main' function
if __name__ == '__main__':
    # fname = 'B_ofl_georama_static_GeoramaExtraCondensed-Black'
    # fname_out = 'B_ofl_georama_static_GeoramaExtraCondensed-BlackItalic'

    # fname = 'B_ofl_sansita_Sansita-Regular'
    # fname_out = 'B_ofl_sansita_Sansita-Bold'

    # fname = 'B_ofl_jost_static_Jost-MediumItalic'
    # fname_out = 'B_ofl_jost_static_Jost-Medium'

    fname = 'B_ofl_jost_static_Jost-Medium'
    fname_out = 'B_ofl_jost_static_Jost-MediumItalic'
    forward_only = True

    save_name = '%s_%s' % (fname, fname_out)
    if forward_only:
        save_name = save_name + '_' + 'forward'
    else:
        save_name = save_name + '_' + 'back'

    os.makedirs(os.path.join(exp_manipulate_dir, save_name), exist_ok=True)

    sample = qdataset.getitem_by_name(fname)
    flags = np.copy(sample['flag_bits'])
    index = 1

    endpoints = [-1] + list(np.where(flags.squeeze() == 100)[0]) + [len(flags) - 1]
    endpoints = list(map(lambda x: int(x + 1), endpoints))
    print(endpoints)

    input = sample['coordinate']
    output = qdataset.getitem_by_name(fname_out)['coordinate']
    # output = manipulate_points(input, endpoints)

    plot_points(input, endpoints, os.path.join(exp_manipulate_dir, save_name, '%s_input') % (fname))
    plot_points(output, endpoints, os.path.join(exp_manipulate_dir, save_name, '%s_target') % (fname))

    if forward_only:
        trainer = TrainerMGT([img2seq_model, seq2img_model], None, logger, None, args.nChars,
                             args.fixed_encoder, args.residual, args.quantize, args.extra_classifier,
                             False, False, False, False, False, False,
                             mapper_path=args.labels_mapper_path, template_path=args.data_template_path)

        trainer.manipulate(sample, output, index, os.path.join(exp_manipulate_dir, save_name), forward_only=True)
        exit()
    else:
        exp_tfb_dir = os.path.join(args.save_path, "tensorboard", args.exp)
        writer = SummaryWriter(exp_tfb_dir)

        trainer = TrainerMGT([img2seq_model, seq2img_model], None, logger, writer, args.nChars,
                             args.fixed_encoder, args.residual, args.quantize, args.extra_classifier,
                             False, False, False, False, False, False,
                             mapper_path=args.labels_mapper_path, template_path=args.data_template_path)

        coor_pred_init, feature_init, coor_preds, feature = trainer.manipulate(
            sample, output, index, os.path.join(exp_manipulate_dir, save_name))
        plot_points(coor_pred_init.detach().cpu().squeeze(), endpoints,
                    os.path.join(exp_manipulate_dir, save_name, '%s_0_output') % (fname))

        for v, e in coor_preds:
            plot_points(v, endpoints,
                        os.path.join(exp_manipulate_dir, save_name, '%s_%s_output') % (fname, e))
