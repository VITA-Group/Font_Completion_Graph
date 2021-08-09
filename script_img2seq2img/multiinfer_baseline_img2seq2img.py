import os
import sys
import torch
import collections
import numpy as np
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
args.test_subset = 'testAll'
args.resume = '/home/alternative/font_exps/checkpoints/train_baseline_img2seq_rewind_condConv_XE_AE_20201116051730/train_baseline_img2seq_20.pth'
args.resume_extra = '/home/alternative/font_exps/checkpoints/train_baseline_seq2img_rewind_condConv_XE_AE_COS_20201113052727/train_baseline_seq2img_10.pth'

# -----------------------------------------------------------------------------------------------------
# Part 2. configurations
# Part 2-1. log configuration
args.exp = '_'.join([args.resume.split('/')[-2].replace('train', 'mul').replace('img2seq', 'img2seq2img'),
                     args.resume.split('_')[-1][:-4],
                     args.resume_extra.split('/')[-2].split('_')[-1],
                     args.resume_extra.split('_')[-1][:-4],
                     args.test_subset])
print(args.exp)

exp_log_dir = os.path.join(args.save_path, "log", args.exp)
exp_multiinfer_dir = os.path.join(args.save_path, "multiinfer", args.exp)
os.makedirs(exp_log_dir, exist_ok=True)
os.makedirs(exp_multiinfer_dir, exist_ok=True)
print('\n', exp_multiinfer_dir, '\n')

logger = Logger(os.path.join(exp_log_dir, args.exp + ".log")).get_logger()
logger.info("argument parser settings: {}".format(args))

# Part 2-4. configurations for loss function, model, and optimizer
model_configs = collections.OrderedDict()
model_configs['output_dim'] = args.output_dim
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

seq_encoder = SeqEncoderConv(n_classes=model_configs['output_dim'],
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
test_dataset = FontDataset('testAll', dataloader_configs['testAll'],
                           args.label_idx, args.char_idx, args.nPoints,
                           args.coord_input_dim, args.padding,
                           residual=args.residual, quantize=args.quantize,
                           shorten=args.shorten, miss_seq=args.miss_ratio,
                           load_img=True, data_transforms=transform_eval, pair=False, DA=False)
test_loader = DataLoader(test_dataset, batch_size=2)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
logger.info("dataloader configuration settings: {}".format(dataloader_configs))


def proc(fnames, font, idx):
    samples_list = [test_dataset.getitem_by_name(f) for f in fnames]
    format = next(iter(test_loader))

    samples_dict = {'filename': [s['filename'] for s in samples_list],
                    'sketch': torch.stack([s['sketch'] for s in samples_list]).type(format['sketch'].dtype),
                    'attention_masks': [
                        torch.stack([torch.tensor(s['attention_masks'][n]) for s in samples_list]).type(m.dtype)
                        for n, m in enumerate(format['attention_masks'])]}
    for k in ['label', 'char', 'coordinate', 'flag_bits', 'residual', 'stroke_len', 'indices']:
        samples_dict[k] = torch.tensor([s[k] for s in samples_list]).type(format[k].dtype)

    subset = test_dataset.subset
    gt_path = '/home/alternative/font_data/google_font_GTimgs/%s_npys' % subset
    gts = np.load(os.path.join(gt_path, font + '.npy'))

    os.makedirs(os.path.join(exp_multiinfer_dir, font), exist_ok=True)
    save_path = os.path.join(exp_multiinfer_dir, font, ''.join([f[0] for f in fnames]))
    trainer.mulinference(samples_dict, gts, idx, save_path=save_path)
    print(save_path)


# Part 5. 'main' function
if __name__ == '__main__':
    trainer = TrainerMGT([img2seq_model, seq2img_model], None, logger, None, args.nChars,
                         args.fixed_encoder, args.residual, args.quantize, args.extra_classifier,
                         False, False, False, False, False, False,
                         mapper_path=args.labels_mapper_path, template_path=args.data_template_path)

    # fnames = ['D_ofl_sourcesanspro_SourceSansPro-BlackItalic',
    #           'E_ofl_sourcesanspro_SourceSansPro-BlackItalic',
    #           'J_ofl_sourcesanspro_SourceSansPro-BlackItalic',
    #           'S_ofl_sourcesanspro_SourceSansPro-BlackItalic',
    #           'U_ofl_sourcesanspro_SourceSansPro-BlackItalic']
    # font = 'ofl_sourcesanspro_SourceSansPro-BlackItalic'
    # idx = [4, 5, 10, 19, 21]

    # fnames = ['E_ofl_faustina_Faustina-Italic[wght]',
    #           'F_ofl_faustina_Faustina-Italic[wght].svg',
    #           'G_ofl_faustina_Faustina-Italic[wght]',
    #           'J_ofl_faustina_Faustina-Italic[wght]',
    #           'K_ofl_faustina_Faustina-Italic[wght]']
    # font = 'ofl_faustina_Faustina-Italic[wght]'

    from string import ascii_uppercase

    font = 'ofl_lora_static_Lora-Italic'
    chars = list(ascii_uppercase)[::-1]
    fnames = []
    idx = []
    for n, c in enumerate(chars):
        fnames.append('%s_%s' % (c, font))
        idx.append(25 - n)
        proc(fnames, font, idx)
