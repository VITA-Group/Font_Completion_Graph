import os
import sys
import torch
import collections
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader

from common.arguments import args, dataloader_configs, transform_eval
from common.Trainer_img2img import TrainerIMG_img2img as TrainerMGT
from utilities.Logger import Logger

from dataset.FontImgDatasetBase import FontDatasetIMG
from models.img_encoder import ImgEncoderConv, Classifier_only
from models.img_decoder import ImgDecoder

# -----------------------------------------------------------------------------------------------------
# Part 2. configurations
# Part 2-1. log configuration
args.exp = "mul_baseline_img2img"
now_str = '_'.join([args.exp, args.padding] +
                   (['onehot'] if args.char_embedded else []) +
                   (['condConv'] if args.conditional_conv else []) +
                   (['condBN'] if args.conditional_BN else []) +
                   (['mgtDec'] if args.mgt_decoder else []) +
                   (['XE'] if args.XE_loss else []) +
                   (['AE'] if args.AE_loss else []) +
                   (['COS'] if args.COS_loss else []) +
                   (['fXE'] if args.flagXE_loss else []) +
                   (['qXE'] if args.quantizeXE_loss else []) +
                   ([str(args.miss_ratio)] if args.miss_ratio > 0 else []) +
                   [datetime.now().strftime("%Y%m%d%H%M%S")])

exp_log_dir = os.path.join(args.save_path, "log", now_str)
exp_multiinfer_dir = os.path.join(args.save_path, "multiinfer", now_str)
os.makedirs(exp_log_dir, exist_ok=True)
os.makedirs(exp_multiinfer_dir, exist_ok=True)
print('\n', exp_multiinfer_dir, '\n')

# writer = SummaryWriter(exp_tfb_dir)
logger = Logger(os.path.join(exp_log_dir, now_str + ".log")).get_logger()
logger.info("argument parser settings: {}".format(args))
# os.system('cp -r %s %s' % (os.path.join(os.getcwd()), exp_log_dir))
# os.system('cp    %s %s' % (args.resume, exp_log_dir))

# Part 2-4. configurations for loss function, model, and optimizer
model_configs = collections.OrderedDict()
model_configs['output_dim'] = args.output_dim
model_configs['input_embed_dim'] = args.input_embed_dim
model_configs['char_embed_dim'] = args.char_embed_dim

model_configs['feed_forward_dim'] = args.feed_forward_dim
model_configs['normalization'] = args.normalization
model_configs['dropout'] = args.dropout
model_configs['mlp_classifier_dropout'] = args.mlp_classifier_dropout

encoder = ImgEncoderConv(n_classes=model_configs['output_dim'],
                         input_embed_dim=model_configs['input_embed_dim'],
                         feed_forward_dim=model_configs['feed_forward_dim'],
                         dropout=model_configs['mlp_classifier_dropout']).cuda()

decoder = ImgDecoder(feed_forward_dim=model_configs['feed_forward_dim'],
                     char_embed_dim=model_configs['char_embed_dim'],
                     char_embedded=args.char_embedded,
                     conditional_conv=args.conditional_conv,
                     conditional_BN=args.conditional_BN,
                     nChars=args.nChars).cuda()

model = {'encoder': encoder, 'decoder': decoder}

if args.resume:
    checkpoint = torch.load(args.resume)
    model['encoder'].load_state_dict(checkpoint['encoder'])
    model['decoder'].load_state_dict(checkpoint['decoder'])

logger.info("model configuration settings: {}".format(model_configs))

# Part 2-3. dataloader instantiation
test_dataset = FontDatasetIMG('test', dataloader_configs['test'],
                              args.label_idx, args.char_idx, transform_eval,
                              load_seq=False, nPoints=args.nPoints,
                              input_dim=args.coord_input_dim, padding=args.padding,
                              residual=args.residual, quantize=args.quantize, pair=False, DA=False)
test_loader = DataLoader(test_dataset, batch_size=dataloader_configs['batch_size'], shuffle=False,
                         num_workers=dataloader_configs['num_workers'])

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
logger.info("dataloader configuration settings: {}".format(dataloader_configs))

# Part 5. 'main' function
if __name__ == '__main__':
    trainer = TrainerMGT(model, None, logger, None, args.nChars,
                         args.fixed_encoder, args.extra_classifier,
                         False, False, False)

    fnames = ['D_ofl_sourcesanspro_SourceSansPro-BlackItalic',
              'E_ofl_sourcesanspro_SourceSansPro-BlackItalic',
              'J_ofl_sourcesanspro_SourceSansPro-BlackItalic',
              'S_ofl_sourcesanspro_SourceSansPro-BlackItalic',
              'U_ofl_sourcesanspro_SourceSansPro-BlackItalic']
    font = 'ofl_sourcesanspro_SourceSansPro-BlackItalic'
    idx = [4, 5, 10, 19, 21]

    # fnames = ['E_ofl_faustina_Faustina-Italic[wght]',
    #           'F_ofl_faustina_Faustina-Italic[wght].svg',
    #           'G_ofl_faustina_Faustina-Italic[wght]',
    #           'J_ofl_faustina_Faustina-Italic[wght]',
    #           'K_ofl_faustina_Faustina-Italic[wght]']
    # font = 'ofl_faustina_Faustina-Italic[wght]'

    samples_list = [test_dataset.getitem_by_name(f) for f in fnames]
    format = next(iter(test_loader))

    samples_dict = {'filename': [s['filename'] for s in samples_list],
                    'sketch': torch.stack([s['sketch'] for s in samples_list]).type(format['sketch'].dtype)}

    for k in ['label', 'char']:
        samples_dict[k] = torch.tensor([s[k] for s in samples_list]).type(format[k].dtype)

    subset = test_dataset.subset
    gt_path = '/home/alternative/font_data/google_font_GTimgs/%s_npys' % subset
    gts = np.load(os.path.join(gt_path, font + '.npy'))

    trainer.mulinference(samples_dict, gts, idx, save_path=os.path.join(exp_multiinfer_dir, fnames[0][2:]))
    print(os.path.join(exp_multiinfer_dir, font))
