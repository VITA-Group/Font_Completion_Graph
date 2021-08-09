import os
import sys
import torch
import collections
from datetime import datetime
from torch.utils.data import DataLoader

from common.arguments import args, dataloader_configs, transform_eval
from common.Trainer_img2seq import TrainerIMG_img2seq as TrainerMGT
from utilities.Logger import Logger

from dataset.FontImgDatasetBase import FontDatasetIMG
from models.img_encoder import ImgEncoderConv, Classifier_only
from models.seq_decoder import SeqDecoder

# -----------------------------------------------------------------------------------------------------
# Part 2. configurations
# Part 2-1. log configuration
args.exp = '_'.join([args.resume.split('/')[-2].replace('train', 'vis'),
                     args.resume.split('_')[-1][:-4], args.test_subset])
print(args.exp)

exp_log_dir = os.path.join(args.save_path, "log", args.exp)
exp_vis_dir = os.path.join(args.save_path, "visulize", args.exp)
os.makedirs(exp_log_dir, exist_ok=True)
os.makedirs(exp_vis_dir, exist_ok=True)
print('\n', exp_vis_dir, '\n')

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

model_configs['n_heads'] = args.n_heads
model_configs['n_layers'] = args.n_layers
model_configs['feed_forward_dim'] = args.feed_forward_dim
model_configs['normalization'] = args.normalization
model_configs['dropout'] = args.dropout
model_configs['mlp_classifier_dropout'] = args.mlp_classifier_dropout

encoder = ImgEncoderConv(n_classes=model_configs['output_dim'],
                         input_embed_dim=model_configs['input_embed_dim'],
                         feed_forward_dim=model_configs['feed_forward_dim'],
                         dropout=model_configs['mlp_classifier_dropout']).cuda()

decoder = SeqDecoder(feed_forward_dim=model_configs['feed_forward_dim'],
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

model = {'encoder': encoder, 'decoder': decoder}

if args.resume:
    checkpoint = torch.load(args.resume)
    model['encoder'].load_state_dict(checkpoint['encoder'])
    model['decoder'].load_state_dict(checkpoint['decoder'])

logger.info("model configuration settings: {}".format(model_configs))

# Part 2-3. dataloader instantiation
test_dataset = FontDatasetIMG(args.exp, dataloader_configs[args.exp],
                              args.label_idx, args.char_idx, transform_eval,
                              load_seq=True, nPoints=args.nPoints,
                              input_dim=args.coord_input_dim, padding=args.padding,
                              residual=args.residual, quantize=args.quantize, pair=False, DA=False)
test_loader = DataLoader(test_dataset, batch_size=dataloader_configs['batch_size'], shuffle=False,
                         num_workers=dataloader_configs['num_workers'])

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
logger.info("dataloader configuration settings: {}".format(dataloader_configs))

# Part 5. 'main' function
if __name__ == '__main__':
    trainer = TrainerMGT(model, None, logger, None, args.nChars,
                         args.fixed_encoder, args.residual, args.quantize, args.extra_classifier,
                         False, False, False, False, False, False)

    trainer.visulize(test_loader, exp_vis_dir, quantize=args.quantize, centers=None)
