import os
import sys
import torch
import collections
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from common.arguments import args, dataloader_configs, lr_protocol, transform_train, transform_eval
from common.Trainer_img2seq2img import TrainerIMG_img2seq2img as TrainerMGT
from utilities.Logger import Logger

from dataset.FontImgDatasetBase import FontDatasetIMG
from dataset.FontSeqDatasets import FontDatasetMGTFC as FontDataset
from models.img_encoder import ImgEncoderConv, Classifier_only
from models.img_decoder import ImgDecoder
from models.seq_encoder import SeqEncoderConv
from models.seq_decoder import SeqDecoder

# -----------------------------------------------------------------------------------------------------
# Part 2. configurations
# Part 2-1. log configuration
args.exp = "train_baseline_img2seq2img"
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
exp_tfb_dir = os.path.join(args.save_path, "tensorboard", now_str)
exp_ckpt_dir = os.path.join(args.save_path, "checkpoints", now_str)
os.makedirs(exp_log_dir, exist_ok=True)
os.makedirs(exp_tfb_dir, exist_ok=True)
os.makedirs(exp_ckpt_dir, exist_ok=True)

writer = SummaryWriter(exp_tfb_dir)
logger = Logger(os.path.join(exp_log_dir, now_str + ".log")).get_logger()
logger.info("argument parser settings: {}".format(args))
os.system('cp -r %s %s' % (os.path.join(os.getcwd()), exp_log_dir))

# Part 2-2. Basic configuration
base_configs = collections.OrderedDict()
base_configs['serial_number'] = args.exp
base_configs['num_epochs'] = args.num_epochs
base_configs['patience'] = 10
base_configs["lr_protocol"] = lr_protocol
logger.info("basic configuration settings: {}".format(base_configs))

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
model_configs['mlp_classifier_dropout'] = args.mlp_classifier_dropout

img_encoder = ImgEncoderConv(n_classes=model_configs['output_dim'],
                             input_embed_dim=model_configs['input_embed_dim'],
                             feed_forward_dim=model_configs['feed_forward_dim'],
                             dropout=model_configs['mlp_classifier_dropout']).cuda()

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

if args.fixed_encoder:
    params = list(seq_encoder.parameters()) + list(img_decoder.parameters())
else:
    params = list(img_encoder.parameters()) + list(seq_decoder.parameters())
    params += list(seq_encoder.parameters()) + list(img_decoder.parameters())

if args.extra_classifier:
    img2seq_classifier = Classifier_only(args.output_dim, model_configs['feed_forward_hidden']).cuda()
    img2seq_model['classifier'] = img2seq_classifier
    params += list(img2seq_classifier.parameters())

    seq2img_classifier = Classifier_only(args.output_dim, model_configs['feed_forward_hidden']).cuda()
    seq2img_model['classifier'] = seq2img_classifier
    params += list(seq2img_classifier.parameters())

optimizer = torch.optim.Adam(params, lr=args.lr)

startpoint = 0
if args.resume:
    checkpoint = torch.load(args.resume)
    if args.fixed_encoder:
        img2seq_model['encoder'].load_state_dict(checkpoint["encoder"])
    else:
        for k, m in img2seq_model.items():
            m.load_state_dict(checkpoint[k])

        if not args.restart:
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'loss' in checkpoint:
                loss = checkpoint['loss']
            if not args.restart:
                startpoint = checkpoint['epoch'] - 1

        if args.resume_extra:
            checkpoint_extra = torch.load(args.resume)

            for k, m in seq2img_model.items():
                m.load_state_dict(checkpoint_extra[k])

logger.info("model configuration settings: {}".format(model_configs))

# Part 2-3. dataloader instantiation
train_dataset = FontDataset('train', dataloader_configs['train'],
                            args.label_idx, args.char_idx, args.nPoints,
                            args.coord_input_dim, args.padding,
                            residual=args.residual, quantize=args.quantize,
                            shorten=args.shorten, miss_seq=args.miss_ratio,
                            load_img=True, data_transforms=transform_train,
                            pair=args.pair_loss, DA=args.DA)
train_loader = DataLoader(train_dataset, batch_size=dataloader_configs['batch_size'], shuffle=True,
                          num_workers=dataloader_configs['num_workers'])
test_dataset = FontDataset('test', dataloader_configs['test'],
                           args.label_idx, args.char_idx, args.nPoints,
                           args.coord_input_dim, args.padding,
                           residual=args.residual, quantize=args.quantize,
                           shorten=args.shorten, miss_seq=args.miss_ratio,
                           load_img=True, data_transforms=transform_eval,
                           pair=args.pair_loss, DA=False)
test_loader = DataLoader(test_dataset, batch_size=dataloader_configs['batch_size'], shuffle=False,
                         num_workers=dataloader_configs['num_workers'])

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
logger.info("dataloader configuration settings: {}".format(dataloader_configs))

# Part 5. 'main' function
if __name__ == '__main__':
    trainer = TrainerMGT([img2seq_model, seq2img_model], optimizer, logger, writer, args.nChars,
                         args.fixed_encoder, args.residual, args.quantize, args.extra_classifier,
                         args.XE_loss, args.AE_loss, args.pair_loss,
                         args.COS_loss, args.flagXE_loss, args.quantizeXE_loss)
    trainer.stream(train_loader, test_loader, base_configs['lr_protocol'], base_configs['patience'],
                   startpoint, base_configs['num_epochs'], exp_ckpt_dir, args.exp)
