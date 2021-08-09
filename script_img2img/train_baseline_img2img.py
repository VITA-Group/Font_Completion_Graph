import os
import sys
import torch
import collections
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from common.arguments import args, dataloader_configs, transform_train, transform_eval
from common.Trainer_img2img import TrainerIMG_img2img as TrainerMGT
from utilities.Logger import Logger

from dataset.FontImgDatasetBase import FontDatasetIMG
from models.img_encoder import ImgEncoderConv, Classifier_only
from models.img_decoder import ImgDecoder

# -----------------------------------------------------------------------------------------------------
# Part 2. configurations
# Part 2-1. log configuration
args.exp = "train_baseline_img2img"
now_str = '_'.join([args.exp, args.padding] +
                   (['onehot'] if args.char_embedded else []) +
                   (['condConv'] if args.conditional_conv else []) +
                   (['condBN'] if args.conditional_BN else []) +
                   (['mgtDec'] if args.mgt_decoder else []) +
                   (['XE'] if args.XE_loss else []) +
                   (['AE'] if args.AE_loss else []) +
                   (['COS'] if args.COS_loss else []) +
                   (['PAIR'] if args.pair_loss else []) +
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
base_configs['learning_rate'] = args.lr  # 1e-3
base_configs['num_epochs'] = args.num_epochs
base_configs['patience'] = 10
base_configs["lr_protocol"] = [(args.decay_epoch * (n + 1), (1e-3) * (args.decay_rate ** n))
                               for n in range(args.num_epochs // args.decay_epoch)]
# [(10, 1e-3), (20, (1e-3) * 0.5), (30, (1e-3) * 0.5 * 0.5),
#  (40, (1e-3) * 0.5 * 0.5 * 0.5), (50, (1e-3) * 0.5 * 0.5 * 0.5 * 0.5)]
logger.info("basic configuration settings: {}".format(base_configs))

# Part 2-4. configurations for loss function, model, and optimizer
model_configs = collections.OrderedDict()
model_configs['output_dim'] = args.output_dim
model_configs['input_embed_dim'] = args.input_embed_dim
model_configs['char_embed_dim'] = args.char_embed_dim

model_configs['feed_forward_dim'] = args.feed_forward_dim
model_configs['normalization'] = args.normalization
model_configs['dropout'] = args.dropout

encoder = ImgEncoderConv(n_classes=model_configs['output_dim'],
                         input_embed_dim=model_configs['input_embed_dim'],
                         feed_forward_dim=model_configs['feed_forward_dim'],
                         dropout=model_configs['dropout']).cuda()

decoder = ImgDecoder(feed_forward_dim=model_configs['feed_forward_dim'],
                     char_embed_dim=model_configs['char_embed_dim'],
                     char_embedded=args.char_embedded,
                     conditional_conv=args.conditional_conv,
                     conditional_BN=args.conditional_BN,
                     nChars=args.nChars).cuda()

model = {'encoder': encoder, 'decoder': decoder}

if args.fixed_encoder:
    params = list(decoder.parameters())
else:
    params = list(encoder.parameters()) + list(decoder.parameters())

if args.extra_classifier:
    classifier = Classifier_only(args.output_dim, model_configs['feed_forward_hidden']).cuda()
    model['classifier'] = classifier
    params += list(classifier.parameters())

optimizer = torch.optim.Adam(params, lr=base_configs['learning_rate'])

startpoint = 0
if args.resume:
    checkpoint = torch.load(args.resume)
    if args.fixed_encoder:
        model['encoder'].load_state_dict(checkpoint["encoder"])
    else:
        for k, m in model.items():
            m.load_state_dict(checkpoint[k])

        if not args.restart:
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'loss' in checkpoint:
                loss = checkpoint['loss']

logger.info("model configuration settings: {}".format(model_configs))

# Part 2-3. dataloader instantiation
train_dataset = FontDatasetIMG('train', dataloader_configs['train'],
                               args.label_idx, args.char_idx, transform_train,
                               load_seq=False, nPoints=args.nPoints,
                               input_dim=args.coord_input_dim, padding=args.padding,
                               residual=args.residual, quantize=args.quantize,
                               pair=args.pair_loss, DA=args.DA)
train_loader = DataLoader(train_dataset, batch_size=dataloader_configs['batch_size'], shuffle=True,
                          num_workers=dataloader_configs['num_workers'])
test_dataset = FontDatasetIMG('test', dataloader_configs['test'],
                              args.label_idx, args.char_idx, transform_eval,
                              load_seq=False, nPoints=args.nPoints,
                              input_dim=args.coord_input_dim, padding=args.padding,
                              residual=args.residual, quantize=args.quantize,
                              pair=args.pair_loss, DA=False)
test_loader = DataLoader(test_dataset, batch_size=dataloader_configs['batch_size'], shuffle=False,
                         num_workers=dataloader_configs['num_workers'])

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
logger.info("dataloader configuration settings: {}".format(dataloader_configs))

# Part 5. 'main' function
if __name__ == '__main__':
    trainer = TrainerMGT(model, optimizer, logger, writer, args.nChars,
                         args.fixed_encoder, args.extra_classifier,
                         args.XE_loss, args.AE_loss, args.pair_loss)
    trainer.stream(train_loader, test_loader, base_configs['lr_protocol'], base_configs['patience'],
                   startpoint, base_configs['num_epochs'], exp_ckpt_dir, args.exp)
