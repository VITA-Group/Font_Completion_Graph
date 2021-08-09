import os
import sys
import torch
import collections
import itertools as it
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader

from common.arguments import args, dataloader_configs, transform_eval
from common.Trainer_seq2img import TrainerMGT_seq2img as TrainerMGT
from utilities.Logger import Logger

from dataset.FontSeqDatasets import FontDatasetMGTFC as FontDataset
from models.seq_encoder import SeqEncoderConv
from models.img_decoder import ImgDecoder

# -----------------------------------------------------------------------------------------------------
# Part 2. configurations
# Part 2-1. log configuration
args.exp = "int_baseline_seq2img"
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
exp_interpolate_dir = os.path.join(args.save_path, "interpolate", now_str)
os.makedirs(exp_log_dir, exist_ok=True)
os.makedirs(exp_interpolate_dir, exist_ok=True)

# writer = SummaryWriter(exp_tfb_dir)
logger = Logger(os.path.join(exp_log_dir, now_str + ".log")).get_logger()
logger.info("argument parser settings: {}".format(args))
os.system('cp -r %s %s' % (os.path.join(os.getcwd()), exp_log_dir))
os.system('cp    %s %s' % (args.resume, exp_log_dir))

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

encoder = SeqEncoderConv(n_classes=model_configs['output_dim'],
                         coord_input_dim=model_configs['coord_input_dim'],
                         quant_input_dim=model_configs['quant_input_dim'],
                         feat_dict_size=model_configs['nPoints'],
                         load_quantize=args.quantize,
                         n_layers=model_configs['n_layers'], n_heads=model_configs['n_heads'],
                         input_embed_dim=model_configs['input_embed_dim'],
                         feed_forward_dim=model_configs['feed_forward_dim'],
                         normalization=model_configs['normalization'],
                         dropout=model_configs['dropout'],
                         mlp_classifier_dropout=model_configs['mlp_classifier_dropout']).cuda()

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
qdataset = FontDataset('query', dataloader_configs['query'],
                       args.label_retrieval, args.char_idx, args.nPoints,
                       args.coord_input_dim, args.padding,
                       residual=args.residual, quantize=args.quantize,
                       shorten=args.shorten, miss_seq=args.miss_ratio,
                       load_img=True, data_transforms=transform_eval, pair=False, DA=False)
# gdataset = FontDataset('gallery', dataloader_configs['gallery'],
#                        args.label_retrieval, args.char_idx, args.nPoints,
#                        args.coord_input_dim, args.padding,
#                        residual=args.residual, quantize=args.quantize,
#                        shorten=args.shorten, miss_seq=args.miss_ratio,
#                        load_img=True, data_transforms=transform_eval, pair=False, DA=False)
qloader = DataLoader(qdataset, batch_size=dataloader_configs['batch_size'], shuffle=False,
                     num_workers=dataloader_configs['num_workers'])
# gloader = DataLoader(gdataset, batch_size=dataloader_configs['batch_size'], shuffle=False,
#                      num_workers=dataloader_configs['num_workers'])

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
logger.info("dataloader configuration settings: {}".format(dataloader_configs))

# Part 5. 'main' function
if __name__ == '__main__':
    trainer = TrainerMGT(model, None, logger, None, args.nChars,
                         args.fixed_encoder, args.residual, args.quantize, args.extra_classifier,
                         False, False, False)

    # os.makedirs(os.path.join(exp_interpolate_dir, 'query'))
    qfs = trainer.extract_dict(qloader)

    # os.makedirs(os.path.join(exp_interpolate_dir, 'gallery'))
    # gfs = trainer.decode(gloader)

    for c in qfs:
        if len(qfs[c].keys()) < 2:
            continue

        all_combinations = list(it.combinations(qfs[c].keys(), 2))
        if args.n_demo:
            all_combinations = all_combinations[:args.n_demo]

        for l1, l2 in tqdm(sorted(all_combinations)):
            os.makedirs(os.path.join(exp_interpolate_dir, '%s_%s_%s' % (c, l1, l2)))
            trainer.interpolate(qfs[c][l1][2], qfs[c][l2][2], qfs[c][l1][0],
                                qfs[c][l1][-1], qfs[c][l2][-1],
                                os.path.join(exp_interpolate_dir, '%s_%s_%s' % (c, l1, l2), '%s_%s_%s' % (c, l1, l2)))
