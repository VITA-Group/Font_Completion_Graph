import os
import pickle
import argparse
import collections
import torchvision.transforms as transforms

# -----------------------------------------------------------------------------------------------------
# Part 1. Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
parser.add_argument('--gpu', type=str, default="0", help='choose GPU')
parser.add_argument('--num_epochs', type=int, default=100, help='num_epochs')
# parser.add_argument('--gpu', type=str, default="0,1,2,3", help='choose GPU')
parser.add_argument("--resume", type=str, default="", help="resume")
parser.add_argument("--resume_extra", type=str, default="", help="resume_extra")
parser.add_argument("--save_path", type=str, default="font_exps", help="save_path")

parser.add_argument("--data_path", type=str, help="data_path",
                    default="font_data/google_font_dataset/uppercase")
parser.add_argument("--img_path", type=str, help="img_path",
                    default="font_data/google_font_dataset/uppercase")
parser.add_argument("--data_template_path", type=str, help="data_template_path",
                    default='font_data/google_font_dataset/uppercase/template.pickle')
parser.add_argument("--labels_mapper_path", type=str, help="labels_mapper_path",
                    default='font_data/google_font_dataset/uppercase/labels_mapper.pkl')
parser.add_argument("--data_4_train", type=str, default='tiny_train_dataset_dict.pickle', help="data_4_train")
parser.add_argument("--data_4_test", type=str, default='tiny_test_dataset_dict.pickle', help="data_4_test")
parser.add_argument("--label_4_train", type=str, default="tiny_train_set.txt", help="label_4_train")
parser.add_argument("--label_4_test", type=str, default="tiny_test_set.txt", help="label_4_test")
parser.add_argument("--label_type", type=str, default='meta', help="label_type")
parser.add_argument("--label_retrieval", type=str, default='fonts', help="label_type")

parser.add_argument("--nChars", type=int, default=26, help="nPoints")
parser.add_argument("--nPoints", type=int, default=150, help="nPoints")
parser.add_argument("--n_heads", type=int, default=8, help="n_heads")
parser.add_argument("--coord_input_dim", type=int, default=4, help="coord_input_dim")
parser.add_argument("--quant_input_dim", type=int, default=101, help="quant_input_dim")
parser.add_argument("--flag_input_dim", type=int, default=3, help="flag_input_dim")
parser.add_argument("--input_embed_dim", type=int, default=256, help="input_embed_dim")
parser.add_argument("--feed_forward_dim", type=int, default=1024, help="feed_forward_dim")
parser.add_argument("--char_embed_dim", type=int, default=32, help="char_embed_dim")
parser.add_argument("--input_embed_dim_extra", type=int, default=256, help="input_embed_dim")
parser.add_argument("--feed_forward_dim_extra", type=int, default=512, help="feed_forward_dim")

parser.add_argument("--lr", type=float, default=0.001, help="lr")
parser.add_argument("--n_layers", type=int, default=4, help="n_layers")
parser.add_argument("--normalization", type=str, default='batch', help="normalization")
parser.add_argument("--dropout", type=float, default=0.25, help="dropout")
parser.add_argument("--mlp_classifier_dropout", type=float, default=0.25, help="mlp_classifier_dropout")

parser.add_argument('--miss_ratio', type=float, default=0, help='miss_ratio')
parser.add_argument('--padding', type=str, default='rewind', help='padding')
parser.add_argument('--shorten', action='store_true')
parser.add_argument('--residual', action='store_true')
parser.add_argument('--quantize', action='store_true')
parser.set_defaults(shorten=False)
parser.set_defaults(residual=False)
parser.set_defaults(quantize=False)

parser.add_argument('--fixed_encoder', action='store_true')
parser.add_argument('--extra_classifier', action='store_true')
parser.add_argument('--extra_seq_decoder', action='store_true')
parser.add_argument('--extra_img_decoder', action='store_true')
parser.set_defaults(fixed_encoder=False)
parser.set_defaults(extra_classifier=False)
parser.set_defaults(extra_seq_decoder=False)
parser.set_defaults(extra_img_decoder=False)

parser.add_argument('--char_embedded', action='store_true')
parser.add_argument('--conditional_conv', action='store_true')
parser.add_argument('--conditional_BN', action='store_true')
parser.add_argument('--mgt_decoder', action='store_true')
parser.set_defaults(char_embedded=False)
parser.set_defaults(conditional_conv=True)
parser.set_defaults(conditional_BN=False)
parser.set_defaults(mgt_decoder=False)

parser.add_argument('--restart', action='store_true')
parser.add_argument('--DA', action='store_true')
parser.add_argument('--XE_loss', action='store_true')
parser.add_argument('--AE_loss', action='store_true')
parser.add_argument('--COS_loss', action='store_true')
parser.add_argument('--pair_loss', action='store_true')
parser.add_argument('--flagXE_loss', action='store_true')
parser.add_argument('--quantizeXE_loss', action='store_true')
parser.set_defaults(restart=False)
parser.set_defaults(DA=False)
parser.set_defaults(XE_loss=False)
parser.set_defaults(AE_loss=False)
parser.set_defaults(COS_loss=False)
parser.set_defaults(pair_loss=False)
parser.set_defaults(flagXE_loss=False)
parser.set_defaults(quantizeXE_loss=False)

parser.add_argument("--test_subset", type=str, default='test')
parser.add_argument("--n_demo", type=int, default=5, help="n_demo")
parser.add_argument('--plot_all', action='store_true')
parser.add_argument('--gallery_all', action='store_true')
parser.add_argument('--vis_retrieval', action='store_true')
parser.set_defaults(plot_all=False)
parser.set_defaults(gallery_all=False)
parser.set_defaults(vis_retrieval=False)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
assert args.nPoints >= 150

label_order = ['filename', 'families', 'fonts', 'charset', 'categories', 'styles', 'weights']
args.char_idx = label_order.index('charset')
if args.label_type == 'meta':
    args.label_idx = -1
    mapper = pickle.load(open(
        os.path.join(args.data_path, 'labels_mapper.pkl'), "rb"))
    args.output_dim = len(mapper['categories']) * len(mapper['styles']) * len(mapper['weights'])
else:
    args.label_idx = label_order.index(args.label_type)
    args.output_dim = len(pickle.load(open(
        os.path.join(args.data_path, 'labels_mapper.pkl'), "rb"))[args.label_type])
args.label_retrieval = label_order.index(args.label_retrieval)

# Part 2-3. dataloader instantiation
dataloader_configs = collections.OrderedDict()
dataloader_configs['batch_size'] = args.batch_size
dataloader_configs['num_workers'] = args.num_workers
dataloader_configs['mapper'] = os.path.join(args.data_path, 'labels_mapper.pkl')
dataloader_configs['glyph2labels'] = os.path.join(args.data_path, 'glyph2labels_remap.pkl')

dataloader_configs['train'] = {
    'sketch_list': os.path.join(args.data_path, args.label_4_train),
    'data_dict_file': os.path.join(args.data_path, args.data_4_train),
    'img_path': args.img_path,
    'mapper': os.path.join(args.data_path, 'labels_mapper.pkl')
}
dataloader_configs['test'] = {
    'sketch_list': os.path.join(args.data_path, args.label_4_test),
    'data_dict_file': os.path.join(args.data_path, args.data_4_test),
    'img_path': args.img_path,
    'mapper': os.path.join(args.data_path, 'labels_mapper.pkl')
}
dataloader_configs['gallery'] = {
    'sketch_list': os.path.join(args.data_path, 'tiny_gallery_set.txt'),
    'data_dict_file': os.path.join(args.data_path, 'tiny_gallery_dataset_dict.pickle'),
    'img_path': args.img_path,
    'mapper': os.path.join(args.data_path, 'labels_mapper.pkl')
}
dataloader_configs['galleryAll'] = {
    'sketch_list': os.path.join(args.data_path, 'tiny_galleryAll_set.txt'),
    'data_dict_file': os.path.join(args.data_path, 'tiny_galleryAll_dataset_dict.pickle'),
    'img_path': args.img_path,
    'mapper': os.path.join(args.data_path, 'labels_mapper.pkl')
}
dataloader_configs['query'] = {
    'sketch_list': os.path.join(args.data_path, 'tiny_query_set.txt'),
    'data_dict_file': os.path.join(args.data_path, 'tiny_query_dataset_dict.pickle'),
    'img_path': args.img_path
}

transform_train = transforms.Compose([
    transforms.Grayscale(),
    transforms.Pad(299),
    transforms.CenterCrop(299),
    transforms.Resize(128),
    transforms.ToTensor()
])

transform_eval = transforms.Compose([
    transforms.Grayscale(),
    transforms.Pad(299),
    transforms.CenterCrop(299),
    transforms.Resize(128),
    transforms.ToTensor()
])

lr_protocol = [(30 * (n + 1), args.lr * (0.9 ** n)) for n in range(10)]

# lr_protocol = [
#     (30, 1e-3), (60, 0.001 * 0.9), (90, 0.001 * (0.9 ** 2)), (120, 0.001 * (0.9 ** 3)),
#     (150, 0.001 * (0.9 ** 4)), (180, 0.001 * (0.9 ** 5)), (210, 1e-4), (240, 1e-5)]
