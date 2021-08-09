import os
import pickle
import random
import collections
import numpy as np
import torch.utils.data as data
from PIL import Image
from dataset.norm_img import load_img_dict
from dataset.norm_seq import load_seq_dict, load_quant_dict
import torchvision.transforms as transforms


class FontDatasetIMG(data.Dataset):
    def __init__(self, subset, config, label_idx, char_idx, data_transforms,
                 load_seq, nPoints, input_dim, padding, residual, quantize, pair, DA):

        self.labels = [d.rstrip().split(' ') for d in open(config['sketch_list']).readlines()]
        if subset in ['test', 'query']:
            self.labels = sorted(self.labels)
        self.fnames = [v[0] for v in self.labels]

        if pair:
            fonts = [k for k, v in collections.Counter([l[2] for l in self.labels]).items()]
            labels_by_font = {f: [] for f in fonts}
            for n, info in enumerate(self.labels):
                # if info[2] in fonts:
                labels_by_font[info[2]].append([n, info[0], info[2]])
            self.fonts = labels_by_font

        self.subset = subset
        self.char_idx = char_idx

        self.label_idx = label_idx
        if label_idx == -1:
            mapper = pickle.load(open(config['mapper'], "rb"))
            keys = sorted(['%s-%s-%s' % (c, s, w)
                           for c in range(len(mapper['categories']))
                           for s in range(len(mapper['styles']))
                           for w in range(len(mapper['weights']))])
            self.mapper = {k: n for n, k in enumerate(keys)}

        self.img_dict = load_img_dict(config['img_path'], subset)
        self.data_transforms = data_transforms

        self.load_seq = load_seq
        if load_seq:
            self.data_dict = load_seq_dict(config['data_dict_file'])
            if quantize:
                self.quant_dict = load_quant_dict(config['data_dict_file'].replace('dict', 'kmeans'))
            self.input_dim = input_dim
            self.nPoints = nPoints
            self.padding = padding
            self.residual = residual
            self.quantize = quantize
            self.DA = DA
        self.pair = pair

    def __len__(self):
        return len(self.labels)

    def get_item(self, filename, item):
        char = int(self.labels[item][self.char_idx].strip())

        if self.label_idx == -1:
            label = self.labels[item]
            c, s, w = int(label[4].strip()), int(label[5].strip()), int(label[6].strip())
            label = self.mapper['%s-%s-%s' % (c, s, w)]
        else:
            label = int(self.labels[item][self.label_idx].strip())

        sketch = self.data_transforms(self.img_dict[filename])
        data = {'filename': filename, 'sketch': sketch, 'label': label, 'char': char}

        if self.load_seq:
            coordinate, flag_bits, stroke_len, seqs, indices = self.data_dict[filename]
            coordinate = coordinate.astype('float32')[:self.nPoints]
            flag_bits = flag_bits.astype('int')[:self.nPoints]
            indices = indices[:self.nPoints]
            # assert self.nPoints >= stroke_len and self.nPoints == len(coordinate)
            stroke_len = min(stroke_len, self.nPoints)

            coordinate = np.array(coordinate)
            if self.DA:
                nums = coordinate[:, :2]

                scale = random.random() + 0.5
                if random.random() < 0.5:
                    nums[:, 0] *= scale
                else:
                    nums[:, 1] *= scale

                min_x, min_y = nums.min(axis=0)
                max_x, max_y = nums.max(axis=0)
                # mean_x, mean_y = nums.mean(axis=0)
                rangs_x, range_y = max_x - min_x, max_y - min_y

                coordinate[:, :2] = (nums[:, :2] - (min_x, min_y)) * 256. / max(rangs_x, range_y)

            if self.residual:
                residual = coordinate[1:stroke_len, :2] - coordinate[:stroke_len - 1, :2]
                residual = np.row_stack((coordinate[:1, :2], residual, coordinate[stroke_len:, :2]))
                residual = np.column_stack((residual, coordinate[:, 2:]))
                coordinate[:, :2] -= coordinate[:1, :2]
            else:
                residual = np.zeros(coordinate.shape)

            if self.padding == 'zero':
                _mask = self.generate_padding_mask(stroke_len, nPoints=self.nPoints)
                coordinate = coordinate * _mask
                flag_bits = flag_bits * _mask
                residual = residual * _mask
            elif self.padding == 'repeat':
                coordinate = np.concatenate([coordinate[:stroke_len]] * (self.nPoints // stroke_len + 1))
                flag_bits = np.concatenate([flag_bits[:stroke_len]] * (self.nPoints // stroke_len + 1))
                residual = np.concatenate([residual[:stroke_len]] * (self.nPoints // stroke_len + 1))
            elif self.padding == 'rewind':
                coordinate = np.concatenate([coordinate[:stroke_len], coordinate[:stroke_len][::-1]])
                flag_bits = np.concatenate([flag_bits[:stroke_len], flag_bits[:stroke_len][::-1]])
                residual = np.concatenate([residual[:stroke_len], residual[:stroke_len][::-1]])

                coordinate = np.concatenate([coordinate] * (self.nPoints // (stroke_len * 2) + 1))
                flag_bits = np.concatenate([flag_bits] * (self.nPoints // (stroke_len * 2) + 1))
                residual = np.concatenate([residual] * (self.nPoints // (stroke_len * 2) + 1))
            else:
                assert self.padding == 'default'

            coordinate = coordinate[:self.nPoints, :self.input_dim]
            residual = residual[:self.nPoints, :self.input_dim]
            flag_bits = flag_bits[:self.nPoints]
            indices = indices[:self.nPoints]
            indices[-1] = 0
            attention_masks = self.generate_padding_mask(stroke_len, nPoints=self.nPoints)
            # attention_masks = self.get_attention(flag_bits, stroke_len)

            data.update({'coordinate': coordinate, 'flag_bits': flag_bits, 'residual': residual,
                         'stroke_len': stroke_len, 'indices': np.expand_dims(indices, 1),
                         'attention_masks': attention_masks})

            if self.quantize:
                quantize = self.quant_dict[filename].astype('long')
                data['quantize'] = quantize

        return data

    def getitem_by_name(self, fname):
        if '/' not in fname:
            fname = fname.split('/')[-1]
        if '.' in fname:
            fname = fname.split('.')[0]

        fname = '%s/%s.svg' % (fname[0], fname)
        item = self.fnames.index(fname)

        data = self.get_item(fname, item)
        return data

    def __getitem__(self, item):
        info = self.labels[item]
        data = self.get_item(info[0], item)

        if not self.pair:
            return data
        else:
            info2 = random.choice(self.fonts[info[2]])
            data2 = self.get_item(info2[1], info2[0])
            return (data, data2)

    def generate_padding_mask(self, stroke_length, nPoints):
        padding_mask = np.ones([nPoints, 1], int)
        padding_mask[stroke_length:, :] = 0
        return padding_mask


if __name__ == '__main__':
    dataset_path = '/home/alternative/font_data/google_font_dataset/uppercase'
    config = {'sketch_list': os.path.join(dataset_path, 'tiny_query_set.txt'),
              'data_dict_file': os.path.join(dataset_path, 'tiny_query_dataset_dict.pickle'),
              'img_path': dataset_path}

    transform = transforms.Compose([
        transforms.Grayscale(),
        # transforms.Resize(299),
        transforms.Pad(299),
        transforms.CenterCrop(299),
        transforms.ToTensor()
    ])

    # subset, config, postfix, data_transforms, label_idx, char_idx
    sampleDS = FontDatasetIMG('query', config, 1, 3, transform,
                              True, 150, 4, 'zero', False, False, False, False)

    sketch = next(iter(sampleDS))['sketch']
    print(sketch.shape, sketch.max())

    data = np.array(sketch.permute(1, 2, 0).cpu().numpy() * 255, dtype=np.ubyte)
    print(data.shape, data.max())

    img = Image.fromarray(data.squeeze())
    img.save(os.path.join('demo/sampleDS.png'))
