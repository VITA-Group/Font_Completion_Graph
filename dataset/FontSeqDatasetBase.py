import os
import pickle
import random
import collections
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader
from dataset.norm_img import load_img_dict
from dataset.norm_seq import load_seq_dict, load_quant_dict
import torchvision.transforms as transforms


class FontDatasetBaseMGT(data.Dataset):

    def __init__(self, subset, config, label_idx, char_idx, nPoints, input_dim, padding,
                 residual, quantize, shorten, miss_seq, load_img, data_transforms, pair, DA):

        self.labels = [d.rstrip().split(' ') for d in open(config['sketch_list']).readlines()]
        if subset in ['test', 'query']:
            self.labels = sorted(self.labels)
        self.fnames = [v[0] for v in self.labels]

        if pair:
            fonts = [k for k, v in collections.Counter([l[2] for l in self.labels]).items()]
            labels_by_font = {f: [] for f in fonts}
            for n, info in enumerate(self.labels):
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

        self.load_img = load_img
        if load_img:
            self.img_dict = load_img_dict(config['img_path'], subset)
            self.data_transforms = data_transforms

        self.data_dict = load_seq_dict(config['data_dict_file'])
        if quantize:
            self.quant_dict = load_quant_dict(config['data_dict_file'].replace('dict', 'kmeans'))
        self.input_dim = input_dim
        self.nPoints = nPoints
        self.padding = padding
        self.shorten = shorten
        self.residual = residual
        self.quantize = quantize
        self.miss_seq = miss_seq
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

        data = {'filename': filename, 'label': label, 'char': char}

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

        if self.shorten:
            stroke_len = np.argwhere(indices == 0).min() if 0 in indices else stroke_len

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
        attention_masks = self.get_attention(flag_bits, stroke_len)

        data.update({'coordinate': coordinate, 'flag_bits': flag_bits, 'residual': residual,
                     'stroke_len': stroke_len, 'indices': np.expand_dims(indices, 1),
                     'attention_masks': attention_masks})

        if self.miss_seq:
            missing_mask = self.generate_missing_mask(stroke_len, self.nPoints, self.miss_seq).astype('float32')
            data['missing_mask'] = missing_mask

        if self.quantize:
            quantize = self.quant_dict[filename].astype('long')
            data['quantize'] = quantize

        if self.load_img:
            sketch = self.data_transforms(self.img_dict[filename])
            data['sketch'] = sketch
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

    def produce_adjacent_matrix_2_neighbors(self, flag_bits, stroke_len, nPoints):
        assert flag_bits.shape == (nPoints, 1)
        adja_matr = np.zeros([nPoints, nPoints], int)

        adja_matr[:][:] = -1e10

        adja_matr[0][0] = 0

        if (flag_bits[0] == 100):
            adja_matr[0][1] = 0

        for idx in range(1, stroke_len):

            assert flag_bits[idx] == 100 or flag_bits[idx] == 101

            adja_matr[idx][idx] = 0

            if (flag_bits[idx - 1] == 100):
                adja_matr[idx][idx - 1] = 0

            if idx == stroke_len - 1:
                break

            if (flag_bits[idx] == 100):
                adja_matr[idx][idx + 1] = 0

        return adja_matr

    def produce_adjacent_matrix_4_neighbors(self, flag_bits, stroke_len, nPoints):
        assert flag_bits.shape == (nPoints, 1)
        adja_matr = np.zeros([nPoints, nPoints], int)
        adja_matr[:][:] = -1e10

        adja_matr[0][0] = 0
        # TODO
        if (flag_bits[0] == 100):
            adja_matr[0][1] = 0
            #
            if (flag_bits[1] == 100):
                adja_matr[0][2] = 0

        for idx in range(1, stroke_len):

            assert flag_bits[idx] == 100 or flag_bits[idx] == 101

            adja_matr[idx][idx] = 0

            if (flag_bits[idx - 1] == 100):
                adja_matr[idx][idx - 1] = 0
                #
                if (idx >= 2) and (flag_bits[idx - 2] == 100):
                    adja_matr[idx][idx - 2] = 0

            if idx == stroke_len - 1:
                break

            #
            if (idx <= (stroke_len - 2)) and (flag_bits[idx] == 100):
                adja_matr[idx][idx + 1] = 0
                #
                if (idx <= (stroke_len - 3)) and (flag_bits[idx + 1] == 100):
                    adja_matr[idx][idx + 2] = 0

        return adja_matr

    def produce_adjacent_matrix_6_neighbors(self, flag_bits, stroke_len, nPoints):
        assert flag_bits.shape == (nPoints, 1)
        adja_matr = np.zeros([nPoints, nPoints], int)
        adja_matr[:][:] = -1e10

        adja_matr[0][0] = 0
        # TODO
        if (flag_bits[0] == 100):
            adja_matr[0][1] = 0
            #
            if (flag_bits[1] == 100):
                adja_matr[0][2] = 0
                if (flag_bits[2] == 100):
                    adja_matr[0][3] = 0

        for idx in range(1, stroke_len):
            #
            adja_matr[idx][idx] = 0

            if (flag_bits[idx - 1] == 100):
                adja_matr[idx][idx - 1] = 0
                #
                if (idx >= 2) and (flag_bits[idx - 2] == 100):
                    adja_matr[idx][idx - 2] = 0
                    if (idx >= 3) and (flag_bits[idx - 3] == 100):
                        adja_matr[idx][idx - 3] = 0

            if idx == stroke_len - 1:
                break

            #
            if (idx <= (stroke_len - 2)) and (flag_bits[idx] == 100):
                adja_matr[idx][idx + 1] = 0
                #
                if (idx <= (stroke_len - 3)) and (flag_bits[idx + 1] == 100):
                    adja_matr[idx][idx + 2] = 0
                    if (idx <= (stroke_len - 4)) and (flag_bits[idx + 2] == 100):
                        adja_matr[idx][idx + 3] = 0

        return adja_matr

    def produce_adjacent_matrix_joint_neighbors(self, flag_bits, stroke_len, nPoints):
        assert flag_bits.shape == (nPoints, 1)
        adja_matr = np.zeros([nPoints, nPoints], int)
        adja_matr[:][:] = -1e10

        adja_matr[0][0] = 0
        adja_matr[0][stroke_len - 1] = 0
        adja_matr[stroke_len - 1][stroke_len - 1] = 0
        adja_matr[stroke_len - 1][0] = 0

        assert flag_bits[0] == 100 or flag_bits[0] == 100 + 1

        if (flag_bits[0] == 101) and stroke_len >= 2:
            adja_matr[0][1] = 0

        for idx in range(1, stroke_len):

            assert flag_bits[idx] == 100 or flag_bits[idx] == 101

            adja_matr[idx][idx] = 0

            if (flag_bits[idx - 1] == 101):
                adja_matr[idx][idx - 1] = 0

            if (idx == stroke_len - 1):
                break

            #
            if (idx <= (stroke_len - 2)) and (flag_bits[idx] == 101):
                adja_matr[idx][idx + 1] = 0

        return adja_matr

    def generate_attention_mask(self, stroke_length, nPoints):
        attention_mask = np.zeros([nPoints, nPoints], int)

        attention_mask[stroke_length:, :] = -1e8
        attention_mask[:, stroke_length:] = -1e8
        return attention_mask

    def generate_padding_mask(self, stroke_length, nPoints):
        padding_mask = np.ones([nPoints, 1], int)
        padding_mask[stroke_length:, :] = 0
        return padding_mask

    def generate_missing_mask(self, stroke_length, nPoints, miss_ratio):
        missing_mask = np.ones([nPoints, 1], int)
        if miss_ratio:
            missing_mask[int(stroke_length * (1 - miss_ratio)):, :] = 0
        return missing_mask

    '''
    def check_adjacent_matrix(adjacent_matrix, stroke_len):
        assert adjacent_matrix.shape == (100, 100)
        for idx in range(1, stroke_len):
            assert adjacent_matrix[idx][idx - 1] == adjacent_matrix[idx - 1][idx]
    '''

    def get_attention(self, flag_bits, stroke_len):
        attention_mask_2_neighbors = self.produce_adjacent_matrix_2_neighbors(
            flag_bits, stroke_len, nPoints=self.nPoints)

        attention_mask_4_neighbors = self.produce_adjacent_matrix_4_neighbors(
            flag_bits, stroke_len, nPoints=self.nPoints)

        attention_mask_6_neighbors = self.produce_adjacent_matrix_6_neighbors(
            flag_bits, stroke_len, nPoints=self.nPoints)

        attention_mask_joint_neighbors = self.produce_adjacent_matrix_joint_neighbors(
            flag_bits, stroke_len, nPoints=self.nPoints)

        attention_mask = self.generate_attention_mask(stroke_len, nPoints=self.nPoints)
        padding_mask = self.generate_padding_mask(stroke_len, nPoints=self.nPoints)

        return [attention_mask_2_neighbors, attention_mask_4_neighbors, attention_mask_6_neighbors,
                attention_mask_joint_neighbors, attention_mask, padding_mask]


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

    sampleDS = FontDatasetBaseMGT('query', config, 1, 3, 150, 4,
                                  'zero', False, False, False, 0, False, transform, False, False)
    sample = next(iter(sampleDS))
    print(sample['filename'][2:-4], '\n',
          sample.keys(), type(sample['coordinate']), sample['coordinate'].shape)

    sampleLoader = DataLoader(sampleDS, batch_size=1, shuffle=False)
    samples = next(iter(sampleLoader))
    print(samples['filename'][0][2:-4], '\n',
          samples.keys(), type(samples['coordinate']), samples['coordinate'][0].shape)

    fname = samples['filename'][0][2:-4]
    sample_by_name = sampleDS.getitem_by_name(fname)
    print(fname, '\n',
          sample_by_name.keys(), type(sample_by_name['coordinate']), sample_by_name['coordinate'].shape)

    print(np.sum(sample['coordinate'] - sample_by_name['coordinate']))
    print(np.sum(samples['coordinate'][0].detach().cpu().numpy() - sample_by_name['coordinate']))
