import os
from torch.utils.data import DataLoader
from dataset.FontSeqDatasetBase import FontDatasetBaseMGT


class FontDatasetMGTBigru(FontDatasetBaseMGT):
    def __init__(self, subset, config, label_idx, char_idx, nPoints, input_dim, padding,
                 residual, quantize, shorten, miss_seq, load_img, data_transforms, pair, DA):
        super().__init__(subset, config, label_idx, char_idx, nPoints, input_dim, padding,
                         residual, quantize, shorten, miss_seq, load_img, data_transforms, pair, DA)

    def get_attention(self, flag_bits, stroke_len):
        attention_mask = self.generate_attention_mask(stroke_len, nPoints=self.nPoints)
        padding_mask = self.generate_padding_mask(stroke_len, nPoints=self.nPoints)

        return [attention_mask, padding_mask]


class FontDatasetMGT2nn(FontDatasetBaseMGT):
    def __init__(self, subset, config, label_idx, char_idx, nPoints, input_dim, padding,
                 residual, quantize, shorten, miss_seq, load_img, data_transforms, pair, DA):
        super().__init__(subset, config, label_idx, char_idx, nPoints, input_dim, padding,
                         residual, quantize, shorten, miss_seq, load_img, data_transforms, pair, DA)

    def get_attention(self, flag_bits, stroke_len):
        attention_mask_2_neighbors = self.produce_adjacent_matrix_2_neighbors(
            flag_bits, stroke_len, nPoints=self.nPoints)
        padding_mask = self.generate_padding_mask(stroke_len, nPoints=self.nPoints)

        return [attention_mask_2_neighbors, padding_mask]


class FontDatasetMGTJnn(FontDatasetBaseMGT):
    def __init__(self, subset, config, label_idx, char_idx, nPoints, input_dim, padding,
                 residual, quantize, shorten, miss_seq, load_img, data_transforms, pair, DA):
        super().__init__(subset, config, label_idx, char_idx, nPoints, input_dim, padding,
                         residual, quantize, shorten, miss_seq, load_img, data_transforms, pair, DA)

    def get_attention(self, flag_bits, stroke_len):
        attention_mask_joint_neighbors = self.produce_adjacent_matrix_joint_neighbors(
            flag_bits, stroke_len, nPoints=self.nPoints)
        padding_mask = self.generate_padding_mask(stroke_len, nPoints=self.nPoints)

        return [attention_mask_joint_neighbors, padding_mask]


class FontDatasetMGT2nnJnn(FontDatasetBaseMGT):
    def __init__(self, subset, config, label_idx, char_idx, nPoints, input_dim, padding,
                 residual, quantize, shorten, miss_seq, load_img, data_transforms, pair, DA):
        super().__init__(subset, config, label_idx, char_idx, nPoints, input_dim, padding,
                         residual, quantize, shorten, miss_seq, load_img, data_transforms, pair, DA)

    def get_attention(self, flag_bits, stroke_len):
        attention_mask_2_neighbors = self.produce_adjacent_matrix_2_neighbors(
            flag_bits, stroke_len, nPoints=self.nPoints)
        attention_mask_joint_neighbors = self.produce_adjacent_matrix_joint_neighbors(
            flag_bits, stroke_len, nPoints=self.nPoints)
        padding_mask = self.generate_padding_mask(stroke_len, nPoints=self.nPoints)

        return [attention_mask_2_neighbors, attention_mask_joint_neighbors, padding_mask]


class FontDatasetMGTFC(FontDatasetBaseMGT):
    def __init__(self, subset, config, label_idx, char_idx, nPoints, input_dim, padding,
                 residual, quantize, shorten, miss_seq, load_img, data_transforms, pair, DA):
        super().__init__(subset, config, label_idx, char_idx, nPoints, input_dim, padding,
                         residual, quantize, shorten, miss_seq, load_img, data_transforms, pair, DA)

    def get_attention(self, flag_bits, stroke_len):
        attention_mask = self.generate_attention_mask(stroke_len, nPoints=self.nPoints)
        padding_mask = self.generate_padding_mask(stroke_len, nPoints=self.nPoints)

        return [attention_mask, padding_mask]
