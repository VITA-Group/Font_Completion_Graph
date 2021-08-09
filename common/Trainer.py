import torch
import torch.nn as nn
from tqdm import tqdm
from common.TrainerBase import TrainerBase


class TrainerMGT(TrainerBase):
    def __init__(self, model, optimizer, logger, writer):
        super().__init__(model, optimizer, logger, writer)

    def _forward(self, data, decode=False, decode_all=False):
        filename = data['filename']
        coordinate = data['coordinate'].cuda()
        label = data['label'].cuda()
        char = data['char'].cuda()
        flag_bits = data['flag_bits'].cuda()
        stroke_len = data['stroke_len'].cuda()
        indices = data['indices'].cuda()
        attention_masks = data['attention_masks']
        for n in range(len(attention_masks)):
            attention_masks[n] = attention_masks[n].cuda()

        # Resize inputs
        flag_bits.squeeze_(2)
        indices.squeeze_(2)
        stroke_len.unsqueeze_(1)

        logit, feature = self.model(coordinate, flag_bits, stroke_len, attention_masks, indices)
        res = {'filename': filename, 'coordinate': coordinate, 'label': label, 'char': char,
               'flag_bits': flag_bits, 'stroke_len': stroke_len, 'indices': indices,
               'feature': feature, 'logit': logit}

        return res

    def forward(self, data):
        res = self._forward(data)

        batch_loss = 0
        loss_dict = {}
        label, logit = res['label'], res['logit']

        XE_loss = nn.CrossEntropyLoss()(logit, label)
        batch_loss += XE_loss
        loss_dict['XE_loss'] = XE_loss

        return label, logit, batch_loss, loss_dict


class TrainerIMG(TrainerBase):
    def __init__(self, model, optimizer, logger, writer):
        super().__init__(model, optimizer, logger, writer)

    def _forward(self, data, decode=False, decode_all=False):
        filename = data['filename']
        label = data['label'].cuda()
        char = data['char'].cuda()
        sketch = data['sketch'].cuda()

        logit, feature = self.model(sketch)
        res = {'filename': filename, 'sketch': sketch, 'label': label, 'char': char,
               'feature': feature, 'logit': logit}

        return res

    def forward(self, data):
        res = self._forward(data)

        batch_loss = 0
        loss_dict = {}
        label, logit = res['label'], res['logit']

        XE_loss = nn.CrossEntropyLoss()(logit, label)
        batch_loss += XE_loss
        loss_dict['XE_loss'] = XE_loss

        return label, logit, batch_loss, loss_dict
