import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from common.TrainerBase import TrainerBase


class TrainerBase_seqDecoder(TrainerBase):
    def __init__(self, model, optimizer, logger, writer, nChars,
                 fixed_encoder, load_residual, load_quantize, extra_classifier,
                 XE_loss, AE_loss, COS_loss, pair_loss, flagXE_loss, quantizeXE_loss):
        super().__init__(model, optimizer, logger, writer)

        self.nChars = nChars
        self.fixed_encoder = fixed_encoder
        self.load_residual = load_residual
        self.load_quantize = load_quantize
        self.extra_classifier = extra_classifier

        self.XE_loss, self.AE_loss, self.COS_loss, self.pair_loss = XE_loss, AE_loss, COS_loss, pair_loss
        self.flagXE_loss, self.quantizeXE_loss = flagXE_loss, quantizeXE_loss

    def forward(self, data):
        res = self._forward(data)

        batch_loss = 0
        loss_dict = {}
        label, logit, feature = res['label'], res['logit'], res['feature']

        if self.XE_loss:
            XE_loss = nn.CrossEntropyLoss()(logit, label)
            batch_loss += XE_loss
            loss_dict['XE_loss'] = XE_loss

        if self.AE_loss:
            coordinate, reconstruct = res['coordinate'], res['reconstruct']
            assert coordinate.shape == reconstruct.shape

            # AE_loss = nn.MSELoss()(reconstruct[:, :stroke_len, :2], coordinate[:, :stroke_len, :2])
            AE_loss = nn.MSELoss()(reconstruct[:, :, :2], coordinate[:, :, :2])
            # AE_loss = nn.L1Loss()(reconstruct[:, :, :2], coordinate[:, :, :2])

            batch_loss += AE_loss
            loss_dict['CR_loss'] = AE_loss

        if self.COS_loss:
            coordinate, reconstruct = res['coordinate'], res['reconstruct']

            # COS_loss = nn.MSELoss()(reconstruct[:, :stroke_len, :2], coordinate[:, :stroke_len, :2])
            COS_loss = nn.MSELoss()(reconstruct[:, :, 2:], coordinate[:, :, 2:])
            # COS_loss = nn.L1Loss()(reconstruct[:, :, 2:], coordinate[:, :, 2:])

            batch_loss += COS_loss
            loss_dict['COS_loss'] = COS_loss

        if self.pair_loss:
            half_length = len(res['feature']) // 2
            PAIR_loss = nn.MSELoss()(res['feature'][:half_length], res['feature'][half_length:])
            loss_dict['PAIR_loss'] = PAIR_loss

        if self.flagXE_loss:
            flag_bits, flag_pred = res['flag_bits'], res['flag_pred']

            flagXE_loss = nn.CrossEntropyLoss()(flag_pred, flag_bits)
            batch_loss += flagXE_loss
            loss_dict['flagXE_loss'] = flagXE_loss

        if self.quantizeXE_loss:
            quantize, quantize_pred = res['quantize'], res['quant_pred']
            quantizeXE_loss = nn.CrossEntropyLoss()(
                quantize_pred.view(-1, quantize_pred.shape[-1]), quantize.view(-1))
            batch_loss += quantizeXE_loss
            loss_dict['quantizeXE_loss'] = quantizeXE_loss

        return res, label, logit, batch_loss, loss_dict

    def decode_all(self, coordinate, char, feature, stroke_len):
        reconstruct_all = [
            self.model['decoder'](feature, torch.ones(char.shape, dtype=char.dtype, device=feature.device) * c)[0]
            for c in range(self.nChars)]
        # 26 * b * 150 * 4

        if coordinate is None:
            coordinate = torch.zeros(reconstruct_all[0].shape, dtype=reconstruct_all[0].dtype,
                                     device=reconstruct_all[0].device)
        reconstructs = torch.stack([coordinate.to(reconstruct_all[0].device)] + reconstruct_all)
        # 27 * b * 150 * 4

        reconstructs = reconstructs.permute(1, 0, 2, 3)
        # b * 27 * 150 * 4

        stroke_lens = torch.stack([stroke_len] + [
            torch.ones(stroke_len.shape, dtype=stroke_len.dtype, device=stroke_len.device) * 512
            for _ in range(self.nChars)])
        stroke_lens = stroke_lens.permute(1, 0, 2)

        return reconstructs, stroke_lens

    def extract_list(self, data_loader, save_path, plot_all=True):
        if isinstance(self.model, dict):
            for k in self.model:
                self.model[k].eval()
        else:
            self.model.eval()

        res_filenames, res_labels, res_chars, res_features = [], [], [], []
        with torch.no_grad():
            for idx, data in enumerate(tqdm(data_loader, ascii=True)):
                res = self._forward(data, decode_all=plot_all)

                feature = res['feature']
                feature = (feature / torch.norm(feature, dim=1, keepdim=True))

                res_filenames += res['filename']
                res_labels += res['label']
                res_chars += res['char']
                res_features += feature

                if 'reconstructs' in res:
                    reconstructs, stroke_lens = res['reconstructs'], res['stroke_lens']
                    for n, output in enumerate(reconstructs):
                        self.points2img_9x3(output, stroke_lens[n], os.path.join(
                            save_path, "ret_%s.png" % res['filename'][n].split('/')[-1].split('.')[0]))

        res_labels = torch.stack(res_labels).cuda()
        res_chars = torch.stack(res_chars).cuda()
        res_features = torch.stack(res_features).cuda()

        return res_filenames, res_labels, res_chars, res_features

    def extract_dict(self, data_loader):
        if isinstance(self.model, dict):
            for k in self.model:
                self.model[k].eval()
        else:
            self.model.eval()

        res_dict = {}
        with torch.no_grad():
            for idx, data in enumerate(tqdm(data_loader, ascii=True)):
                res = self._forward(data)

                for c, l, v, n, f in zip(res['char'], res['filename'], res['coordinate'],
                                         res['stroke_len'], res['feature']):
                    _c = c.detach().cpu().item()
                    _l = l.split('_')[3].split('.')[0]

                    if _c in res_dict:
                        res_dict[_c][_l] = (c, l, v, n, f)
                    else:
                        res_dict[_c] = {_l: (c, l, v, n, f)}

            return res_dict

    def interpolate(self, coordinate1, coordinate2, char, feature1, feature2, stroke_len, save_path):
        reconstructs, stroke_lens = self.decode_all(
            coordinate1.unsqueeze(0), char.unsqueeze(0), feature1.unsqueeze(0), stroke_len.unsqueeze(0))

        for n, output in enumerate(reconstructs):
            self.points2img_9x3(output, stroke_lens[n], save_path + '_10.png')

        reconstructs, stroke_lens = self.decode_all(
            coordinate2.unsqueeze(0), char.unsqueeze(0), feature2.unsqueeze(0), stroke_len.unsqueeze(0))

        for n, output in enumerate(reconstructs):
            self.points2img_9x3(output, stroke_lens[n], save_path + '_0.png')

        inter_char = torch.stack([char] * 9).type(char.dtype).to(char.device)
        inter_feature = torch.stack([feature1 * (0.1 * n) + feature2 * (1 - 0.1 * n)
                                     for n in range(1, 10)]).type(feature1.dtype).to(feature1.device)
        inter_strokeLen = torch.stack([stroke_len] * 9).type(stroke_len.dtype).to(stroke_len.device)

        reconstructs, stroke_lens = self.decode_all(
            coordinate=None, char=inter_char, feature=inter_feature, stroke_len=inter_strokeLen)

        for n, output in enumerate(reconstructs):
            self.points2img_9x3(output, stroke_lens[n], save_path + '_%s.png' % (n + 1))
