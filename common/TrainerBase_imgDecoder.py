import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image

from common.TrainerBase import TrainerBase


class TrainerBase_imgDecoder(TrainerBase):
    def __init__(self, model, optimizer, logger, writer, nChars,
                 fixed_encoder, extra_classifier,
                 XE_loss, AE_loss, pair_loss):
        super().__init__(model, optimizer, logger, writer)

        self.nChars = nChars
        self.fixed_encoder = fixed_encoder
        self.extra_classifier = extra_classifier

        self.XE_loss, self.AE_loss, self.pair_loss = XE_loss, AE_loss, pair_loss

    def forward(self, data):
        res = self._forward(data)

        batch_loss = 0
        loss_dict = {}
        label, logit = res['label'], res['logit']

        if self.XE_loss:
            XE_loss = nn.CrossEntropyLoss()(logit, label)
            batch_loss += XE_loss
            loss_dict['XE_loss'] = XE_loss

        if self.AE_loss:
            sketch, reconstruct = res['sketch'], res['reconstruct']
            assert sketch.shape == reconstruct.shape

            AE_loss = nn.MSELoss()(reconstruct, sketch)

            batch_loss += AE_loss
            loss_dict['PX_loss'] = AE_loss * 1000.

        if self.pair_loss:
            half_length = len(res['feature']) // 2
            PAIR_loss = nn.MSELoss()(res['feature'][:half_length], res['feature'][half_length:])
            loss_dict['PAIR_loss'] = PAIR_loss * 1000.

        return res, label, logit, batch_loss, loss_dict

    def decode_all(self, sketch, char, feature):
        reconstruct_all = [
            self.model['decoder'](feature, torch.ones(char.shape, dtype=char.dtype, device=char.device) * c)
            for c in range(self.nChars)]
        # 26 * b * 150 * 128 * 128

        if sketch is None:
            sketch = torch.zeros(reconstruct_all[0].shape, dtype=reconstruct_all[0].dtype,
                                 device=reconstruct_all[0].device)
        reconstructs = torch.stack([sketch] + reconstruct_all)
        # 27 * b * 150 * 128 * 128

        reconstructs = reconstructs.permute(1, 0, 2, 3, 4)
        # b * 27 * 150 * 128 * 128

        return reconstructs

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

                res_filenames += res['filename']
                res_labels += res['label']
                res_chars += res['char']
                feature = res['feature']
                res_features += (feature / torch.norm(feature, dim=1, keepdim=True))

                if 'reconstructs' in res:
                    reconstructs = res['reconstructs']
                    for n, output in enumerate(reconstructs):
                        save_image(1 - output, os.path.join(
                            save_path, "ret_%s.png" % res['filename'][n].split('/')[-1].split('.')[0]), nrow=9)

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

                for c, l, v, f in zip(res['char'], res['filename'], res['sketch'], res['feature']):
                    _c = c.detach().cpu().item()
                    _l = l.split('_')[3].split('.')[0]

                    if _c in res_dict:
                        res_dict[_c][_l] = (c, l, v, f)
                    else:
                        res_dict[_c] = {_l: (c, l, v, f)}

            return res_dict

    def interpolate(self, sketch1, sketch2, char, feature1, feature2, save_path):
        reconstructs = self.decode_all(sketch1.unsqueeze(0), char.unsqueeze(0), feature1.unsqueeze(0))

        for n, output in enumerate(reconstructs):
            save_image(1 - output, save_path + '_10.png', nrow=9)

        reconstructs = self.decode_all(sketch2.unsqueeze(0), char.unsqueeze(0), feature2.unsqueeze(0))

        for n, output in enumerate(reconstructs):
            save_image(1 - output, save_path + '_0.png', nrow=9)

        inter_char = torch.stack([char] * 9).type(char.dtype).to(char.device)
        inter_feature = torch.stack([feature1 * (0.1 * n) + feature2 * (1 - 0.1 * n)
                                     for n in range(1, 10)]).type(feature1.dtype).to(feature1.device)

        reconstructs = self.decode_all(sketch=None, char=inter_char, feature=inter_feature)

        for n, output in enumerate(reconstructs):
            save_image(1 - output, save_path + '_%s.png' % (n + 1), nrow=9)
