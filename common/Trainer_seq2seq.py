import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
from common.TrainerBase_seqDecoder import TrainerBase_seqDecoder


class TrainerMGT_seq2seq(TrainerBase_seqDecoder):
    def __init__(self, model, optimizer, logger, writer, nChars,
                 fixed_encoder, load_residual, load_quantize, extra_classifier,
                 XE_loss, AE_loss, COS_loss, pair_loss, flagXE_loss, quantizeXE_loss):
        super().__init__(model, optimizer, logger, writer, nChars,
                         fixed_encoder, load_residual, load_quantize, extra_classifier,
                         XE_loss, AE_loss, COS_loss, pair_loss, flagXE_loss, quantizeXE_loss)

    def _forward(self, data, decode_all=False):
        if isinstance(data, tuple) or isinstance(data, list):
            data_update = {k: torch.cat([d[k] for d in data]) for k in data[0] if
                           k not in ['filename', 'attention_masks']}
            data_update['filename'] = sum([d['filename'] for d in data], [])

            if 'attention_masks' in data[0]:
                masks = [d['attention_masks'] for d in data]
                data_update['attention_masks'] = [torch.cat([d[n] for d in masks]) for n in range(len(masks[0]))]

            data = data_update

        sketch = data['sketch'].cuda() if 'sketch' in data else None

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

        res = {'filename': filename, 'sketch': sketch, 'coordinate': coordinate, 'label': label, 'char': char,
               'flag_bits': flag_bits, 'stroke_len': stroke_len, 'indices': indices}

        if self.fixed_encoder:
            self.model['encoder'].eval()

        if self.load_quantize:
            input_quantize = data['quantize'].cuda()
            res['quantize'] = input_quantize

            logit, feature = self.model['encoder'](
                input_quantize, flag_bits, stroke_len, attention_masks, indices)

        else:
            input_coordinate = data['residual' if self.load_residual else 'coordinate'].cuda()

            if 'missing_mask' in data:
                missing_mask = data['missing_mask'].cuda()
                input_coordinate = input_coordinate * missing_mask

            logit, feature = self.model['encoder'](
                input_coordinate, flag_bits, stroke_len, attention_masks, indices)

        coor_pred, flag_pred, quant_pred = self.model['decoder'](feature, char, attention_masks)
        res.update({'feature': feature, 'logit': logit, 'reconstruct': coor_pred,
                    'flag_pred': flag_pred, 'quant_pred': quant_pred})

        if self.extra_classifier:
            logit = self.model['classifier'](feature)
            res.update({'logit': logit})

        if decode_all:
            res['reconstructs'], res['stroke_lens'] = self.decode_all(
                coordinate, char, feature, stroke_len)

        return res

    def visulize(self, data_loader, save_path, quantize=False, centers=None):
        if isinstance(self.model, dict):
            for k in self.model:
                self.model[k].eval()
        else:
            self.model.eval()

        with torch.no_grad():
            for idx, data in enumerate(tqdm(data_loader, ascii=True)):
                res = self._forward(data)

                sketch = res['sketch']
                coordinate = res['coordinate']
                reconstruct = res['reconstruct']
                if quantize:
                    quantize_idx = res['quant_pred'].argmax(-1)
                    reconstruct = torch.Tensor(np.cumsum(centers[quantize_idx.cpu().numpy()],
                                                         axis=-1)).type_as(reconstruct)
                stroke_len = res['stroke_len']
                indices = res['indices']

                save_image(1 - sketch, os.path.join(save_path, str(idx).zfill(6) + "_sketch.png"))

                self.points2img_8x8(coordinate, stroke_len,
                                    os.path.join(save_path, str(idx).zfill(6) + "_coordinate.png"), indices)

                self.points2img_8x8(reconstruct, stroke_len,
                                    os.path.join(save_path, str(idx).zfill(6) + "_recover.png"), indices)
