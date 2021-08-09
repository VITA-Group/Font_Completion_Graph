import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from common.TrainerBase_imgDecoder import TrainerBase_imgDecoder


class TrainerIMG_img2img(TrainerBase_imgDecoder):
    def __init__(self, model, optimizer, logger, writer, nChars,
                 fixed_encoder, extra_classifier, XE_loss, AE_loss, pair_loss):
        super().__init__(model, optimizer, logger, writer, nChars,
                         fixed_encoder, extra_classifier, XE_loss, AE_loss, pair_loss)

    def _forward(self, data, decode=False, decode_all=False):
        if isinstance(data, tuple) or isinstance(data, list):
            data_update = {k: torch.cat([d[k] for d in data]) for k in data[0] if
                           k not in ['filename', 'attention_masks']}
            data_update['filename'] = sum([d['filename'] for d in data], [])

            if 'attention_masks' in data[0]:
                masks = [d['attention_masks'] for d in data]
                data_update['attention_masks'] = [torch.cat([d[n] for d in masks]) for n in range(len(masks[0]))]

            data = data_update

        sketch = data['sketch'].cuda()
        filename = data['filename']
        label = data['label'].cuda()
        char = data['char'].cuda()

        res = {'filename': filename, 'sketch': sketch, 'label': label, 'char': char}

        logit, feature = self.model['encoder'](sketch)
        img_pred = self.model['decoder'](feature, char)
        res.update({'feature': feature, 'logit': logit, 'reconstruct': img_pred})

        if self.extra_classifier:
            logit = self.model['classifier'](feature)
            res.update({'logit': logit})

        if decode_all:
            res['reconstructs'] = self.decode_all(sketch, char, feature)

        return res

    def visulize(self, data_loader, save_path):

        if isinstance(self.model, dict):
            for k in self.model:
                self.model[k].eval()
        else:
            self.model.eval()

        os.makedirs(os.path.join(save_path, 'visualize'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'vq_metrics'), exist_ok=True)

        with torch.no_grad():
            subset = data_loader.dataset.subset
            gt_path = '/home/alternative/font_data/google_font_GTimgs/%s_npys' % subset
            gts = {f.split('.')[0]: np.load(os.path.join(gt_path, f))
                   for f in tqdm(os.listdir(gt_path)) if f.endswith('npy')}

            for _, data in enumerate(tqdm(data_loader, ascii=True)):
                res = self._forward(data, decode_all=True)

                reconstructs = res['reconstructs']
                for idx, output in enumerate(reconstructs):
                    save_name = res['filename'][idx].split('/')[-1].split('.')[0]
                    font = save_name[2:]

                    assert 0 <= res['char'][idx] < 26
                    assert output.shape[0] == 27

                    vq_metrics = []
                    for i, pred in enumerate(output[1:].clone().detach().cpu().numpy()):
                        gt = gts[font][i].transpose(1, 2, 0)
                        pred = np.clip(pred.transpose(1, 2, 0), 0., 1.)
                        vq_metrics.append([round(mse(gt, pred), 2), round(psnr(gt, pred), 2),
                                           round(ssim(gt, pred, multichannel=True), 2)])

                    with open(os.path.join(save_path, 'vq_metrics', "%s.txt" % save_name), 'w') as f:
                        f.write(str('mse(gt, pred), psnr(gt, pred), ssim(gt, pred, multichannel=True)' + '\n'))
                        for vq in vq_metrics:
                            f.write(str(vq) + '\n')

                    output[res['char'][idx] + 1][:, :5, :] = torch.tensor([0, 1, 1])[:, None, None]
                    output[res['char'][idx] + 1][:, :, :5] = torch.tensor([0, 1, 1])[:, None, None]
                    output[res['char'][idx] + 1][:, -5:, :] = torch.tensor([0, 1, 1])[:, None, None]
                    output[res['char'][idx] + 1][:, :, -5:] = torch.tensor([0, 1, 1])[:, None, None]

                    save_image(1 - output[1:], os.path.join(save_path, "%s_all.png" % save_name), nrow=26)

    def mulinference(self, data, gts, idx, save_path):
        self.set_model_status(train=False)

        with torch.no_grad():
            res = self._forward(data)

            reconstructs = self.decode_all(sketch=None, char=res['char'][:1],
                                           feature=torch.mean(res['feature'], dim=0, keepdim=True))

            for n, output in enumerate(reconstructs):

                vq_metrics = []
                for i, pred in enumerate(output[1:].clone().detach().cpu().numpy()):
                    gt = gts[i].transpose(1, 2, 0)
                    pred = np.clip(pred.transpose(1, 2, 0), 0., 1.)
                    vq_metrics.append([round(mse(gt, pred), 2), round(psnr(gt, pred), 2),
                                       round(ssim(gt, pred, multichannel=True), 2)])

                with open(os.path.join(save_path + '_vq.txt'), 'w') as f:
                    f.write(str('mse(gt, pred), psnr(gt, pred), ssim(gt, pred, multichannel=True)' + '\n'))
                    for vq in vq_metrics:
                        f.write(str(vq) + '\n')

                output[idx, :, :5, :] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
                output[idx, :, :, :5] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
                output[idx, :, -5:, :] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
                output[idx, :, :, -5:] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]

                save_image(1 - output[1:], save_path + '_fused.png', nrow=26)
