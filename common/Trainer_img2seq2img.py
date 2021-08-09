import os
import torch
import pickle
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torchvision.utils import save_image
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from common.TrainerBase import TrainerBase
from common.Trainer_img2seq import TrainerIMG_img2seq
from common.Trainer_seq2img import TrainerMGT_seq2img


class TrainerIMG_img2seq2img(TrainerBase):
    def __init__(self, model, optimizer, logger, writer, nChars,
                 fixed_encoder, load_residual, load_quantize, extra_classifier,
                 XE_loss, AE_loss, COS_loss, pair_loss, flagXE_loss, quantizeXE_loss,
                 mapper_path=None, template_path=None):
        super().__init__(model, optimizer, logger, writer)

        self.trainer_img2seq = TrainerIMG_img2seq(
            model[0], optimizer, logger, writer, nChars,
            fixed_encoder, load_residual, load_quantize, extra_classifier,
            XE_loss, AE_loss, COS_loss, pair_loss, flagXE_loss, quantizeXE_loss)
        self.trainer_seq2img = TrainerMGT_seq2img(
            model[1], optimizer, logger, writer, nChars,
            fixed_encoder, load_residual, load_quantize, extra_classifier,
            XE_loss, AE_loss, pair_loss)

        self.nChars = nChars
        self.fixed_encoder = fixed_encoder
        self.load_residual = load_residual
        self.load_quantize = load_quantize
        self.extra_classifier = extra_classifier

        self.XE_loss, self.AE_loss, self.COS_loss = XE_loss, AE_loss, COS_loss
        self.flagXE_loss, self.quantizeXE_loss = flagXE_loss, quantizeXE_loss

        if mapper_path is not None and template_path is not None:
            char_mapper = pickle.load(open(mapper_path, 'rb'))['charset']
            template = pickle.load(open(template_path, 'rb'))
            self.template = {char_mapper[k]: [torch.tensor(v) for v in vs] for k, vs in template.items()}
            # points, flag, length, seqs, indices

    def set_model_status(self, train=True):
        def _set(model):
            if train:
                if isinstance(model, dict):
                    for k in model:
                        model[k].train()
                else:
                    model.train()
            else:
                if isinstance(model, dict):
                    for k in model:
                        model[k].eval()
                else:
                    model.eval()

        _set(self.model[0])
        _set(self.model[1])

    def save_model(self, save_dir, save_name, epoch):
        def _save(model, tag):
            checkpoint_path = os.path.join(save_dir, '_'.join([tag, save_name, str(epoch + 1)]) + '.pth')

            if isinstance(model, dict):
                model_state = {"epoch": epoch + 1,
                               'optimizer': self.optimizer.state_dict()}
                for k, m in model.items():
                    model_state[k] = m.state_dict()
                torch.save(model_state, checkpoint_path)

            else:
                model_state = {"epoch": epoch + 1,
                               'optimizer': self.optimizer.state_dict(),
                               "model": model.state_dict()}
                torch.save(model_state, checkpoint_path)

        _save(self.model[0], 'img2seq')
        _save(self.model[1], 'seq2img')

    def batch_process(self, data):
        img2seq_res, label, img2seq_logit, img2seq_bl, img2seq_ld = self.trainer_img2seq.forward(data)
        img2seq_ld = {'img2seq_' + k: v for k, v in img2seq_ld.items()}

        data['coordinate'] = img2seq_res['reconstruct']
        seq2img_res, label, seq2img_logit, seq2img_bl, seq2img_ld = self.trainer_seq2img.forward(data)
        seq2img_ld = {'seq2img_' + k: v for k, v in seq2img_ld.items()}

        return label, (img2seq_logit + seq2img_logit) / 2, seq2img_bl + seq2img_bl, {**img2seq_ld, **seq2img_ld}

    def decode_all(self, sketch, coordinate, char, feature, stroke_len):
        img2seq_reconstructs, _ = self.trainer_img2seq.decode_all(coordinate, char, feature, stroke_len)

        seq2img_reconstructs = []
        for n, reconstruct in enumerate(img2seq_reconstructs.permute(1, 0, 2, 3)[1:]):
            # 26 * b * 150 * 4
            batch_size = len(reconstruct)
            data = {
                'sketch': torch.zeros(sketch.shape) if sketch is not None else torch.zeros((batch_size, 3, 128, 128)),
                'filename': '',
                'label': torch.zeros(char.shape),
                'coordinate': reconstruct,
                'char': torch.ones(char.shape, dtype=char.dtype, device=char.device) * n,
                'flag_bits': torch.stack([self.template[n][1]] * batch_size).type(char.dtype).to(char.device),
                'stroke_len': torch.stack([self.template[n][2]] * batch_size).type(char.dtype).to(char.device),
                'indices': torch.stack([self.template[n][-4]] * batch_size).type(char.dtype).to(char.device),
                'attention_masks': []
            }

            res = self.trainer_seq2img._forward(data)
            seq2img_reconstructs.append(res['reconstruct'])

        if sketch is None:
            sketch = torch.zeros(seq2img_reconstructs[0].shape, dtype=seq2img_reconstructs[0].dtype,
                                 device=seq2img_reconstructs[0].device)
        seq2img_reconstructs = torch.stack([sketch.to(seq2img_reconstructs[0].device)] + seq2img_reconstructs)
        seq2img_reconstructs = seq2img_reconstructs.permute(1, 0, 2, 3, 4)

        strok_len_alphabet = torch.stack([self.template[n][2] for n in sorted(self.template.keys())])
        stroke_lens = torch.cat([stroke_len, strok_len_alphabet.repeat(len(stroke_len), 1).to(stroke_len.device)],
                                dim=1)
        return img2seq_reconstructs, seq2img_reconstructs, stroke_lens

    def visulize(self, data_loader, save_path):
        self.set_model_status(train=False)

        os.makedirs(os.path.join(save_path, 'visualize'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'vq_metrics'), exist_ok=True)

        with torch.no_grad():
            subset = data_loader.dataset.subset
            gt_path = '/home/alternative/font_data/google_font_GTimgs/%s_npys' % subset
            gts = {f.split('.')[0]: np.load(os.path.join(gt_path, f))
                   for f in tqdm(os.listdir(gt_path)) if f.endswith('npy')}

            for _, data in enumerate(tqdm(data_loader, ascii=True)):
                img2seq_res = self.trainer_img2seq._forward(data, decode_all=True)

                coordinate = data['coordinate'].clone()
                data['coordinate'] = img2seq_res['reconstruct']
                seq2img_res = self.trainer_seq2img._forward(data)
                data['coordinate'] = coordinate

                img2seq_reconstructs, seq2img_reconstructs, stroke_lens = self.decode_all(
                    data['sketch'], data['coordinate'], data['char'], img2seq_res['feature'],
                    img2seq_res['stroke_len'])

                # sketch = img2seq_res['sketch']
                # coordinate = img2seq_res['coordinate']
                # reconstruct_seq = img2seq_res['reconstruct']
                # reconstruct_img = seq2img_res['reconstruct']
                # stroke_len = img2seq_res['stroke_len']
                # indices = img2seq_res['indices']
                #
                # self.points2img_8x8(coordinate, stroke_len,
                #                     os.path.join(save_path, str(batch_size).zfill(6) + "_coordinate.png"), indices)
                # self.points2img_8x8(reconstruct_seq, stroke_len,
                #                     os.path.join(save_path, str(batch_size).zfill(6) + "_coordinate_recover.png"), indices)
                # save_image(1 - sketch, os.path.join(save_path, str(batch_size).zfill(6) + "_sketch.png"))
                # save_image(1 - reconstruct_img, os.path.join(save_path, str(batch_size).zfill(6) + "_sketch_recover.png"))

                for n, output in enumerate(seq2img_reconstructs):
                    save_name = seq2img_res['filename'][n].split('/')[-1].split('.')[0]
                    font = save_name[2:]

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

                    output[seq2img_res['char'][n] + 1][:, :5, :] = torch.tensor([0, 1, 1])[:, None, None]
                    output[seq2img_res['char'][n] + 1][:, :, :5] = torch.tensor([0, 1, 1])[:, None, None]
                    output[seq2img_res['char'][n] + 1][:, -5:, :] = torch.tensor([0, 1, 1])[:, None, None]
                    output[seq2img_res['char'][n] + 1][:, :, -5:] = torch.tensor([0, 1, 1])[:, None, None]
                    save_image(1 - output[1:], os.path.join(save_path, "%s_all.png" % save_name), nrow=26)

                for n, output in enumerate(img2seq_reconstructs):
                    self.points2img_row(output[1:], stroke_lens[n][1:], save_path + '_seq.png')

    def extract_list(self, data_loader, save_path, plot_all=True):
        self.set_model_status(train=False)

        res_filenames, res_labels, res_chars, res_features = [], [], [], []
        with torch.no_grad():
            for idx, data in enumerate(tqdm(data_loader, ascii=True)):
                img2seq_res = self.trainer_img2seq._forward(data, decode_all=plot_all)

                coordinate = data['coordinate'].clone()
                data['coordinate'] = img2seq_res['reconstruct']
                seq2img_res = self.trainer_seq2img._forward(data)
                data['coordinate'] = coordinate

                if plot_all:
                    _, seq2img_res['reconstructs'], _ = self.decode_all(
                        data['sketch'], data['coordinate'], data['char'], img2seq_res['feature'],
                        img2seq_res['stroke_len'])

                res_filenames += img2seq_res['filename']
                res_labels += img2seq_res['label']
                res_chars += img2seq_res['char']
                feature = img2seq_res['feature']
                res_features += (feature / torch.norm(feature, dim=1, keepdim=True))

                if 'reconstructs' in img2seq_res:
                    reconstructs, stroke_lens = img2seq_res['reconstructs'], img2seq_res['stroke_lens']
                    for n, output in enumerate(reconstructs):
                        self.points2img_9x3(output, stroke_lens[n], os.path.join(
                            save_path, "ret_seq_%s.png" % img2seq_res['filename'][n].split('/')[-1].split('.')[0]))

                if 'reconstructs' in seq2img_res:
                    reconstructs = seq2img_res['reconstructs']
                    for n, output in enumerate(reconstructs):
                        save_image(1 - output, os.path.join(
                            save_path, "ret_img_%s.png" % seq2img_res['filename'][n].split('/')[-1].split('.')[0]),
                                   nrow=9)

        res_labels = torch.stack(res_labels).cuda()
        res_chars = torch.stack(res_chars).cuda()
        res_features = torch.stack(res_features).cuda()
        return res_filenames, res_labels, res_chars, res_features

    def extract_dict(self, data_loader):
        self.set_model_status(train=False)

        res_dict = {}
        with torch.no_grad():
            for idx, data in enumerate(tqdm(data_loader, ascii=True)):
                img2seq_res = self.trainer_img2seq._forward(data)

                for c, l, v1, v2, n, f in zip(img2seq_res['char'], img2seq_res['filename'],
                                              img2seq_res['sketch'], img2seq_res['coordinate'],
                                              img2seq_res['stroke_len'], img2seq_res['feature']):
                    _c = c.detach().cpu().item()
                    _l = l.split('_')[3].split('.')[0]

                    if _c in res_dict:
                        res_dict[_c][_l] = (c, l, v1, v2, n, f)
                    else:
                        res_dict[_c] = {_l: (c, l, v1, v2, n, f)}

            return res_dict

    def mulinference(self, data, gts, idx, save_path):
        self.set_model_status(train=False)

        with torch.no_grad():
            img2seq_res = self.trainer_img2seq._forward(data)

            img2seq_reconstructs, seq2img_reconstructs, stroke_lens = self.decode_all(
                sketch=None, coordinate=None, char=data['char'][:1],
                feature=torch.mean(img2seq_res['feature'], dim=0, keepdim=True),
                stroke_len=img2seq_res['stroke_len'][:1])

            for n, output in enumerate(seq2img_reconstructs):
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

            for n, output in enumerate(img2seq_reconstructs):
                self.points2img_row(output[1:], stroke_lens[n][1:], save_path + '_fused_seq.png')

    def interpolate(self, sketch1, sketch2, coordinate1, coordinate2, char,
                    feature1, feature2, stroke_len, save_path, idx=None):
        img2seq_reconstructs, seq2img_reconstructs, stroke_lens = self.decode_all(
            sketch1.unsqueeze(0), coordinate1.unsqueeze(0),
            char.unsqueeze(0), feature1.unsqueeze(0), stroke_len.unsqueeze(0))

        for n, output in enumerate(seq2img_reconstructs):
            if idx:
                output[idx, :, :5, :] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
                output[idx, :, :, :5] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
                output[idx, :, -5:, :] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
                output[idx, :, :, -5:] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
            save_image(1 - output[1:], save_path + '_img_10.png', nrow=26)
        for n, output in enumerate(img2seq_reconstructs):
            self.points2img_row(output[1:], stroke_lens[n][1:], save_path + '_seq_10.png')

        img2seq_reconstructs, seq2img_reconstructs, stroke_lens = self.decode_all(
            sketch2.unsqueeze(0), coordinate2.unsqueeze(0),
            char.unsqueeze(0), feature2.unsqueeze(0), stroke_len.unsqueeze(0))

        for n, output in enumerate(seq2img_reconstructs):
            if idx:
                output[idx, :, :5, :] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
                output[idx, :, :, :5] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
                output[idx, :, -5:, :] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
                output[idx, :, :, -5:] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
            save_image(1 - output[1:], save_path + '_img_0.png', nrow=26)
        for n, output in enumerate(img2seq_reconstructs):
            self.points2img_row(output[1:], stroke_lens[n][1:], save_path + '_seq_0.png')

        inter_char = torch.stack([char] * 9).type(char.dtype).to(char.device)
        inter_feature = torch.stack([feature1 * (0.1 * n) + feature2 * (1 - 0.1 * n)
                                     for n in range(1, 10)]).type(feature1.dtype).to(feature1.device)
        inter_strokeLen = torch.stack([stroke_len] * 9).type(stroke_len.dtype).to(stroke_len.device)

        img2seq_reconstructs, seq2img_reconstructs, stroke_lens = self.decode_all(
            sketch=None, coordinate=None, char=inter_char, feature=inter_feature, stroke_len=inter_strokeLen)

        for n, output in enumerate(seq2img_reconstructs):
            if idx:
                output[idx, :, :5, :] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
                output[idx, :, :, :5] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
                output[idx, :, -5:, :] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
                output[idx, :, :, -5:] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
            save_image(1 - output[1:], save_path + '_img_%s.png' % (n + 1), nrow=26)
        for n, output in enumerate(img2seq_reconstructs):
            self.points2img_row(output[1:], stroke_lens[n][1:], save_path + '_seq_%s.png' % (n + 1))

    def manipulate(self, data, target, idx, save_img_dir, forward_only=False):
        self.set_model_status(train=False)

        idx += 1
        target = torch.tensor(target).cuda().unsqueeze(0)
        sketch = torch.tensor(data['sketch']).cuda().unsqueeze(0)
        coordinate = torch.tensor(data['coordinate']).cuda().unsqueeze(0)
        label = torch.tensor(data['label']).cuda().unsqueeze(0)
        char = torch.tensor(data['char']).cuda().unsqueeze(0)
        flag_bits = torch.tensor(data['flag_bits']).cuda().unsqueeze(0)
        stroke_len = torch.tensor(data['stroke_len']).cuda().unsqueeze(0)
        indices = torch.tensor(data['indices']).cuda().unsqueeze(0)
        attention_masks = data['attention_masks']
        for n in range(len(attention_masks)):
            attention_masks[n] = torch.tensor(attention_masks[n]).cuda().unsqueeze(0)
            attention_masks[n].requires_grad = False

        target.requires_grad = False
        sketch.requires_grad = False
        coordinate.requires_grad = False
        label.requires_grad = False
        char.requires_grad = False
        flag_bits.requires_grad = False
        stroke_len.requires_grad = False
        indices.requires_grad = False

        # Resize inputs
        flag_bits.squeeze_(2)
        indices.squeeze_(2)
        stroke_len.unsqueeze_(1)

        if forward_only:
            _, feature = self.trainer_img2seq.model['encoder'](sketch)
            img2seq_reconstructs, seq2img_reconstructs, stroke_lens = self.decode_all(
                sketch, coordinate, char, feature.unsqueeze(0), stroke_len)

            for n, output in enumerate(seq2img_reconstructs):
                output[idx, :, :5, :] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
                output[idx, :, :, :5] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
                output[idx, :, -5:, :] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
                output[idx, :, :, -5:] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
                save_image(1 - output[1:], os.path.join(save_img_dir, 'img_init.png'), nrow=26)

            for n, output in enumerate(img2seq_reconstructs):
                self.points2img_row(output[1:], stroke_lens[n][1:], os.path.join(
                    save_img_dir, 'seq_init.png'))

            _, tmp = self.trainer_seq2img.model['encoder'](
                target, flag_bits, stroke_len, attention_masks, indices)
            render = self.trainer_seq2img.model['decoder'](tmp, char)

            _, feature = self.trainer_img2seq.model['encoder'](render)
            img2seq_reconstructs, seq2img_reconstructs, stroke_lens = self.decode_all(
                sketch, coordinate, char, feature.unsqueeze(0), stroke_len)

            for n, output in enumerate(seq2img_reconstructs):
                output[idx, :, :5, :] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
                output[idx, :, :, :5] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
                output[idx, :, -5:, :] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
                output[idx, :, :, -5:] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
                save_image(1 - output[1:], os.path.join(save_img_dir, 'img_target.png'), nrow=26)

            for n, output in enumerate(img2seq_reconstructs):
                self.points2img_row(output[1:], stroke_lens[n][1:], os.path.join(
                    save_img_dir, 'seq_target.png'))

            return

        self.logger.info("Extract feature before back propogation")
        logit, feature_init = self.trainer_img2seq.model['encoder'](sketch)
        coor_pred_init, _, _ = self.trainer_img2seq.model['decoder'](
            feature_init.unsqueeze(0), char, attention_masks)

        assert target.shape == coordinate.shape
        AE_loss = nn.MSELoss()(coordinate[:, :, :2], target[:, :, :2])
        COS_loss = nn.MSELoss()(coordinate[:, :, 2:], target[:, :, 2:])
        self.logger.info(
            "GT - AE_loss: {}, COS_loss: {}".format(round(float(AE_loss), 2), round(float(COS_loss), 2)))

        AE_loss = nn.MSELoss()(coordinate[:, :, :2], coor_pred_init[:, :, :2])
        COS_loss = nn.MSELoss()(coordinate[:, :, 2:], coor_pred_init[:, :, 2:])
        self.logger.info(
            "PD - AE_loss: {}, COS_loss: {}".format(round(float(AE_loss), 2), round(float(COS_loss), 2)))

        AE_loss = nn.MSELoss()(target[:, :, :2], coor_pred_init[:, :, :2])
        COS_loss = nn.MSELoss()(target[:, :, 2:], coor_pred_init[:, :, 2:])
        self.logger.info(
            "TG - AE_loss: {}, COS_loss: {}".format(round(float(AE_loss), 2), round(float(COS_loss), 2)))

        feature_init = feature_init.detach().clone().unsqueeze(0)
        feature_init.requires_grad = False

        img2seq_reconstructs, seq2img_reconstructs, stroke_lens = self.decode_all(
            sketch, coordinate, char, feature_init, stroke_len)

        for n, output in enumerate(seq2img_reconstructs):
            output[idx, :, :5, :] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
            output[idx, :, :, :5] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
            output[idx, :, -5:, :] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
            output[idx, :, :, -5:] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]

            save_image(1 - output[1:], os.path.join(
                save_img_dir, 'img_%s.png' % str(0).zfill(6)), nrow=26)
        for n, output in enumerate(img2seq_reconstructs):
            self.points2img_row(output[1:], stroke_lens[n][1:], os.path.join(
                save_img_dir, 'seq_%s.png' % str(0).zfill(6)))

        feature = Variable(feature_init.clone().unsqueeze(0), requires_grad=True)
        self.optimizer = torch.optim.Adam([feature], lr=0.0001)

        coor_preds = []
        for epoch in range(int(1e4)):
            logit = self.trainer_img2seq.model['encoder'].mlp_classifier(feature)
            coor_pred, flag_pred, quant_pred = self.trainer_img2seq.model['decoder'](feature, char, attention_masks)

            XE_loss = nn.CrossEntropyLoss()(logit, label)
            AE_loss = nn.MSELoss()(coor_pred[:, :, :2], target[:, :, :2])
            COS_loss = nn.MSELoss()(coor_pred[:, :, 2:], target[:, :, 2:])
            ref_loss = nn.MSELoss()(feature, feature_init)

            loss = AE_loss + COS_loss + ref_loss
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar("optimize/XE_loss", XE_loss, epoch + 1)
            self.writer.add_scalar("optimize/AE_loss", AE_loss, epoch + 1)
            self.writer.add_scalar("optimize/COS_loss", COS_loss, epoch + 1)
            self.writer.add_scalar("optimize/ref_loss", ref_loss, epoch + 1)

            if (epoch + 1) % 100 == 0:
                self.logger.info("epoch: {}, XE_loss: {}, AE_loss: {}, COS_loss: {}, ref_loss: {}".format(
                    epoch + 1, round(float(XE_loss), 2), round(float(AE_loss), 2), round(float(COS_loss), 2),
                    round(float(ref_loss), 2)))

                img2seq_reconstructs, seq2img_reconstructs, stroke_lens = self.decode_all(
                    None, target, char, feature, stroke_len)

                for n, output in enumerate(seq2img_reconstructs):
                    output[idx, :, :5, :] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
                    output[idx, :, :, :5] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
                    output[idx, :, -5:, :] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]
                    output[idx, :, :, -5:] = torch.tensor([0, 1, 1]).type(output.dtype).to(output.device)[:, None, None]

                    save_image(1 - output[1:], os.path.join(
                        save_img_dir, 'img_%s.png' % str(epoch + 1).zfill(6)), nrow=26)
                for n, output in enumerate(img2seq_reconstructs):
                    self.points2img_row(output[1:], stroke_lens[n][1:], os.path.join(
                        save_img_dir, 'seq_%s.png' % str(epoch + 1).zfill(6)))

                coor_preds.append([coor_pred.detach().cpu().squeeze(), epoch + 1])

            # if (epoch + 1) % 2000 == 0:
            #     self.save_model(save_ckpt_dir, 'model', epoch)

        return coor_pred_init, feature_init, coor_preds, feature
