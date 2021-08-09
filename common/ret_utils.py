import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from dataset.norm_img import getf_npy

img_path = '/home/alternative/font_data/google_font_svg+img'


def compute_mAP(index, good_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size()[-1] == 0:  # if empty
        cmc[0] = -1
        return ap, cmc, None

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc[:20], index[:20]


def draw_img(fname, bord_color):
    # rescale
    dst = Image.new('RGB', (299, 299), color='white')
    image = getf_npy(os.path.join(img_path, fname.replace('svg', 'npy')), reverse_color=True)
    pos = (299 - image.width) // 2, (299 - image.height) // 2
    dst.paste(image, pos)

    dst_with_border = Image.new('RGB', (310, 310), color=bord_color)
    dst_with_border.paste(dst, (5, 5))

    return dst_with_border


def get_scores(query_labels, query_features, gallery_labels, gallery_features):
    # CMC = torch.IntTensor(len(gallery_labels)).zero_()
    CMC = torch.IntTensor(20).zero_()
    ap = 0.0

    query_labels = query_labels.cpu()
    gallery_labels = gallery_labels.cpu()

    for i, ql in enumerate(tqdm(query_labels)):
        score = torch.mm(gallery_features, query_features[i].view(-1, 1))
        score = score.squeeze(1).cpu().numpy()

        index = np.argsort(score)[::-1]
        good_index = np.argwhere(gallery_labels == ql)

        # bad_index = np.argwhere(query_chars[i] == gallery_chars)
        # good_index = np.setdiff1d(good_index, bad_index, assume_unique=True)

        ap_tmp, CMC_tmp, index_tmp = compute_mAP(index, good_index)

        if CMC_tmp[0] == -1:
            continue

        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC / len(query_labels)  # average CMC
    print('top1: %.2f top5: %.2f top10: %.2f mAP: %.2f' %
          (CMC[0] * 100., CMC[4] * 100., CMC[9] * 100., ap / len(query_labels) * 100.))


def vis_retrieval(query_files, query_labels, query_chars, query_features,
                  gallery_files, gallery_labels, gallery_chars, gallery_features, save_path='./tmp'):
    topk = 5

    query_idx = (query_chars < 26).nonzero().squeeze().to(query_labels.device)
    query_files = [query_files[n] for n in query_idx]
    query_labels = torch.index_select(query_labels, 0, query_idx)
    query_chars = torch.index_select(query_chars, 0, query_idx)
    query_features = torch.index_select(query_features, 0, query_idx)

    gallery_idx = (gallery_chars < 26).nonzero().squeeze().to(gallery_labels.device)
    gallery_files = query_files + [gallery_files[n] for n in gallery_idx]
    gallery_labels = torch.cat([query_labels, torch.index_select(gallery_labels, 0, gallery_idx)])
    gallery_chars = torch.cat([query_chars, torch.index_select(gallery_chars, 0, gallery_idx)])
    gallery_features = torch.cat([query_features, torch.index_select(gallery_features, 0, gallery_idx)])

    print(len(query_idx), len(gallery_idx))
    for i, ql in enumerate(tqdm(query_labels)):
        score = torch.mm(gallery_features, query_features[i].view(-1, 1))
        score = score.squeeze(1).cpu().numpy()
        index = np.argsort(score)[::-1]

        query_labels = query_labels.cpu()
        query_chars = query_chars.cpu()
        gallery_labels = gallery_labels.cpu()
        gallery_chars = gallery_chars.cpu()

        dst = Image.new('RGB', (320 * (topk + 1), 320), color='white')
        query_image = draw_img(query_files[i], 'white')
        dst.paste(query_image, (0, 0))

        good_index, bad_index = [], []
        for idx in index:
            if gallery_files[idx] == query_files[i]:
                continue
            if gallery_chars[idx] == query_chars[i]:
                good_index.append(idx)
            else:
                bad_index.append(idx)

        for count, idx in enumerate(good_index[:topk]):
            bord_color = ('green' if gallery_labels[idx] == query_labels[i] else
                          'blue' if gallery_chars[idx] == query_chars[i] else 'red')
            retrieval_image = draw_img(gallery_files[idx], bord_color)
            dst.paste(retrieval_image, (320 * (count + 1), 0))
        dst.save(os.path.join(save_path, '%s_glyph_%s.png' % (query_chars[i].item(), i)))

        for count, idx in enumerate(bad_index[:topk]):
            bord_color = ('green' if gallery_labels[idx] == query_labels[i] else
                          'blue' if gallery_chars[idx] == query_chars[i] else 'red')
            retrieval_image = draw_img(gallery_files[idx], bord_color)
            dst.paste(retrieval_image, (320 * (count + 1), 0))
        dst.save(os.path.join(save_path, '%s_style_%s.png' % (query_chars[i].item(), i)))
