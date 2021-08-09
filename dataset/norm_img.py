import os
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image


def normalize(img):
    width, height = img.width, img.height

    if width < height:
        new_height = 256
        new_width = int(new_height * width / height)
    else:
        new_width = 256
        new_height = int(new_width * height / width)

    return img.resize((new_width, new_height), Image.ANTIALIAS)


def getf_img(p):
    img = Image.open(p, 'r').convert('RGB')
    exit()  # NEED_TO_CHECK

    return normalize(img)


def getf_npy(p, reverse_color=False):
    if reverse_color:
        img = Image.fromarray(np.uint8(np.load(p)), 'L')  # .convert('RGB')
    else:
        img = Image.fromarray(255 - np.uint8(np.load(p)), 'L')  # .convert('RGB')

    return normalize(img)


def load_img(p):
    return getf_img(p) if p.split('.')[-1] in ['png', 'jpg'] else getf_npy(p)


def load_img_dict(data_path, subset):
    filelist = [f for f in os.listdir(data_path)
                if f.startswith('tiny_%s_img_' % subset) and f.endswith('.pickle')]

    img_dict = {}
    for f in filelist:
        print('loading %s' % os.path.join(data_path, f))
        img_dict.update(pickle.load(open(os.path.join(data_path, f), 'rb')))

    return img_dict


if __name__ == '__main__':
    data_path = '/home/alternative/font_data/google_font_dataset'
    subset = 'val'

    img_dict = load_img_dict(data_path, subset)
    print(len(img_dict))

    img = list(img_dict.values())[0]
    img.save('tmp/img.png')
