import os

import numpy as np

import cv2
from dataloader1 import Datases_loader as dataloader
from torch.utils.data import DataLoader

from tqdm import tqdm


def save_original_img(loader, savepath):
    imgspath = os.path.join(savepath, 'imgs')
    exist1 = os.path.exists(imgspath)
    if not exist1:
        os.makedirs(imgspath)
        print(f'create folder {imgspath}')

    for id, _ in enumerate(loader):
        img = _['image']
        B = img.shape[0]
        for j in range(B):
            sv = img[j].permute(1, 2, 0)
            sv = np.array(sv) * 255
            cv2.imwrite(imgspath + f'\img{id + 1}_{j + 1}.jpg', sv)


def save(savepath, preds, labs):
    size = 512
    predspath = os.path.join(savepath, 'preds')
    labspath = os.path.join(savepath, 'labs')

    exist1 = os.path.exists(predspath)
    if not exist1:
        os.makedirs(predspath)
        print(f'create folder {predspath}')

    exist2 = os.path.exists(labspath)
    if not exist2:
        os.makedirs(labspath)
        print(f'create folder {labspath}')

    num = len(preds)

    for _ in tqdm(range(num)):
        # print(preds[_].shape[0])
        for j in range(preds[_].shape[0]):
            img = preds[_][j].reshape(size, size)
            lab = labs[_][j].reshape(size, size)

            cv2.imwrite(predspath + f'\pred{_ + 1}_{j + 1}.png', img)
            cv2.imwrite(labspath + f'\label{_ + 1}_{j + 1}.png', lab)


def loadnp(filepath):
    preds = []
    labs = []
    filenum = len(os.listdir(filepath))
    print(f'load file num ---- {filenum}')

    for i in range(filenum//2):
        pred = os.path.join(filepath, f'pred{i+1}.npy')
        lab = os.path.join(filepath, f'label{i+1}.npy')
        print(f'load--{pred} and {lab}')
        pred = np.load(pred)
        lab = np.load(lab)
        pred[pred > 0] = 255.
        pred[pred <= 0] = 0.
        lab[lab > 0] = 255.
        lab[lab <= 0] = 0.
        preds.append(pred)
        labs.append(lab)

    print('load numpy ending...')
    return preds, labs


if __name__ == '__main__':
    batchsz = 3
    filepath = r'E:\论文四\537-10次-0.8584\results\Net2_3'
    savepath = r'D:\Desktop\new4\result-537'
    p, l = loadnp(filepath)
    save(savepath, p, l)

    loader = dataloader(r'D:\Desktop\new2\Deepcrack\Deepcrack\test_img', r'D:\Desktop\new2\Deepcrack\Deepcrack\test_img', 512, 512)
    print(f'img num: {loader.num_of_samples()}')
    loader = DataLoader(loader, batch_size=batchsz, shuffle=False)
    save_original_img(loader, savepath)

