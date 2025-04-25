import torch
import numpy as np
from torch.utils.data import DataLoader
from Net11_5 import Net
import os
from dataloader2 import Datases_loader as dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchsz = 1
model= Net().to(device)
model= model.eval()
savedir = r'C:\Users\15504\Desktop\new9\weights\Net2_3.pth'
#imgdir = r'C:\Users\15504\Desktop\new6\Deepcrack\CrackTree260\test_img'
#labdir = r'C:\Users\15504\Desktop\new6\Deepcrack\CrackTree260\test_lab'
#imgdir = r'C:\Users\15504\Desktop\new6\Deepcrack\Crack315\Crack315\test_img'
#labdir = r'C:\Users\15504\Desktop\new6\Deepcrack\Crack315\Crack315\test_lab'
imgdir = r'C:\Users\15504\Desktop\new6\Deepcrack\Deepcrack\test_img'
labdir = r'C:\Users\15504\Desktop\new6\Deepcrack\Deepcrack\test_lab'

imgsz = 512
resultsdir = r'C:\Users\15504\Desktop\new9\results\Net2_3'

dataset = dataloader(imgdir, labdir, imgsz, imgsz)
testsets = DataLoader(dataset, batch_size=batchsz, shuffle=False)

def test():
    model.load_state_dict(torch.load(savedir))
    exist = os.path.exists(resultsdir)
    if not exist:
        os.makedirs(resultsdir)
    for idx, samples in enumerate(testsets):
        img, lab = samples['image'], samples['mask']
        img, lab = img.to(device), lab.to(device)

        pred = model(img)

        np.save(resultsdir + r'/pred' + str(idx+1) + '.npy', pred.detach().cpu().numpy())
        np.save(resultsdir + r'/label' + str(idx+1) + '.npy', lab.detach().cpu().numpy())

if __name__ == '__main__':
    test()