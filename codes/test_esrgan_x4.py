#from dataset LR  x2 set5  to  result
import os.path as osp
import glob
import cv2
import numpy as np
import torch
from models import  RRDBNet_arch as arch
import  os

model_paths = '/content/drive/Shared drives/songyi1999/srganbackup/experiments/002_RRDB_ESRGANx2_DIV2K/models/'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
test_img_folder = '/content/Set14/LRbicx4/*.*'
result_folder='/content/drive/Shared drives/songyi1999/srganbackup/results_set14/'


def build_image(model_path):
    save_folder= result_folder +  model_path.split('/')[-1].split('_')[0]
    cmd="mkdir  -p '"+save_folder+"'"
    print(cmd)
    os.system(cmd)
    device = torch.device('cuda')
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    
    print('Model path {:s}. \nTesting...'.format(model_path))
    
    idx = 0
    for path in glob.glob(test_img_folder):
        idx += 1
        base = osp.splitext(osp.basename(path))[0]
        print(idx, base)
        
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)
    
        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        cv2.imwrite('{}/{}.png'.format(save_folder,base), output)




model_list=glob.glob(model_paths+"/*_G.pth")
# dirlist= [ x.split('/')[-1].split('_')[0]    for x in  model_list  ]
# print(dirlist)
for model_path in  model_list:
    build_image(model_path)
