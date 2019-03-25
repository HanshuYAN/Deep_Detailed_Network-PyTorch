import argparse
import re
import os, glob, datetime, time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from PyTorch_dataset_Rain100 import Rain100_Test
from PyTorch_GuidedFilter import guided_filter
from skimage import measure
import skimage.io as io

# network structure
class DeRain(nn.Module):
    def __init__(self, n_features=16, n_channels=3, use_bnorm=True, kernel_size=3, padding = 1):
        super(DeRain, self).__init__()
        
        # layer 1, block 1
        layers = []
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_features, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.BatchNorm2d(n_features, eps=0.001, momentum = 0.99))
        layers.append(nn.ReLU())
        self.b1 = nn.Sequential(*layers)
        # layers 2 to 25, block 2
        layers = []
        for i in range(12):
            layers.append(nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size, padding=padding, bias=True))
            layers.append(nn.BatchNorm2d(n_features, eps=0.001, momentum = 0.99))
            layers.append(nn.ReLU())

            layers.append(nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size, padding=padding, bias=True))
            layers.append(nn.BatchNorm2d(n_features, eps=0.001, momentum = 0.99))
            layers.append(nn.ReLU())
        self.b2 = nn.Sequential(*layers)

        # layer 26, block 3
        layers = []
        layers.append(nn.Conv2d(in_channels=n_features, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.BatchNorm2d(n_channels, eps=0.001, momentum = 0.99))
        self.b3 = nn.Sequential(*layers)

        self._initialize_weights()

    def forward(self, images):
        # base
        base = guided_filter(images, images, 15, 1, nhwc=True)
        detail = images - base
        output_shortcut = self.b1(detail)
        output_shortcut = self.b2(output_shortcut) + output_shortcut
        neg_residual = self.b3(output_shortcut)
        
        final_out = images + neg_residual
        return final_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                # init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)



# Params
parser = argparse.ArgumentParser(description='PyTorch DeRain')
# dataset
parser.add_argument('--input_path_test', default="./Data/Rain100_Test/rain/X2")
parser.add_argument('--gt_path_test', default="./Data/Rain100_Test/norain/")
# model and device
parser.add_argument('--model_dir', type=str, default="./model/rain100L")
parser.add_argument('--device_ids', type=list, default=[0])
# results
parser.add_argument('--output_dir', type=str, default="./experiments/derain")
args = parser.parse_args()

cuda = torch.cuda.is_available()
assert cuda

output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(seed=0)
    torch.manual_seed(0)

    # model selection
    print('===> Building model')
    model = torch.load(os.path.join(args.model_dir, 'model_best.pth'))
    model.eval()  # evaluation mode

    if cuda:
        model = model.cuda()

    # dataset
    DDataset_test = Rain100_Test(args)
    DLoader_test = DataLoader(dataset=DDataset_test, num_workers=1, drop_last=True, batch_size=1, shuffle=False)

    # add log
    log_file = os.path.join(output_dir,'test_results.txt')
    with open(log_file,'a') as f:
        f.write('----Begin logging----')
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))

    # trainning
    psnrs = []
    with torch.no_grad():
        i = 0
        for _, batch_test in enumerate(DLoader_test):
            print(i)
            i +=1
            img_c = batch_test[1].cuda()
            img_c = img_c.view(img_c.shape[1],img_c.shape[2], img_c.shape[3]).cpu().numpy().astype(np.float32)
            img_c = np.transpose(img_c, (1,2,0))

            img_dn = model(batch_test[0].cuda())
            img_dn = img_dn.view(img_dn.shape[1],img_dn.shape[2], img_dn.shape[3]).cpu().numpy().astype(np.float32)
            img_dn = np.transpose(img_dn, (1,2,0))
            psnr_dn = measure.compare_psnr(img_c, img_dn)
            # import pdb; pdb.set_trace()
            save_file = batch_test[2][0] + '.png'
            
            # save_file = '{:03d}.png'.format(i)
            io.imsave(os.path.join(output_dir, save_file), np.clip(img_dn, 0, 1))

            psnrs.append(psnr_dn)
    psnr_avg = np.mean(psnrs)
    message1 ='[PSNR of {}-imgs]:{:.3f}.'.format(i, psnr_avg)
    
    print(message1)
    with open(log_file,'a') as f:
        f.write(message1)


