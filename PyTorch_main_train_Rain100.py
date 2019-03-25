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

from PyTorch_dataset_Rain100 import Rain100
from PyTorch_GuidedFilter import guided_filter
from skimage import measure

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
parser.add_argument('--patch_size', default=64)
parser.add_argument('--input_path', default="./Data/Rain100_Train/rain/")
parser.add_argument('--gt_path', default="./Data/Rain100_Train/norain/")
parser.add_argument('--ext', default='img')
# model
# parser.add_argument('--model', default='DnCNN_hanshu', type=str)
parser.add_argument('--batch_size', default=20, type=int)
parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--milestones', default=[50,100,150,170], type=list)
parser.add_argument('--device_ids', type=list, default=[0,1,2,3])
# log
parser.add_argument('--save_dir', type=str, default="./model/rain100-trial")
args = parser.parse_args()

cuda = torch.cuda.is_available()
assert cuda

save_dir = args.save_dir
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(seed=0)
    torch.manual_seed(0)

    # model selection
    print('===> Building model')
    model = DeRain()
    initial_epoch=0
    # model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
    criterion = nn.MSELoss()

    # criterion = sum_squared_error()
    if cuda:
        model = model.cuda()
        device_ids = args.device_ids
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
        criterion = criterion.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)  # learning rates

    # dataset
    DDataset = Rain100(args, isTrain=True)
    DLoader = DataLoader(dataset=DDataset, num_workers=12, drop_last=True, batch_size=args.batch_size, shuffle=True)
    DDataset_eval = Rain100(args, isTrain=False)
    DLoader_eval = DataLoader(dataset=DDataset_eval, num_workers=1, drop_last=True, batch_size=1, shuffle=False)

    # add log
    log_file = os.path.join(save_dir,'train_result.txt')
    with open(log_file,'a') as f:
        f.write('----Begin logging----')
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))
        f.write('================ Training loss ================\n')

    # trainning
    best_epoch = {'epoch':0, 'psnr':0}
    for epoch in range(initial_epoch, args.epoch):

        scheduler.step(epoch)  # step to the learning rate in this epcoh
        epoch_loss = 0
        start_time = time.time()

        # training phase
        model.train()
        for n_count, batch_tr in enumerate(DLoader):
                optimizer.zero_grad()
                batch_n = batch_tr[0].cuda(); batch_c= batch_tr[1].cuda()
                # import pdb; pdb.set_trace()
                loss = criterion(model(batch_n), batch_c)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                if n_count % 50 == 0:
                    message = '[{}] {} / {} loss = {}'.format(epoch+1, n_count, len(DLoader), loss.item()/args.batch_size)
                    print(message)
                    with open(log_file,'a') as f:
                        f.write(message); f.write('\n')
        elapsed_time = time.time() - start_time

        # evaluation phase
        model.eval()
        psnrs = []
        with torch.no_grad():
            i = 0
            for _, batch_eval in enumerate(DLoader_eval):
                i +=1
                img_c = batch_eval[1].cuda()
                img_c = img_c.view(img_c.shape[1],img_c.shape[2], img_c.shape[3]).cpu().numpy().astype(np.float32)
                img_c = np.transpose(img_c, (1,2,0))

                img_dn = model(batch_eval[0].cuda())
                img_dn = img_dn.view(img_dn.shape[1],img_dn.shape[2], img_dn.shape[3]).cpu().numpy().astype(np.float32)
                img_dn = np.transpose(img_dn, (1,2,0))

                psnr_dn = measure.compare_psnr(img_c, img_dn)
                psnrs.append(psnr_dn)
        psnr_avg = np.mean(psnrs)

        # add log
        if psnr_avg > best_epoch['psnr']:
            torch.save(model, os.path.join(save_dir, 'model_best.pth'))
            best_epoch['psnr'] = psnr_avg
            best_epoch['epoch'] = epoch+1

        message1 ='epcoh = {:03d}, [time] = {:.2f}s, [PSNR of {}-imgs]:{:.3f}, [loss] = {:.7f}.'.format(epoch+1, elapsed_time, i, psnr_avg, epoch_loss)
        message2 ='Best @ {:03d}, with value {:.3f}. \n'.format(best_epoch['epoch'], best_epoch['psnr'])
        print(message1)
        print(message2)
        with open(log_file,'a') as f:
            f.write(message1)
            f.write(message2)
        # torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        torch.save(model, os.path.join(save_dir, 'model_latest.pth'))


