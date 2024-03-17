import torch
import torch.nn as nn
from torch.autograd import Variable
# from logger import Logger
import numpy as np
import argparse
from skimage import io

from dataloaderNetM import get_loader
from modelNetMDistill import EncoderNet, DecoderNet, ClassNet, EPELoss
from icecream import ic

parser = argparse.ArgumentParser(description='GeoNetM')
parser.add_argument('--epochs', type=int, default=30000, metavar='N')
parser.add_argument('--reg', type=float, default=0.1, metavar='REG')
parser.add_argument('--lr', type=float, default=0.00005, metavar='LR')
parser.add_argument('--data_num', type=int, default=800, metavar='N')
parser.add_argument('--batch_size', type=int, default=4, metavar='N')
parser.add_argument("--dataset_dir", type=str, default='C:\\Users\\SiChengLi\\Documents\\Code\\ZJU\\GeoProj\\dataset\\moving')
parser.add_argument("--train_dir", type=str, default='C:\\Users\\SiChengLi\\Documents\\Code\\ZJU\\GeoProj\\dataset\\moving\\train_distorted')
parser.add_argument("--val_dir", type=str, default='C:\\Users\\SiChengLi\\Documents\\Code\\ZJU\\GeoProj\\dataset\\moving\\test_distorted')
parser.add_argument("--train_flow_dir", type=str, default='C:\\Users\\SiChengLi\\Documents\\Code\\ZJU\\GeoProj\\dataset\\moving\\train_flow')
parser.add_argument("--val_flow_dir", type=str, default='C:\\Users\\SiChengLi\\Documents\\Code\\ZJU\\GeoProj\\dataset\\moving\\test_flow')
parser.add_argument("--distortion_type", type=list, default=['barrel'])
args = parser.parse_args()

use_GPU = torch.cuda.is_available()
ic( '%s%s' % (args.dataset_dir, '/train_distorted'))

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

train_loader = get_loader(distortedImgDir = r'C:\Users\SiChengLi\Documents\Code\ZJU\GeoProj\dataset\moving\train_distorted',
                flowDir = args.train_flow_dir,
                  batch_size = args.batch_size,
                  distortion_type = args.distortion_type,
                  data_num = args.data_num)
print("train_loader done",len(train_loader))
val_loader = get_loader(distortedImgDir = args.val_dir,
                flowDir = args.val_flow_dir,
                batch_size = args.batch_size,
                distortion_type = args.distortion_type,
                data_num = int(args.data_num*0.1) + 50000)


model_en = EncoderNet([1,1,1,1,2])
model_de = DecoderNet([1,1,1,1,2])

model_class = ClassNet()
total_params_en = sum(p.numel() for p in model_en.parameters())
total_params_de = sum(p.numel() for p in model_de.parameters())
total_params_class = sum(p.numel() for p in model_class.parameters())
total_params = total_params_en + total_params_de + total_params_class
total_size = total_params * 4  # for float32 weights
total_size_mb = total_size / (1024 * 1024)  # convert bytes to megabytes

print('EncoderNet: {} parameters, {} MB'.format(total_params_en, total_size_mb))
print('DecoderNet: {} parameters, {} MB'.format(total_params_de, total_size_mb))
print('ClassNet: {} parameters, {} MB'.format(total_params_class, total_size_mb))
print('Total: {} parameters, {} MB'.format(total_params, total_size_mb))

print(count_parameters(model_en))
print(count_parameters(model_de))
print(count_parameters(model_class))

criterion = EPELoss()
criterion_clas = nn.CrossEntropyLoss()

print('dataset type:',args.distortion_type)
print('dataset number:',args.data_num)
print('batch size:', args.batch_size)
print('epochs:', args.epochs)
print('lr:', args.lr)
print('reg:', args.reg)
print('train_loader',len(train_loader), 'train_num', args.batch_size*len(train_loader))
print('val_loader', len(val_loader),   'test_num', args.batch_size*len(val_loader))


if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model_en = nn.DataParallel(model_en)
    model_de = nn.DataParallel(model_de)
    model_class = nn.DataParallel(model_class)

if torch.cuda.is_available():
    model_en = model_en.cuda()
    model_de = model_de.cuda()
    model_class = model_class.cuda()
    criterion = criterion.cuda()
    criterion_clas = criterion_clas.cuda()

reg = args.reg
lr = args.lr
optimizer = torch.optim.Adam(list(model_en.parameters()) + list(model_de.parameters()) + list(model_class.parameters()), lr=lr)

step = 0
# logger = Logger('./logs')
# model_en.state_dict(torch.load("./model_en_401.pkl"))
# model_de.state_dict(torch.load("./model_de_401.pkl"))
# model_class.state_dict(torch.load("./model_class_401.pkl"))


model_en.train()
model_de.train()
model_class.train()

for epoch in range(402,args.epochs):
    for i, (disimgs, disx, disy, labels) in enumerate(train_loader):

        if use_GPU:
            disimgs = disimgs.cuda()
            disx = disx.cuda()
            disy = disy.cuda()
            labels = labels.cuda()

        disimgs = Variable(disimgs)
        labels_x = Variable(disx)
        labels_y = Variable(disy)
        labels_clas = Variable(labels)
        flow_truth = torch.cat([labels_x, labels_y], dim=1)

        # Forward + Backward + Optimize
        optimizer.zero_grad()

        middle = model_en(disimgs)
        flow_output = model_de(middle)
        clas = model_class(middle)

        loss1 = criterion(flow_output, flow_truth)
        loss2 = criterion_clas(clas, labels_clas)*reg

        loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        print("Epoch [%d], Iter [%d], Loss: %.4f" %(epoch + 1, i + 1, loss.item()))

        #============ TensorBoard logging ============#
        step = step + 1
        #Log the scalar values
        info = {'loss': loss.item()}
        for tag, value in info.items():
            pass
    if epoch % 50 == 0:
        torch.save(model_en.state_dict(), '%s%s%s' % ('model_en_',epoch + 1,'.pkl'))
        torch.save(model_de.state_dict(), '%s%s%s' % ('model_de_',epoch + 1,'.pkl'))
        torch.save(model_class.state_dict(), '%s%s%s' % ('model_class_',epoch + 1,'.pkl'))

torch.save(model_en.state_dict(), 'model_en_last.pkl')
torch.save(model_de.state_dict(), 'model_de_last.pkl')
torch.save(model_class.state_dict(), 'model_class_last.pkl')

# Test
total = 0

model_en.eval()
model_de.eval()
model_class.eval()

for i, (disimgs, disx, disy, labels) in enumerate(val_loader):

    if use_GPU:
        disimgs = disimgs.cuda()
        disx = disx.cuda()
        disy = disy.cuda()
        labels = labels.cuda()

    disimgs = Variable(disimgs)
    labels_x = Variable(disx)
    labels_y = Variable(disy)
    labels_clas = Variable(labels)
    flow_truth = torch.cat([labels_x, labels_y], dim=1)

    middle = model_en(disimgs)
    flow_output = model_de(middle)
    clas = model_class(middle)

    loss1 = criterion(flow_output, flow_truth)
    loss2 = criterion_clas(clas, labels_clas)*reg

    loss = loss1 + loss2

    total = total + loss.item()
    print(loss.item(), loss1.item(), loss2.item())

print('val loss',total/(i+1))

