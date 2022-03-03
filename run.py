# Not made for now. 
import argparse
from engine import *
from model import *
from coco import *
from util import *
import torch
import torchvision


parser = argparse.ArgumentParser(description='WILDCAT Training')
# parser.add_argument('data', default='data',
#                     metavar='DIR', help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=224, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


args = parser.parse_args()
print("Hello World")
print(args)


def main_coco():
    global args, best_prec1, use_gpu
    # args = parser.parse_args()
    root = ''
    use_gpu = torch.cuda.is_available()

    train_dataset = COCO2014(root, phase='val', inp_name='/content/drive/MyDrive/GCN/coco_glove_word2vec (1).pkl')
    val_dataset = COCO2014(root, phase='val', inp_name='/content/drive/MyDrive/GCN/coco_glove_word2vec (1).pkl')
    num_classes = 80
    ml = torchvision.models.resnet50(pretrained=True)
    model = GNNResnet(ml, num_classes=num_classes, t=0.4, adj_file='/content/drive/MyDrive/GCN/coco_adj.pkl')

    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(0.1, 0.1),
                                lr=0.1,
                                momentum=0.9,
                                weight_decay=1e-4)

    state = {'batch_size': 32, 'image_size': 224, 'max_epochs': 20,
             'evaluate': False, 'resume': None, 'num_classes':num_classes}
    state['difficult_examples'] = True
    state['save_model_path'] = 'checkpoint/coco/'
    state['workers'] = 1
    state['epoch_step'] = [30]
    state['lr'] = 0.1
    state['d_ids'] = [0]
    # if True:
    #     state['evaluate'] = True
    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)