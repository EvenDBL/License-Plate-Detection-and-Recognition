import argparse
import os
import torch
from model.segnet import SegNet
import torch.optim as optim
from tool import utils
from tool.summary_writer import Writer
import numpy as np
from seg_eval import Eval
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data.seg_dataset import CCPDDataset
from data.seg_dataset import CollectFN
import numpy as np

parser = argparse.ArgumentParser(description='Text Detection Training')
parser.add_argument('--data_dir', default='/home/admin1/datasets/CLPD', type=str, help='Image path for training')
parser.add_argument('--train_list', default='/home/admin1/datasets/CLPD/CLPD_train.txt', type=str, help='Train text for training')
parser.add_argument('--valid_list', default='/home/admin1/datasets/CLPD/CLPD_test.txt', type=str, help='Valid text for training')
parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--num_workers', default=4, type=int, help='Number of dataloader workers')

parser.add_argument('--displayInterval', type=int, default=20, help='Interval to be displayed')
parser.add_argument('--valInterval', type=int, default=1120, help='Interval to be valid')# 1120
parser.add_argument('--saveInterval', type=int, default=5000, help='Interval to be save')

parser.add_argument('--display', action='store_true',  help='display maps in tensorboard')
parser.add_argument('--expr_dir', type=str, default='experiments', help='weights save to path for training')
parser.add_argument('--pre_trained', default='experiments/clpdSegNet_900_final_778950.pth', type=str, help='pre-trained model')
parser.add_argument('--pre_backbone', default='', type=str, help='pre-trained model of resnet18') # resnet18-5c106cde.pth
parser.add_argument('--epochs', default=1800, type=int, help='Number of training epochs')

# parser.add_argument('--resume', default='weights/pre-trained-model-synthtext-resnet18', type=str, help='Resume from checkpoint')
parser.add_argument('--validate', action='store_true', dest='validate', help='Validate during training')
parser.add_argument('--visualize', action='store_true', help='visualize maps in tensorboard')
args = parser.parse_args()
print(args)

def val(net, eval_operator):

    net.eval()
    print('Start val')
    precision, recall, hmean, loss_val, display_data = eval_operator.eval(net)
    print('precision: %f   recall: %f   hmean: %f   valid loss: %f'%(precision, recall, hmean, loss_val))
    net.train()
    return precision, recall, hmean, loss_val, display_data

def load_pretrained(pretrained_path, net):
    print('loading pretrained model from %s' %pretrained_path)
    steps_done = 0
    epochs_done = 0
    if 'pretrained' in pretrained_path:
        print('use pretrained for training')
    else:
        weight_path = pretrained_path
        steps_done = int(weight_path.split('/')[1].split('.')[0].split('_')[-1])
        epochs_done = int(weight_path.split('/')[1].split('.')[0].split('_')[-3])
    net.load_state_dict(torch.load(pretrained_path))

    return steps_done, epochs_done

def update_learning_rate(optimizer, epoch, total_epochs, init_lr):

    rate = np.power(1.0 - epoch / float(total_epochs + 1), 0.9)
    new_lr = rate * init_lr

    for group in optimizer.param_groups:
        group['lr'] = new_lr
    current_lr = new_lr
    return current_lr

def get_dataset_name(path):
    dataset_name = ''
    if 'CCPD' in path:
        if 'base' in path:
            dataset_name = 'ccpd_base'
        elif 'db' in path:
            dataset_name = 'ccpd_db'
        elif 'fn' in path:
            dataset_name = 'ccpd_fn'
        elif 'rotate' in path:
            dataset_name = 'ccpd_ro'
        elif 'weather' in path:
            dataset_name = 'ccpd_we'
        elif 'challenge' in path:
            dataset_name = 'ccpd_ch'
        elif 'tilt' in path:
            dataset_name = 'ccpd_ti'
        # dataset_name = 'ccpd'
    elif 'ALPR' in path:
        dataset_name = 'alpr'
    elif 'CLPD' in path:
        dataset_name = 'clpd'
    elif 'AOLP' in path:
        if 'AC' in path:
            dataset_name = 'aolpAc'
        elif 'RP' in path:
            dataset_name = 'aolpRp'
        elif 'LE' in path:
            dataset_name = 'aolpLe'
    if dataset_name != '':
        return dataset_name
    else:
        raise Exception('数据集路径不合法:%s'%path)

if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    dataset_name = get_dataset_name(args.data_dir)

    net = None
    device = torch.device('cuda')
    if args.pre_backbone != '':
        pre_backbone = args.pre_backbone
        net = SegNet(device, pre_backbone=pre_backbone)
    else:
        net = SegNet(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    ccpd_train_dataset = CCPDDataset(args.data_dir, args.train_list, use_argument=True)
    ccpd_valid_dataset = CCPDDataset(args.data_dir, args.valid_list, use_argument=False)
    train_loader = DataLoader(ccpd_train_dataset, args.batch_size, collate_fn=CollectFN())
    valid_loader = DataLoader(ccpd_valid_dataset, args.batch_size, collate_fn=CollectFN())

    epochs_done = 0
    steps_done = 0

    if args.pre_trained != '':
        steps_done, epochs_done = load_pretrained(args.pre_trained, net)

    loss_avg_train = utils.averager()
    eval_operator = Eval(valid_loader, iou_threshold=0.5, display=args.display)

    writer_t = Writer(path='summary/%s/train'%dataset_name)
    writer_v = Writer(path='summary/%s/valid'%dataset_name)
    total_epochs = args.epochs
    while(epochs_done<total_epochs):
        net.train()
        i=0
        current_lr = update_learning_rate(optimizer, epochs_done, total_epochs, args.lr)
        writer_t.add_scalar('lr', current_lr, steps_done)
        for batch in train_loader:

            loss, pred = net(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            steps_done += 1
            i += 1
            loss_avg_train.add(loss)

            if steps_done % args.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f Steps:%d learn_rate:%f' %
                      (epochs_done, args.epochs, i, len(train_loader), loss_avg_train.val(), steps_done, current_lr))
                writer_t.add_scalar(scope_name='loss', y=loss_avg_train.val(), x=steps_done)
                loss_avg_train.reset()

            if steps_done % args.valInterval == 0:
                precision, recall, hmean, loss_val, display_data = val(net,eval_operator)
                if display_data is not None:
                    display_img = display_data['image']
                    display_img_with_poly = display_data['img_with_poly']
                    display_mask_pre = display_data['mask_pre']
                    display_mask = display_data['mask']

                    # display_map_pre = display_data['heat_map']
                    # h, w = batch['original_size'][0]
                    # figure = plt.figure('heat_map', figsize=(w/100, h/100))
                    # heat_map = sns.heatmap(display_map_pre, xticklabels=False, yticklabels=False, vmin=0, vmax=1)
                    # plt.close()
                    # writer_v.add_figure('display/heat_map', figure)

                    writer_v.add_image('display/img', display_img)
                    writer_v.add_image('display/img_with_poly', display_img_with_poly)
                    writer_v.add_image('display/mask_pre', display_mask_pre)
                    writer_v.add_image('display/mask', display_mask)

                writer_v.add_scalars('pre_rec_Fm', {'precision':precision,'recall':recall,'hmean':hmean}, steps_done)
                writer_v.add_scalar('loss', loss_val, steps_done)

            if steps_done % args.saveInterval == 0:
                torch.save(net.state_dict(), '{0}/{1}SegNet_{2}_{3}_{4}.pth'.format(args.expr_dir, dataset_name, epochs_done, i, steps_done))

        epochs_done += 1

    torch.save(net.state_dict(), '{0}/{1}SegNet_{2}_final_{3}.pth'.format(args.expr_dir, dataset_name, epochs_done, steps_done))
