import torch
import torch.nn.functional as F

import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from setup import set_lr_scheduler, select_model, load_dataset, MyOptimizer
from utils.alarm import send_alarm_to_slack

import os.path
import pandas as pd


# train
def train(model, train_loader, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.float().to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target.view_as(predicted)).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total

    return train_loss, train_accuracy


# inception의 보조분류기 때문에 따로 만듦
def train_inception(model, train_loader, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.float().to(device), target.to(device)
        optimizer.zero_grad()
        output, aux_output = model(data)
        loss1 = F.cross_entropy(output, target)
        loss2 = F.cross_entropy(aux_output, target)
        loss = loss1 + 0.4*loss2
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target.view_as(predicted)).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total

    return train_loss, train_accuracy


# test
def evaluate(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.float().to(device), target.to(device)
            output = model(data)

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target.view_as(predicted)).sum().item()

    test_accuracy = 100. * correct / total

    return test_accuracy


def save_data(data, path):
    df = pd.DataFrame(data)
    df = df.transpose()
    if not os.path.exists(path):
        df.to_csv(path, mode ='w')
    else :
        df.to_csv(path, mode='a', header=False)


class Training:
    def __init__(self, options_dict, index_num, device, csv_path):
        self.start_epoch = 1
        self.options_dict = options_dict
        self.index_num =index_num
        self.current_data = {}
        self.device_num = device
        self.csv_path = csv_path
        # '/home/work/test1/result2.csv'

    def apply_option(self, ):
        self.model, self.device = select_model(self.options_dict['net'], self.options_dict['dataset'],
                                                self.device_num)
        self.train_loader, self.val_loader, self.test_loader = load_dataset(batch_size=self.options_dict['batch_size'],
                                                           dataset=self.options_dict['dataset'])
        self.optimizer = MyOptimizer(net=self.model, lr=self.options_dict['initial_lr'],
                                     momentum=self.options_dict['initial_momentum'])\
            .select_optimizer(opt=self.options_dict['optimizer'])
        self.ckpt_info = '{}_{}_{}_{}_lr{}_m{}_{}'.format(self.options_dict['dataset'], self.options_dict['net'],
                                            self.options_dict['optimizer'], self.options_dict['epochs'],
                                            self.options_dict['initial_lr'], self.options_dict['initial_momentum'],
                                            datetime.today().strftime("%Y%m%d-%H%M"))
        ckpt_dir = '/home/work/test1/runs2/'+self.ckpt_info
        self.writer = SummaryWriter(ckpt_dir)

        print("settings [{}]".format(self.ckpt_info))

    def operation(self,):
        # # load pt file
        # temp = '{}_{}_{}_{}_lr{}_m{}'.format(self.options_dict['dataset'], self.options_dict['net'],
        #                                     self.options_dict['optimizer'], 1000,
        #                                     self.options_dict['initial_lr'], self.options_dict['initial_momentum'] )
        # checkpoint = torch.load('/home/work/test1/ckpt/ckpt_'+temp+'.pt')
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.start_epoch = checkpoint['epoch'] + 1
        # print("Loading checkpoint...\nstart ",checkpoint['epoch'],"epoch")

        # # learning rate scheduler
        # lr_scheduler = set_lr_scheduler(optimizer=self.optimizer,
        #                                         epochs=self.start_epoch + self.options_dict['epochs']
        #                                         , last_ep=self.start_epoch - 1)

        if (self.options_dict['net'] == 'inception') or (self.options_dict['net'] == 'inception_1ch'):
            train_network = train_inception
        else:
            train_network = train

        self.current_data[self.index_num] = {
            'batch_size': self.options_dict['batch_size'],
            'dataset': self.options_dict['dataset'],
            'model': self.options_dict['net'],
            'optimizer': self.options_dict['optimizer'],
            'initial_lr': self.options_dict['initial_lr'],
            'initial_momentum': (self.options_dict['initial_momentum'] if self.options_dict['initial_momentum'] != 'Adam' else "")
        }

        print("training start")
        start_time = time.time()

        for epoch in range(self.start_epoch, self.options_dict['epochs']+1):
            train_loss, train_accuracy = train_network(self.model, self.train_loader, self.optimizer, self.device)
            val_accuracy = evaluate(self.model, self.val_loader, self.device)
            # lr_scheduler.step()

            if epoch % 100 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                },
                    '/home/work/test1/ckpt/ckpt_'+self.ckpt_info+'_{}epoch'.format(epoch)+'.pt')

                test_accuracy = evaluate(self.model, self.test_loader, self.device)
                
                self.current_data[self.index_num]['epochs'] = epoch
                self.current_data[self.index_num]['train acc'] = '{:.2f}'.format(train_accuracy)
                self.current_data[self.index_num]['val acc'] = '{:.2f}'.format(val_accuracy)
                self.current_data[self.index_num]['test acc'] = '{:.2f}'.format(test_accuracy)
                self.current_data[self.index_num]['time'] = '{:.2f}'.format((time.time() - start_time))
                save_data(self.current_data, self.csv_path)
                print('Train Accuracy: {:.2f}%, Val Accuracy: {:.2f}%, Test Accuracy: {:.2f}%'.format(
                    train_accuracy, val_accuracy, test_accuracy), epoch, "epoch result saved")
            else:
                print('[{}] Train loss: {:.4f}, Train Acc: {:.2f}%, Val Acc: {:.2f}%, time: {:.2f}s'.format(
                    epoch, train_loss, train_accuracy, val_accuracy, (time.time() - start_time)))

            self.writer.add_scalar('train loss', train_loss, epoch)
            self.writer.add_scalar('train acc', train_accuracy, epoch)
            self.writer.add_scalar('val acc', val_accuracy, epoch)

    def finish(self):
        torch.cuda.empty_cache()
        send_alarm_to_slack(self.ckpt_info+" done")
        print("task done\n")