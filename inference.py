#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - Medical Chest X-ray classification Test
# 2020 07 31 by Seongwon Jo
###########################################################################
_description = '''\
====================================================
inference.py : T
                    Written by Seongwon Jo @ 2022-07-31
====================================================
Example : python inference.py -o "all" -i ./test2
          python inference.py -o "onebyone" -i ./test
          python inference.py -o "one" -i ./test/BACTERIA-134339-0001.jpeg  
'''
import argparse
import textwrap
import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import pprint

from utils.utils import yml_to_dict, custom_pil_loader, Transforms
from setup import Initializer, MyDataset
from trainer import evaluate
import classification_settings


def ArgumentParse(_intro_msg, L_Param, bUseParam=False):
    parser = argparse.ArgumentParser(
        prog='inference.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(_intro_msg))

    parser.add_argument('-y', '--yml_path', default="./train_options.yml",
                        help="path to yml file contains options")
    parser.add_argument('-i', '--image_path', default="./test2")
    parser.add_argument('-o', '--order', default="all")
    parser.add_argument('-d', '--device', default="cuda:0")
    parser.add_argument('-p', '--pt_path', required=True ,help="please input pt file path")

    args = parser.parse_args()
    return args


class Inference:
    def __init__(self, L_Param=None):
        self.args       = ArgumentParse(_description, L_Param, bUseParam=False)
        # self.args       = ArgumentParse()
        self.o_dict     = yml_to_dict(self.args.yml_path)['parameters']

        _model, _device = self.initialization(device=self.args.device)
        self.model      = _model
        self.device     = _device

    def initialization(self, device):
        initializer = Initializer(net=self.o_dict['net'][0],
                                  dataset=self.o_dict['dataset'][0],
                                  device_num=device)
        model = initializer.model
        device = initializer.device
        model.load_state_dict(torch.load(self.args.pt_path,
                                         map_location=device)['model_state_dict'], strict=False)

        return model, device

    @staticmethod
    def loader_with_transforms(root="./test", batch_size=16):
        my_dataset_root = root
        # _transforms = classification_settings.transforms_T
        _transforms = Transforms(classification_settings.custom_test)

        # data_path == image folder path
        dataset = datasets.ImageFolder(root=my_dataset_root,
                                       transform=_transforms, loader=custom_pil_loader)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3)

        return data_loader

    # class 폴더에 분류가 되어있는 경우
    def operation_all(self):
        data_loader = self.loader_with_transforms(root=self.args.image_path)
        test_accuracy, report , _l = evaluate(self.model, data_loader, self.device, is_test=True)
        print("=============================================")
        # pprint.pprint(report)
        print('prediction Accuracy: {:.2f}%'.format(report['accuracy']*100))
        print("=============================================")
        print('details')
        print("---------------------------------------------")
        print('NORMAL')
        pprint.pprint(report['NORMAL'])
        print("\n")
        print('PNEUMONIA')
        pprint.pprint(report['PNEUMONIA'])
        print("=============================================")


    def operation_onebyone(self):
        _transforms = Transforms(classification_settings.custom_test)

        f_list = os.listdir(self.args.image_path)
        transformed_list = []
        for file in f_list:
            img = custom_pil_loader(self.args.image_path + '/' + file)
            transformed_list.append(_transforms(img))

        label_tags = {
            0: 'Normal',
            1: 'Pneumonia',
        }
        r_list = []

        self.model.eval()

        with torch.no_grad():
            for data in transformed_list:
                output = self.model(data.unsqueeze(0).float().to(self.device))

                _score, predicted = output.max(1)
                pred = label_tags[predicted.item()]
                r_list.append([pred, _score.item()])

        r_dict = dict(zip(f_list, r_list))
        print('prediction result: {file : [prediction, score]}\n')
        print("="*80)
        pprint.pprint(r_dict)
        print("="*80)

    def operation_one_image(self):
        _transforms = Transforms(classification_settings.custom_test)
        img = custom_pil_loader(self.args.image_path)
        transformed = _transforms(img)

        label_tags = {
            0: 'Normal',
            1: 'Pneumonia',
        }
        self.model.eval()

        with torch.no_grad():
            output = self.model(transformed.unsqueeze(0).float().to(self.device))

            _score, predicted = output.max(1)
            pred = label_tags[predicted.item()]

        print('prediction result: %s  Score: %f' %(pred, _score))

    def selected_operation(self,):
        if self.args.order == 'all':
            self.operation_all()
        elif self.args.order == "onebyone":
            self.operation_onebyone()
        elif self.args.order == "one":
            self.operation_one_image()
        else:
            print("Please check 'operation' input.")
        # ops = {
        #     'all': self.operation_all(),
        #     'onebyone': self.operation_onebyone(),
        #     'one': self.operation_one_image()
        # }.get(self.args.order, "Please check 'operation' input.")


if __name__ == '__main__':
    infer = Inference()
    infer.selected_operation()

