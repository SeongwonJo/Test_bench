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

from utils.utils import yml_to_dict, custom_pil_loader
from setup import Initializer, MyDataset
from trainer import evaluate
import classification_settings


def ArgumentParse(_intro_msg, L_Param, bUseParam=False):
    parser = argparse.ArgumentParser(
        prog='inference.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(_intro_msg))

    parser.add_argument('-y', '--yml_path', default="./inference_settings.yml",
                        help="path to yml file contains options")
    parser.add_argument('-i', '--image_path', default="./test2")
    parser.add_argument('-o', '--operation', default="all")
    parser.add_argument('-d', '--device', default="cuda:0")

    args = parser.parse_args()
    return args


class Inference:
    def __init__(self, L_Param=None):
        self.args       = ArgumentParse(_description, L_Param, bUseParam=False)

        _model, _device = self.initialization(device=self.args.device)
        self.model      = _model
        self.device     = _device

    @staticmethod
    def initialization(device):
        initializer = Initializer(net=classification_settings.infer_net,
                                  dataset=classification_settings.infer_dataset,
                                  device_num=device)
        model = initializer.model
        device = initializer.device
        model.load_state_dict(torch.load(classification_settings.infer_pt_path,
                                         map_location=device)['model_state_dict'], strict=False)

        return model, device

    @staticmethod
    def torch_transformed(root="./test", batch_size=16):
        my_dataset_root = root
        _transforms = classification_settings.transforms_T

        # data_path == image folder path
        dataset = datasets.ImageFolder(root=my_dataset_root,
                                       transform=_transforms, loader=custom_pil_loader)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3)

        return data_loader

    # class 폴더에 분류가 되어있는 경우
    def operation_all(self):
        data_loader = self.torch_transformed(root=self.args.image_path)
        test_accuracy = evaluate(self.model, data_loader, self.device)
        print('prediction Accuracy: {:.2f}%'.format(test_accuracy))

    def operation_onebyone(self):
        _transforms = classification_settings.transforms_T

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
                output = self.model(data.unsqueeze(0).to(self.device))

                _score, predicted = output.max(1)
                pred = label_tags[predicted.item()]
                r_list.append([pred, _score.item()])

        r_dict = dict(zip(f_list, r_list))
        print('prediction result: {file : [prediction, score]}\n', r_dict)

    def operation_one_image(self):
        _transforms = classification_settings.transforms_T
        img = custom_pil_loader(self.args.image_path)
        transformed = _transforms(img)

        label_tags = {
            0: 'Normal',
            1: 'Pneumonia',
        }
        self.model.eval()

        with torch.no_grad():
            output = self.model(transformed.unsqueeze(0).to(self.device))

            _score, predicted = output.max(1)
            pred = label_tags[predicted.item()]

        print('prediction result: %s  Score: %f' %(pred, _score))

    def selected_operation(self,):
        ops = {
            'all': self.operation_all(),
            'onebyone': self.operation_onebyone(),
            'one': self.operation_one_image()
        }.get(self.args.operation, "Please check 'operation' input.")


if __name__ == '__main__':
    infer = Inference()
    infer.selected_operation()

