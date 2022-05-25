import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as T

import yaml
import argparse

from setup import custom_pil_loader, select_model
from trainer import evaluate


def yml_to_dict(filepath):
    with open(filepath) as f:
        taskdict = yaml.load(f, Loader=yaml.FullLoader)
    return taskdict


def initialization(option_dict):
    transforms = T.Compose([
        T.Resize((option_dict['resolution'], option_dict['resolution'])),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])

    # data_path == image folder path
    dataset = datasets.ImageFolder(root=option_dict['data_path'],
                                   transform=transforms, loader=custom_pil_loader)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=3)

    model, device = select_model(option_dict['net'], option_dict['dataset'], option_dict['device'])
    model.load_state_dict(torch.load(option_dict['pt_path'], map_location=device)['model_state_dict'], strict=False)

    return data_loader, model, device


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Image Classification')
    parser.add_argument("--yml_path", required=True, default="./inference_setting.yml",
                        help="path to yml file contains options")
    args = parser.parse_args()

    o_dict = yml_to_dict(args.yml_path)
    test_loader, model, device = initialization(o_dict)
    test_accuracy = evaluate(model, test_loader, device)
    print('Test Accuracy: {:.2f}%'.format(test_accuracy))