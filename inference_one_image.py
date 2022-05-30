import yaml
import argparse
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet
from models import densenet_1ch


def yml_to_dict(filepath):
    with open(filepath) as f:
        taskdict = yaml.load(f, Loader=yaml.FullLoader)
    return taskdict


# load image on dataloader with grayscale
def custom_pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img.load()
        return img.convert('L')


# resnet200
def resnet200(pretrained: bool = False, progress: bool = True, **kwargs) -> resnet.ResNet:
    return resnet._resnet('resnet200', resnet.Bottleneck, [3, 24, 36, 3], pretrained, progress,
                   **kwargs)


# select model
def select_model(net, dataset, device_num):
    num_classes = {
        'custom': 2,
    }.get(dataset, "error")

    model = {
        'resnet18': resnet.resnet18(num_classes=num_classes),
        'resnet50': resnet.resnet50(num_classes=num_classes),
        'resnet101': resnet.resnet101(num_classes=num_classes),
        'resnet152': resnet.resnet152(num_classes=num_classes),
        'resnet200': resnet200(num_classes=num_classes),
        'densenet121': densenet_1ch.densenet121(num_classes=num_classes),
        'densenet169': densenet_1ch.densenet169(num_classes=num_classes),
        'densenet201': densenet_1ch.densenet201(num_classes=num_classes),
    }.get(net, "error")

    if model == "error":
        raise ValueError('please check your "net" input.')

    if net[0:6] == 'resnet':
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    device = torch.device(device_num if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("device:", device)

    return model, device


def transformed(option_dict, image_path):
    transforms = T.Compose([
        T.Resize((option_dict['resolution'], option_dict['resolution'])),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])

    img = custom_pil_loader(image_path)
    transformed = transforms(img)

    return transformed


def initialization(option_dict):
    model, device = select_model(option_dict['net'], option_dict['dataset'], option_dict['device'])
    model.load_state_dict(torch.load(option_dict['pt_path'], map_location=device)['model_state_dict'], strict=False)

    return model, device


def one_evaluate(model, data, device):
    label_tags = {
        0: 'Normal',
        1: 'Pneumonia',
    }
    model.eval()

    with torch.no_grad():
        output = model(data.unsqueeze(0).to(device))

        _, predicted = output.max(1)
        pred = label_tags[predicted.item()]

    return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Image Classification')
    parser.add_argument("--yml_path", required=True, default="./inference_setting.yml",
                        help="path to yml file contains options")
    parser.add_argument("--image_path", required=True)
    args = parser.parse_args()

    o_dict = yml_to_dict(args.yml_path)
    model, device = initialization(o_dict)
    transformed_img = transformed(o_dict, args.image_path)
    pred = one_evaluate(model, transformed_img, device)
    print('prediction result:', pred)