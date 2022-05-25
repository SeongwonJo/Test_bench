import yaml
import argparse
from PIL import Image

import torch
import torchvision.transforms as T

from setup import select_model


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


def initialization(option_dict):
    transforms = T.Compose([
        T.Resize((option_dict['resolution'], option_dict['resolution'])),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])

    # data_path == image path
    img = custom_pil_loader(option_dict['data_path'])
    transformed = transforms(img)

    model, device = select_model(option_dict['net'], option_dict['dataset'], option_dict['device'])
    model.load_state_dict(torch.load(option_dict['pt_path'], map_location=device)['model_state_dict'], strict=False)

    return transformed, model, device


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
    args = parser.parse_args()

    o_dict = yml_to_dict(args.yml_path)
    transformed_img, model, device = initialization(o_dict)
    pred = one_evaluate(model, transformed_img, device)
    print('prediction result:', pred)