import torch
import torch.nn as nn

from torch import optim
from torchvision import transforms, datasets
from torchvision.models import resnet,vgg, inception
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from models import resnet_cifar10, inceptionv3_1ch, xception

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import numpy as np
from PIL import Image


# augmentation
# only ToTensor (mnist)
common_transform = transforms.Compose([transforms.ToTensor()])

# CIFAR-10, 100
cifar10_transform = A.Compose(
    [
        A.PadIfNeeded(min_height=40, min_width=40),
        A.RandomCrop(height=32, width=32),
        A.HorizontalFlip(),
        A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
        ToTensorV2(),
    ]
)

cifar100_transform = A.Compose(
    [
        A.PadIfNeeded(min_height=40, min_width=40),
        A.RandomCrop(height=32, width=32),
        A.HorizontalFlip(),
        A.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
        ToTensorV2(),
    ]
)

# ImageNet
sample_list = list(range(256, 481))
imagenet_transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=sample_list),
        A.HorizontalFlip(),
        A.RandomCrop(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
# custom
custom_transform = A.Compose(
    [
        A.Resize(height=256, width=256),
        A.Rotate(10),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ]
)

# for inception
custom_transform_i = A.Compose(
    [
        A.Resize(height=299, width=299),
        A.Rotate(10),
        ToTensorV2(),
    ]
)

# same custom_transform (with torch transforms)
# transform_train = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.RandomRotation(10),
#     transforms.ToTensor(),
# ])
# transform_test = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
# ])


# testdata setting
cifar10_test = A.Compose(
    [
        A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
        ToTensorV2(),
    ]
)

cifar100_test = A.Compose(
    [
        A.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
        ToTensorV2(),
    ]
)

imagenet_test = A.Compose(
    [
        A.Resize(height=256, width=256),
        A.CenterCrop(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

custom_test = A.Compose(
    [
        A.Resize(height=256, width=256),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ]
)

custom_test_i = A.Compose(
    [
        A.Resize(height=299, width=299),
        ToTensorV2(),
    ]
)


# apply albumentations on torch dataloader
class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))['image']


class Cifar10SearchDataset(datasets.CIFAR10):
    def __init__(self, root,
                 train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


class Cifar100SearchDataset(datasets.CIFAR100):
    def __init__(self, root,
                 train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


class ImageNetSearchDataset(datasets.ImageNet):
    def __init__(self, root,
                 split='train', transform=None):
        super().__init__(root=root, split=split, transform=transform)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


# load image on dataloader with grayscale
def custom_pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img.load()
        return img.convert('L')


# train_validation split sampler
def train_val_split(dataset):
    dataset_size =len(dataset)  # 전체크기
    indices = list(range(dataset_size)) # 전체 인덱스 리스트만들고
    split = int(np.floor(0.2*dataset_size)) # 내림함수로 20% 지점 인덱스
    np.random.seed(42)
    np.random.shuffle(indices) # 인덱스 리스트 섞어줌

    # 섞어진 리스트에서 처음부터 ~번째 까지 val, ~+1번째부터 끝 인덱스까지 train
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    return train_sampler, val_sampler


# dataset select
def load_dataset(*,data_src='/home/work/test1/data', batch_size, dataset) :
    # train_loader = None
    # test_loader = None
    if dataset == 'mnist':
        train_dataset = datasets.MNIST(root=data_src,train=True, transform=common_transform, download=True)
        # val_dataset = datasets.MNIST(root=data_src,train=True, transform=common_transform, download=True)
        train_sampler, val_sampler = train_val_split(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, sampler=val_sampler, pin_memory=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(
            dataset=datasets.MNIST(root=data_src, train=False, transform=common_transform, download=True),
            batch_size=batch_size)
    elif dataset == 'cifar10':
        train_dataset = Cifar10SearchDataset(root=data_src, train=True, transform=cifar10_transform)
        val_dataset = Cifar10SearchDataset(root=data_src, train=True, transform=cifar10_test)
        train_sampler, val_sampler = train_val_split(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=3)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=batch_size, sampler=val_sampler, pin_memory=True, num_workers=3)
        test_loader = torch.utils.data.DataLoader(
            dataset=Cifar10SearchDataset(root=data_src, train=False,transform=cifar10_test),
            batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3)
    elif dataset == 'cifar100':
        train_dataset = Cifar100SearchDataset(root=data_src, train=True, transform=cifar100_transform)
        val_dataset = Cifar100SearchDataset(root=data_src, train=True, transform=cifar100_test)
        train_sampler, val_sampler = train_val_split(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=3)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=batch_size, sampler=val_sampler, pin_memory=True, num_workers=3)
        test_loader = torch.utils.data.DataLoader(
            dataset=Cifar100SearchDataset(root=data_src, train=False, transform=cifar100_test),
            batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3)
    elif dataset == 'imagenet':
        train_dataset = ImageNetSearchDataset(root=data_src+'/imagenet', split='train',transform=imagenet_transform)
        val_dataset = ImageNetSearchDataset(root=data_src+'/imagenet', split='train',transform=imagenet_test)
        train_sampler, val_sampler = train_val_split(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=3)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=batch_size, sampler=val_sampler, pin_memory=True, num_workers=3)
        test_loader = torch.utils.data.DataLoader(
            dataset=ImageNetSearchDataset(root=data_src+'/imagenet', split='val',transform=imagenet_test),
            batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3)
    elif dataset == 'custom':
        train_dataset = datasets.ImageFolder(root=data_src+'/chest_xray/train',
                                             transform=Transforms(custom_transform), loader=custom_pil_loader)
        val_dataset = datasets.ImageFolder(root=data_src+'/chest_xray/train',
                                           transform=Transforms(custom_test), loader=custom_pil_loader)
        test_dataset = datasets.ImageFolder(root=data_src+'/chest_xray/test',
                                         transform=Transforms(custom_test), loader=custom_pil_loader)
        train_sampler, val_sampler = train_val_split(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=3)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=batch_size, sampler=val_sampler, pin_memory=True, num_workers=3)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3)
    elif dataset == 'custom3ch':
        train_dataset = datasets.ImageFolder(root=data_src+'/chest_xray/train',
                                             transform=Transforms(custom_transform))                                  
        val_dataset = datasets.ImageFolder(root=data_src+'/chest_xray/train',
                                           transform=Transforms(custom_test), loader=custom_pil_loader)
        test_dataset = datasets.ImageFolder(root=data_src+'/chest_xray/test',
                                         transform=Transforms(custom_test))
        train_sampler, val_sampler = train_val_split(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=3)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=batch_size, sampler=val_sampler, pin_memory=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3)
    elif dataset == 'custom3ch_i':
        train_dataset = datasets.ImageFolder(root=data_src+'/chest_xray/train',
                                             transform=Transforms(custom_transform_i))                                  
        val_dataset = datasets.ImageFolder(root=data_src+'/chest_xray/train',
                                           transform=Transforms(custom_test_i))
        test_dataset = datasets.ImageFolder(root=data_src+'/chest_xray/test',
                                         transform=Transforms(custom_test_i))
        train_sampler, val_sampler = train_val_split(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=3)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=batch_size, sampler=val_sampler, pin_memory=True, num_workers=3)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3)
    elif dataset == 'custom_i':
        train_dataset = datasets.ImageFolder(root=data_src+'/chest_xray/train',
                                             transform=Transforms(custom_transform_i), loader=custom_pil_loader)                                  
        val_dataset = datasets.ImageFolder(root=data_src+'/chest_xray/train',
                                           transform=Transforms(custom_test_i), loader=custom_pil_loader)
        test_dataset = datasets.ImageFolder(root=data_src+'/chest_xray/test',
                                         transform=Transforms(custom_test_i), loader=custom_pil_loader)
        train_sampler, val_sampler = train_val_split(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=3)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=batch_size, sampler=val_sampler, pin_memory=True, num_workers=3)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3)
    elif dataset == 'custom_3class':
        train_dataset = datasets.ImageFolder(root=data_src + '/chest_xray3/train',
                                             transform=Transforms(custom_transform), loader=custom_pil_loader)
        val_dataset = datasets.ImageFolder(root=data_src + '/chest_xray3/train',
                                           transform=Transforms(custom_test), loader=custom_pil_loader)
        test_dataset = datasets.ImageFolder(root=data_src + '/chest_xray3/test',
                                            transform=Transforms(custom_test), loader=custom_pil_loader)
        train_sampler, val_sampler = train_val_split(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=3)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=batch_size, sampler=val_sampler, pin_memory=True, num_workers=3)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3)
    else:
        raise ValueError('please check your "dataset" input.')

    return train_loader, val_loader, test_loader


# resnet200
def resnet200(pretrained: bool = False, progress: bool = True, **kwargs) -> resnet.ResNet:
    return resnet._resnet('resnet200', resnet.Bottleneck, [3, 24, 36, 3], pretrained, progress,
                   **kwargs)


# select model
def select_model(net, dataset, device_num):
    num_classes = {
        'cifar10': 10,
        'cifar100': 100,
        'custom': 2,
        'custom_i': 2,
        'custom_3class': 3,
        'custom3ch': 2,
        'custom3ch_i': 2,
        'imagenet': 1000,
        'mnist': 10
    }.get(dataset, "error")
    
    if dataset == 'cifar10' or dataset == 'cifar100':
        model = {
            'resnet20': resnet_cifar10.resnet20(num_classes=num_classes),
            'resnet32': resnet_cifar10.resnet32(num_classes=num_classes),
            'resnet44': resnet_cifar10.resnet44(num_classes=num_classes),
            'resnet56': resnet_cifar10.resnet56(num_classes=num_classes),
            'resnet110': resnet_cifar10.resnet110(num_classes=num_classes),
            'resnet152': resnet_cifar10.resnet152(num_classes=num_classes),
            'resnet200': resnet_cifar10.resnet200(num_classes=num_classes),
        }.get(net, "error")
    else:
        model = {
            'resnet18': resnet.resnet18(num_classes=num_classes),
            'resnet50': resnet.resnet50(num_classes=num_classes),
            'resnet101': resnet.resnet101(num_classes=num_classes),
            'resnet152': resnet.resnet152(num_classes=num_classes),
            'resnet200': resnet200(num_classes=num_classes),
            'vgg16': vgg.vgg16(num_classes=num_classes),
            'inception': inception.inception_v3(num_classes=num_classes, init_weights=True),
            'inception_1ch': inceptionv3_1ch.inception_v3(num_classes=num_classes, init_weights=True),
            'xception' : xception.xception(input_channel=1, num_classes=num_classes)
        }.get(net, "error")

    if model == "error":
        raise ValueError('please check your "net" input.')

    if dataset == 'mnist':
        model.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    elif dataset == 'custom' or dataset == 'custom2':
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    device = torch.device(device_num if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("device:",device)

    # parallel processing (under construction)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model).cuda()

    return model, device


class MyOptimizer:
    def __init__(self, net, lr, momentum):
        self.net = net
        self.lr = lr
        self.momentum = momentum

    def SGD(self):
        return MySGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0.0005)

    def Adam(self):
        return optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=0.001)  # default lr = 0.001

    def Nesterov(self):
        return optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0.0005, nesterov=True)

    def RMSprop(self):
        return optim.RMSprop(self.net.parameters(), lr=self.lr)  # default lr = 0.01

    def select_optimizer(self, opt):
        switch_case = {
            "SGD": self.SGD(),
            "Adam": self.Adam(),
            "NAG": self.Nesterov(),
            "RMSprop": self.RMSprop(),
        }.get(opt, "error")

        if switch_case == "error":
            raise ValueError('please check your "opt" input.')
        return switch_case


class MySGD(optim.Optimizer):
    def __init__(self, params, lr=0.1, momentum=0.0, weight_decay=0.0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        super(MySGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MySGD, self).__setstate__(state)

    @torch.no_grad()
    def step(self):
        loss = None

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = [] # grad 값들
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            # dampening = group['dampening']
            # nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            #sgd
            for i, param in enumerate(params_with_grad):
                d_p = d_p_list[i]
                if weight_decay != 0:
                    d_p = d_p.add(param, alpha=weight_decay)

                if momentum != 0:
                    buf = momentum_buffer_list[i]

                    if buf is None:
                        buf = torch.clone(d_p).detach() # 복사하고 gradient가 없는 tensor로 만듬
                        momentum_buffer_list[i] = buf # 초기값, buf가 식에서 v 인듯
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1) # \mu v + g

                    d_p = buf
                    # if nesterov:
                    #     d_p = d_p.add(buf, alpha=momentum) # g + \mu v

                param.add_(d_p, alpha=-lr) # w - lr * g
                # Dataloader 에서 batch_size, shuffle 정할 수 있으므로
                # batch_size 때문에 w = w - lr * sum_i=1^batchsize g_i 라고할수있다.(minibatch SGD)
                # shuffle 이 random sampling 을 맡는다고 생각하면 될듯

            # update momentum_buffers
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


def set_lr_scheduler(optimizer, epochs, last_ep):
    if last_ep == 0:
        last_ep = -1
    decay_step = int(epochs / 10)
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
    #                                               milestones=[60, 120, 160],
    #                                               gamma=0.2, last_epoch=last_ep)
    # lr_scheduler = LambdaScheduler(optimizer, lr_lambda=lambda epoch: 1 / 2 ** (epoch // decay_step),
    #                        momentum_lambda=lambda epoch: 1 if (epoch // decay_step) == 0 else 9
    #                        if (epoch // decay_step) > 8 else 1 * (epoch // decay_step + 1))

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=decay_step, gamma=1, last_epoch=last_ep)

    # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=1, eta_min=0)
    return lr_scheduler
