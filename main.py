import argparse

from utils.alarm import send_alarm_to_slack
from utils.yml_to_tasklist import yml_to_tasklist
from trainer import Training


def argument():
    parser = argparse.ArgumentParser(description='PyTorch Image Classification')
    parser.add_argument('-y', "--yml_path", default="./properties.yml",
                        help="path to yml file contains experiment options")
    parser.add_argument('-d', "--device", default="cuda:0")
    parser.add_argument('-c', "--csv_path", default='./result.csv')
    parser.add_argument('-t', "--task", default="train")
    arguments = parser.parse_args()
    return arguments


def sequential_task(tasklist, device, csv_path):
    for i in tasklist:
        trainer = Training(tasklist[i], index_num=i, device=device, csv_path=csv_path)
        trainer.apply_option()
        trainer.operation()
        trainer.finish()
    send_alarm_to_slack("All task done")


def inference_task(yml_path):
    print('will be implemented')


def select_task(arguments):
    if arguments.task == 'train':
        tasklist = yml_to_tasklist(arguments.yml_path)
        sequential_task(tasklist, arguments.device, arguments.csv_path)
    elif arguments.task == 'inference':
        inference_task(arguments.yml_path)


if __name__ == '__main__':
    args = argument()
    select_task(args)



