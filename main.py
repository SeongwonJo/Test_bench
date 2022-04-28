import yaml
import argparse
import copy

from utils.alarm import send_alarm_to_slack
from utils.yml_to_tasklist import yml_to_tasklist
from trainer import Training


def sequential_task(tasklist, device, csv_path):
    for i in tasklist:
        trainer = Training(tasklist[i], index_num=i, device=device, csv_path=csv_path)
        trainer.apply_option()
        trainer.operation()
        trainer.finish()
    send_alarm_to_slack("All task done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Image Classification')
    parser.add_argument("--yml_path", required=True, default="./properties.yml",
                        help="path to yml file contains experiment options")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--csv_path", default='/home/work/test1/result.csv')
    args = parser.parse_args()

    tasklist = yml_to_tasklist(args.yml_path)
    sequential_task(tasklist, args.device, args.csv_path)

