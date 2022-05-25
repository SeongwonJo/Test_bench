import argparse

from utils.alarm import send_alarm_to_slack
from utils.yml_to_tasklist import yml_to_tasklist
from trainer import Training, evaluate
import inference


def argument():
    parser = argparse.ArgumentParser(description='PyTorch Image Classification')
    parser.add_argument("--yml_path", required=True, default="./properties.yml",
                        help="path to yml file contains experiment options")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--csv_path", default='/home/work/test1/result.csv')
    parser.add_argument("--task", default="train")
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
    o_dict = inference.yml_to_dict(yml_path)
    test_loader, model, device = inference.initialization(o_dict)
    test_accuracy = evaluate(model, test_loader, device)
    print('Test Accuracy: {:.2f}%'.format(test_accuracy))


def select_task(task):
    if task == 'train':
        tasklist = yml_to_tasklist(args.yml_path)
        sequential_task(tasklist, args.device, args.csv_path)
    elif task == 'inference':
        inference_task(args.yml_path)


if __name__ == '__main__':
    args = argument()
    select_task(args.task)



