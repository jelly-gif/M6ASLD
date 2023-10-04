import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from datetime import datetime
from models.handler import train, test
import argparse
from data_loader.SiteBinding_dataloader import *
import numpy as np
import pandas as pd
# from .models.Utils import *
from models.Utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--dataset', type=str, default='m6A_data')
parser.add_argument('--window_size', type=int, default=1)
parser.add_argument('--horizon', type=int, default=0)
parser.add_argument('--train_length', type=float, default=7)
parser.add_argument('--valid_length', type=float, default=2)
parser.add_argument('--test_length', type=float, default=1)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--multi_layer', type=int, default=5)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=145)
parser.add_argument('--norm_method', type=str, default='z_score')
parser.add_argument('--optimizer', type=str, default='RMSProp')
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--exponential_decay_step', type=int, default=20)
parser.add_argument('--decay_rate', type=float, default=0.1) #0.5
parser.add_argument('--dropout_rate', type=float, default=0.8) #0.5
parser.add_argument('--leakyrelu_rate', type=int, default=0.5) #0.2

parser.add_argument('--target', type=bool, default=False)
parser.add_argument('--path', type=str, default=False)



args = parser.parse_args()
print(f'Training configs: {args}')

result_train_file = os.path.join('output', args.dataset, 'M41')
result_test_file = os.path.join('output', args.dataset, 'test')
if not os.path.exists(result_train_file):
    os.makedirs(result_train_file)
if not os.path.exists(result_test_file):
    os.makedirs(result_test_file)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if args.train: #训练加验证
        try:
            before_train = datetime.now().timestamp()
            i=0
            all_result=[]
            while i<10:
                print('fold '+str(i)+' ')
                print('-'*99)
                train_data = []
                valid_data = []

                ReadMyCsv(train_data,args.path)
                ReadMyCsv(valid_data,args.path)
                print('Train begining!')
                forecast_feature,result=train(train_data, valid_data, args, result_train_file,i)
                all_result.append(result)

                i+=1

            after_train = datetime.now().timestamp()
            print(f'Training took {(after_train - before_train) / 60} minutes')

        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')

    print('done')




