import argparse
import copy
import random
import numpy
import torch
import os
from torch import optim
from localTrain import localTrain1,localTrain2,localTrain3,localTrain
from Fed import FedAvg
from Nets import task_classifier, GeneDistrNet, Discriminator, feature_extractor
from test import test1, test1_g
import numpy as np
import time
from sampling import  get_dataset
from utilss import *


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def args_parser():
    paser = argparse.ArgumentParser()
    paser.add_argument('--fft1', type=str, default=True, help="time samples or frequency samples")
    paser.add_argument('--class_num', type=int, default=7, help="number of classes")
    paser.add_argument('--dataset', type=str, default='PU', help='name of dataset')
    paser.add_argument('--batch-size', type=int, default=128, help='batch size for training')
    paser.add_argument('--workers', type=int, default=4, help='number of data-loading workers')
    paser.add_argument('--pin', type=bool, default=True, help='pin-memory')
    paser.add_argument('--lr0', type=float, default=0.05, help='learning rate 0')
    paser.add_argument('--lr1', type=float, default=0.07, help='learning rate 1')
    paser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    paser.add_argument('--weight-dec', type=float, default=1e-5, help='0.005weight decay coefficient default 1e-5')
    paser.add_argument('--rp-size', type=int, default=1024, help='Random Projection size 1024')

    paser.add_argument('--epochs__self', type=int, default=1, help='rounds of training')
    paser.add_argument('--epochs__cross', type=int, default=1, help='rounds of training')


    paser.add_argument('--current_epoch', type=int, default=1, help='current epoch in training')
    paser.add_argument('--factor', type=float, default=0.2, help='lr decreased factor (0.1)')
    paser.add_argument('--patience', type=int, default=20, help='number of epochs to want before reduce lr (20)')
    paser.add_argument('--lr-threshold', type=float, default=1e-4, help='lr schedular threshold')
    paser.add_argument('--ite-warmup', type=int, default=100, help='LR warm-up iterations (default:500)')
    paser.add_argument('--label_smoothing', type=float, default=0.1, help='the rate of wrong label(default:0.2)')
    paser.add_argument('--input_size', type=int, default=128, help='the size of hidden feature')
    paser.add_argument('--hidden_size', type=int, default=256, help='the size of hidden feature')
    paser.add_argument('--global_epochs', type=int, default=150, help='the num of global train epochs')
    paser.add_argument('--i_epochs', type=int, default=3, help='the num of independent epochs in local')
    paser.add_argument('--path_root', type=str, default='../data/PACS/', help='the root of dataset')
    args = paser.parse_args()
    return args
if __name__ == '__main__':
    args = args_parser()

    args.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
    numpy.random.seed(1)
    torch.manual_seed(1)
    random.seed(1)

    src_tar = np.array([[7,9,8,6],[6,8,9,7],[6,7,9,8],[6,7,8,9]])


    for taskindex in range(4):
        source1 = src_tar[taskindex][0]
        source2 = src_tar[taskindex][1]
        source3 = src_tar[taskindex][2]
        target = src_tar[taskindex][3]

        for repeat in range(5):

            Train_Loss_list = []
            Train_Accuracy_list = []
            Test_Loss_list = []
            Test_Accuracy_list = []


            start = time.time()

            src_name1 = 'load' + str(source1) + '_train'
            src_name2 = 'load' + str(source2) + '_train'
            src_name3 = 'load' + str(source3) + '_train'
            test_name = 'load' + str(target) + '_test'

            client = [src_name1, src_name2, src_name3, test_name]

            torch.cuda.empty_cache()

            train_loaders, target_loader = get_dataset(args, client)  ### Valid_loaders？？

            # initial the global model
            global_fetExtrac = feature_extractor(optim.SGD, args.lr0, args.weight_dec)

            # global_fetExtrac.load_state_dict(load_FCparas("alexnet"), strict=False)

            global_fetExtrac.optimizer = optim.Adam(global_fetExtrac.parameters(), args.lr0, weight_decay=args.weight_dec)

            global_classifier_1 = task_classifier(args.hidden_size, optim.Adam, args.lr0, args.weight_dec,
                                                class_num=args.class_num)
            global_classifier_2 = task_classifier(args.hidden_size, optim.Adam, args.lr0, args.weight_dec,
                                                  class_num=args.class_num)
            global_classifier_3 = task_classifier(args.hidden_size, optim.Adam, args.lr0, args.weight_dec,
                                                  class_num=args.class_num)



            #

            local_cc =  []
            for i in range(3):  # local discriminator

                global_c = task_classifier(args.hidden_size, optim.Adam, args.lr0, args.weight_dec,
                                                class_num=args.class_num)

                global_c.optimizer = optim.Adam(global_c.parameters(), args.lr1, weight_decay=args.weight_dec)

                local_cc.append(global_c)
            #
            # server execution phase
            models_global = []
            model_best_paras, best_acc, best_id = {}, 0., 0
            for t in range(args.global_epochs):
                print('global training epoch: %d ' % (t + 1))
                args.current_epoch = t + 1
                w_locals, avg_ac = [], 0.
                # client update

                if t<=10:
                    for i in range(3):
                        if i % 3 == 0:
                            print('source domain {}/3'.format(i + 1))
                            local_f = copy.deepcopy(global_fetExtrac)
                            local_c1 = copy.deepcopy(global_classifier_1)

                            trainer = localTrain(local_f, local_c1, train_loaders[i], args)

                            acc, w, wc = trainer.train()  ###这一步训练很慢
                            w_locals.append(w)

                            local_cc[i] = wc

                            avg_ac += acc

                        if i % 3 == 1:
                            print('source domain {}/3'.format(i + 1))
                            local_f = copy.deepcopy(global_fetExtrac)

                            local_c2 = copy.deepcopy(global_classifier_2)


                            trainer = localTrain(local_f,  local_c2,  train_loaders[i],
                                                  args)  ###这一步训练很慢
                            acc, w, wc = trainer.train()  ###这一步训练很慢

                            w_locals.append(w)

                            local_cc[i] = wc

                            avg_ac += acc

                        if i % 3 == 2:
                            print('source domain {}/3'.format(i + 1))
                            local_f = copy.deepcopy(global_fetExtrac)


                            local_c3 = copy.deepcopy(global_classifier_3)

                            trainer = localTrain(local_f,local_c3, train_loaders[i], args)

                            acc, w, wc = trainer.train()  ###这一步训练很慢

                            w_locals.append(w)

                            local_cc[i] = wc

                            avg_ac += acc

                    models_global.clear()

                    # aggregation
                    models_global = FedAvg(w_locals)

                    model_best_paras['F'] = models_global[0]

                    global_fetExtrac.load_state_dict(models_global[0])

                    global_classifier_1.load_state_dict(local_cc[0])
                    global_classifier_2.load_state_dict(local_cc[1])
                    global_classifier_3.load_state_dict(local_cc[2])

                    acc_target = 0.
                    acc_target += test1_g(global_fetExtrac, global_classifier_1, global_classifier_2,
                                          global_classifier_3, target_loader)

                    train_loss = 0
                    train_acc = 0
                    test_loss = 0

                    Train_Accuracy_list.append(train_acc)
                    Train_Loss_list.append(train_loss)

                    Test_Accuracy_list.append(acc_target)
                    Test_Loss_list.append(test_loss)


                if t>10:
                    for i in range(3):
                        if i % 3 == 0:
                            print('source domain {}/3'.format(i + 1))
                            local_f = copy.deepcopy(global_fetExtrac)
                            local_c1 = copy.deepcopy(global_classifier_1)
                            local_c2 = copy.deepcopy(global_classifier_2)
                            local_c3 = copy.deepcopy(global_classifier_3)

                            trainer = localTrain1(local_f, local_c1, local_c2, local_c3, train_loaders[i], args)

                            acc, w, wc = trainer.train()  ###这一步训练很慢
                            w_locals.append(w)

                            local_cc[i] = wc

                            avg_ac += acc

                        if i % 3 == 1:
                            print('source domain {}/3'.format(i + 1))
                            local_f = copy.deepcopy(global_fetExtrac)

                            local_c1 = copy.deepcopy(global_classifier_1)
                            local_c2 = copy.deepcopy(global_classifier_2)
                            local_c3 = copy.deepcopy(global_classifier_3)

                            trainer = localTrain2(local_f, local_c1, local_c2, local_c3, train_loaders[i],
                                                  args)  ###这一步训练很慢
                            acc, w, wc = trainer.train()  ###这一步训练很慢

                            w_locals.append(w)

                            local_cc[i] = wc

                            avg_ac += acc

                        if i % 3 == 2:
                            print('source domain {}/3'.format(i + 1))
                            local_f = copy.deepcopy(global_fetExtrac)

                            local_c1 = copy.deepcopy(global_classifier_1)
                            local_c2 = copy.deepcopy(global_classifier_2)
                            local_c3 = copy.deepcopy(global_classifier_3)

                            trainer = localTrain3(local_f, local_c1, local_c2, local_c3, train_loaders[i], args)

                            acc, w, wc = trainer.train()  ###这一步训练很慢

                            w_locals.append(w)

                            local_cc[i] = wc

                            avg_ac += acc

                    models_global.clear()

                    # aggregation
                    models_global = FedAvg(w_locals)

                    model_best_paras['F'] = models_global[0]

                    global_fetExtrac.load_state_dict(models_global[0])

                    global_classifier_1.load_state_dict(local_cc[0])
                    global_classifier_2.load_state_dict(local_cc[1])
                    global_classifier_3.load_state_dict(local_cc[2])

                    acc_target = 0.
                    acc_target += test1_g(global_fetExtrac, global_classifier_1, global_classifier_2,
                                          global_classifier_3, target_loader)

                    train_loss = 0
                    train_acc = 0
                    test_loss = 0

                    Train_Accuracy_list.append(train_acc)
                    Train_Loss_list.append(train_loss)

                    Test_Accuracy_list.append(acc_target)
                    Test_Loss_list.append(test_loss)


