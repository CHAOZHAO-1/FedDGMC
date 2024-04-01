import sys
import torch
import numpy as np
from torch import optim

from utilss import LabelSmoothingLoss, GradualWarmupScheduler


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

a=1
b=5

class localTrain1(object):
    def __init__(self, fetExtrac, classifier1,classifier2,classifier3, train_loader, args):
        self.args = args
        self.fetExtrac = fetExtrac.cuda()

        self.classifier1 = classifier1.cuda()
        self.classifier2 = classifier2.cuda()
        self.classifier3 = classifier3.cuda()

        self.train_loader = train_loader

        self.lossFunc = LabelSmoothingLoss(args.label_smoothing, lbl_set_size=7).cuda()
        self.opti_task = optim.Adam(list(self.fetExtrac.parameters()) + list(self.classifier1.parameters()), args.lr0, weight_decay=args.weight_dec)

        afsche_task = optim.lr_scheduler.ReduceLROnPlateau(self.opti_task, factor=args.factor, patience=args.patience,
                                                          threshold=args.lr_threshold, min_lr=1e-7)
        self.sche_task = GradualWarmupScheduler(self.opti_task, total_epoch=args.ite_warmup,
                                               after_scheduler=afsche_task)


    def train(self):

        ac = [0.]
        best_model_dict = {}


        # local training (E1)
        for i in range(self.args.epochs__self):



            print('\r{}/{}'.format(i + 1, self.args.epochs__self), end='')
            for t, batch in enumerate(self.train_loader):
                self.train_step(batch) ##7s



            self.sche_task.step(i+self.args.i_epochs+1, 1.-ac[-1])
        #

            best_model_dict['F'] = self.fetExtrac.state_dict()
            best_model_dict['C'] = self.classifier1.state_dict()


        loc_w = [best_model_dict['F'], best_model_dict['F']]
        #
        return np.max(ac), loc_w, best_model_dict['C']

    def train_step(self, batch):



        x, y = batch
        x = x.cuda()
        y = y.cuda()


        self.fetExtrac.train()

        self.classifier1.train()


        # training feature extractor and classifier

        self.opti_task.zero_grad()
        fakez = self.fetExtrac(x)
        pre1 = self.classifier1(fakez)
        pre2 = self.classifier2(fakez)
        pre3 = self.classifier3(fakez)

        loss_cla = self.lossFunc(pre1, y)*a+self.lossFunc(pre2, y)*b+self.lossFunc(pre3, y)*b



        pre_2 = torch.max(pre2, 1)[1].data.squeeze()
        num_correct_2 = pre_2.eq(y.data.view_as(pre_2)).cpu().sum().numpy()

        pre_3 = torch.max(pre3, 1)[1].data.squeeze()
        num_correct_3 = pre_3.eq(y.data.view_as(pre_3)).cpu().sum().numpy()





        loss_cla = loss_cla
        loss_cla.backward()
        self.opti_task.step()

        return loss_cla.item()


class localTrain2(object):
    def __init__(self, fetExtrac, classifier1, classifier2, classifier3, train_loader, args):
        self.args = args
        self.fetExtrac = fetExtrac.cuda()

        self.classifier1 = classifier1.cuda()
        self.classifier2 = classifier2.cuda()
        self.classifier3 = classifier3.cuda()

        self.train_loader = train_loader

        self.lossFunc = LabelSmoothingLoss(args.label_smoothing, lbl_set_size=7).cuda()
        self.opti_task = optim.Adam(list(self.fetExtrac.parameters()) + list(self.classifier2.parameters()), args.lr0,
                                    weight_decay=args.weight_dec)

        afsche_task = optim.lr_scheduler.ReduceLROnPlateau(self.opti_task, factor=args.factor, patience=args.patience,
                                                           threshold=args.lr_threshold, min_lr=1e-7)
        self.sche_task = GradualWarmupScheduler(self.opti_task, total_epoch=args.ite_warmup,
                                                after_scheduler=afsche_task)

    def train(self):

        ac = [0.]
        best_model_dict = {}

        # local training (E1)
        for i in range(self.args.epochs__self):

            print('\r{}/{}'.format(i + 1, self.args.epochs__self), end='')
            for t, batch in enumerate(self.train_loader):
                self.train_step(batch)  ##7s

            self.sche_task.step(i + self.args.i_epochs + 1, 1. - ac[-1])
            #

            best_model_dict['F'] = self.fetExtrac.state_dict()
            best_model_dict['C'] = self.classifier2.state_dict()

        loc_w = [best_model_dict['F'], best_model_dict['F']]
        #
        return np.max(ac), loc_w, best_model_dict['C']

    def train_step(self, batch):

        x, y = batch
        x = x.cuda()
        y = y.cuda()

        self.fetExtrac.train()

        self.classifier2.train()

        # training feature extractor and classifier

        self.opti_task.zero_grad()
        fakez = self.fetExtrac(x)
        pre1 = self.classifier1(fakez)
        pre2 = self.classifier2(fakez)
        pre3 = self.classifier3(fakez)

        loss_cla = self.lossFunc(pre1, y)*b + self.lossFunc(pre2, y)*a + self.lossFunc(pre3, y)*b

        loss_cla = loss_cla
        loss_cla.backward()
        self.opti_task.step()

        return loss_cla.item()


class localTrain3(object):
    def __init__(self, fetExtrac, classifier1, classifier2, classifier3, train_loader, args):
        self.args = args
        self.fetExtrac = fetExtrac.cuda()

        self.classifier1 = classifier1.cuda()
        self.classifier2 = classifier2.cuda()
        self.classifier3 = classifier3.cuda()

        self.train_loader = train_loader

        self.lossFunc = LabelSmoothingLoss(args.label_smoothing, lbl_set_size=7).cuda()
        self.opti_task = optim.Adam(list(self.fetExtrac.parameters()) + list(self.classifier3.parameters()), args.lr0,
                                    weight_decay=args.weight_dec)

        afsche_task = optim.lr_scheduler.ReduceLROnPlateau(self.opti_task, factor=args.factor, patience=args.patience,
                                                           threshold=args.lr_threshold, min_lr=1e-7)
        self.sche_task = GradualWarmupScheduler(self.opti_task, total_epoch=args.ite_warmup,
                                                after_scheduler=afsche_task)

    def train(self):

        ac = [0.]
        best_model_dict = {}

        # local training (E1)
        for i in range(self.args.epochs__self):

            print('\r{}/{}'.format(i + 1, self.args.epochs__self), end='')
            for t, batch in enumerate(self.train_loader):
                self.train_step(batch)  ##7s

            self.sche_task.step(i + self.args.i_epochs + 1, 1. - ac[-1])
            #

            best_model_dict['F'] = self.fetExtrac.state_dict()
            best_model_dict['C'] = self.classifier3.state_dict()

        loc_w = [best_model_dict['F'], best_model_dict['F']]
        #
        return np.max(ac), loc_w, best_model_dict['C']

    def train_step(self, batch):

        x, y = batch
        x = x.cuda()
        y = y.cuda()

        self.fetExtrac.train()

        self.classifier3.train()

        # training feature extractor and classifier

        self.opti_task.zero_grad()
        fakez = self.fetExtrac(x)
        pre1 = self.classifier1(fakez)
        pre2 = self.classifier2(fakez)
        pre3 = self.classifier3(fakez)

        loss_cla = self.lossFunc(pre1, y)*b + self.lossFunc(pre2, y)*b + self.lossFunc(pre3, y)*a

        loss_cla = loss_cla
        loss_cla.backward()
        self.opti_task.step()

        return loss_cla.item()


class localTrain(object):
    def __init__(self, fetExtrac, classifier,train_loader, args):
        self.args = args
        self.fetExtrac = fetExtrac.cuda()

        self.classifier = classifier.cuda()


        self.train_loader = train_loader

        self.lossFunc = LabelSmoothingLoss(args.label_smoothing, lbl_set_size=7).cuda()
        self.opti_task = optim.Adam(list(self.fetExtrac.parameters()) + list(self.classifier.parameters()), args.lr0,
                                    weight_decay=args.weight_dec)

        afsche_task = optim.lr_scheduler.ReduceLROnPlateau(self.opti_task, factor=args.factor, patience=args.patience,
                                                           threshold=args.lr_threshold, min_lr=1e-7)
        self.sche_task = GradualWarmupScheduler(self.opti_task, total_epoch=args.ite_warmup,
                                                after_scheduler=afsche_task)

    def train(self):

        ac = [0.]
        best_model_dict = {}

        # local training (E1)
        for i in range(self.args.epochs__self):

            print('\r{}/{}'.format(i + 1, self.args.epochs__self), end='')
            for t, batch in enumerate(self.train_loader):
                self.train_step(batch)  ##7s

            self.sche_task.step(i + self.args.i_epochs + 1, 1. - ac[-1])
            #

            best_model_dict['F'] = self.fetExtrac.state_dict()
            best_model_dict['C'] = self.classifier.state_dict()

        loc_w = [best_model_dict['F'], best_model_dict['F']]
        #
        return np.max(ac), loc_w, best_model_dict['C']

    def train_step(self, batch):

        x, y = batch
        x = x.cuda()
        y = y.cuda()

        self.fetExtrac.train()

        self.classifier.train()

        # training feature extractor and classifier
        self.fetExtrac.train()
        self.opti_task.zero_grad()
        fakez = self.fetExtrac(x)
        pre= self.classifier(fakez)


        loss_cla = self.lossFunc(pre, y)

        loss_cla = loss_cla
        loss_cla.backward()
        self.opti_task.step()

        return loss_cla.item()