import copy
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random

from torch.utils.data import DataLoader
from collections import defaultdict
from selftrainloss import SelfTrainingLoss

import torchmetrics
from torchmetrics.classification import MulticlassJaccardIndex

class Client:
    def __init__(self, client_id, dataset, model, num_epochs=10, lr=0.05, batch_size=4,
                 momentum=0.9, weight_decay=5e-5, device=None, pseudo_lab=False, teacher_model=None):
        self.id = client_id
        self.dataset = dataset
        self._model = model
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = device
        self.batch_size = batch_size
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.pseudo_lab = pseudo_lab

        self.dataset.set_client(self.id) #IMPORTANT
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, drop_last=True)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.metric = MulticlassJaccardIndex(num_classes=19, ignore_index=255).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        if self.pseudo_lab:
            self.criterion = SelfTrainingLoss()
            self.criterion.set_teacher(teacher_model)

    def run_epoch(self, net):
        loss_tot, miou, count = 0.0, 0.0, 0
        for cur_step, (images, labels) in enumerate(self.loader):
            images = images.half().to(self.device)
            labels = labels.half().squeeze().to(self.device, dtype=torch.long)

            self.optimizer.zero_grad()

            outputs = net(images)
            if self.pseudo_lab:
                loss = self.criterion(outputs, images)
            else:
                loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            loss_tot += loss
            miou += self.metric(outputs, labels)
            count += 1

        return miou.item()/count, loss_tot.item()/count

    def generate_update(self):
        return copy.deepcopy(self.model.state_dict())

    def train(self):
        self.dataset.set_client(self.id) #IMPORTANT
        num_train_samples = len(self.dataset)

        self.model.train()
        net = self.model.half()

        print(f'Client #{self.id+1}:', end=' ')
        for epoch in range(self.num_epochs):
            miou, loss = self.run_epoch(net)
            print(f'{miou:.3f}', end=' ')
        print('')

        update = self.generate_update()
        return num_train_samples, update

    def __str__(self):
        return self.id

    @property
    def num_samples(self):
        return len(self.dataset)

    def len_loader(self):
        return len(self.loader)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
