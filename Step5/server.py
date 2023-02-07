import numpy as np
import torch
import copy
import torch.optim as optim
import matplotlib.pyplot as plt

from copy import deepcopy
from collections import OrderedDict

import torchmetrics
from torchmetrics.classification import MulticlassJaccardIndex

class Server:
    def __init__(self, model, pseudo_lab=False, T=0, model_path=None):
        self.model = copy.deepcopy(model).half()
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.selected_clients = []
        self.updates = []
        self.optimizer = optim.SGD(params=self.model.parameters(), lr=1, momentum=0.9)
        self.total_grad = 0 
        self.round = 0
        self.pseudo_lab = pseudo_lab
        self.T = T
        self.model_path = model_path

        self.history = {'val_miou': [0.0]} 

    def select_clients(self, my_round, possible_clients, num_clients=10):
        self.round = my_round
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

    def add_updates(self, num_samples, update):
        self.updates.append((num_samples, update))

    def _compute_client_delta(self, cmodel):
        delta = OrderedDict.fromkeys(cmodel.keys())
        for k, x, y in zip(self.model_params_dict.keys(), self.model_params_dict.values(), cmodel.values()):
            delta[k] = y - x if "running" not in k and "num_batches_tracked" not in k else y
        return delta

    def load_server_model_on_client(self, client):
        client.model.load_state_dict(self.model_params_dict)

    def train_model(self):
        self.optimizer.zero_grad()
        clients = self.selected_clients

        for i, c in enumerate(clients):
            if self.pseudo_lab and self.T!=0 and self.round%self.T==0:
                c.criterion.set_teacher(deepcopy(self.model).half())
            self.load_server_model_on_client(c)
            num_samples, update = c.train()

            update = self._compute_client_delta(update)
            self.add_updates(num_samples=num_samples, update=update)

    def _server_opt(self, pseudo_gradient):

        for n, p in self.model.named_parameters():
            p.grad = -1.0 * pseudo_gradient[n]

        self.optimizer.step()

        bn_layers = OrderedDict(
            {k: v for k, v in pseudo_gradient.items() if "running" in k or "num_batches_tracked" in k})
        self.model.load_state_dict(bn_layers, strict=False)

    def _aggregation(self):
        total_weight = 0.
        base = OrderedDict()

        for (client_samples, client_model) in self.updates:
            total_weight += client_samples
            for key, value in client_model.items():
                if key in base:
                    base[key] += client_samples * value
                else:
                    base[key] = client_samples * value
        averaged_sol_n = copy.deepcopy(self.model_params_dict)
        for key, value in base.items():
            if total_weight != 0:
                averaged_sol_n[key] = value.to('cuda') / total_weight

        return averaged_sol_n

    def _get_model_total_grad(self):
        total_norm = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_grad = total_norm ** 0.5
        return total_grad

    def update_model(self):
        """FedAvg on the clients' updates for the current round.
        Weighted average of self.updates, where the weight is given by the number
        of samples seen by the corresponding client at training time.
        Saves the new central model in self.client_model and its state dictionary in self.model
        """
        averaged_sol_n = self._aggregation()

        self._server_opt(averaged_sol_n)
        self.total_grad = self._get_model_total_grad()
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.updates = []
    
    def test_model(self, test_dataloader, device='cuda'):
        self.model.eval()
        net = self.model.half()
        metric = MulticlassJaccardIndex(num_classes=19, ignore_index=255).to(device)

        miou = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_dataloader):
                images = images.half().to(device)
                labels = labels.half().squeeze().to(device, dtype=torch.long)

                outputs = net(images, test=True)
                miou += metric(outputs, labels)
        miou = miou.item()/len(test_dataloader)

        if miou > max(self.history['val_miou']): #If this is the best validation mIoU, save the model
            print(f'** Saving model with mIoU = {miou}')
            self.history['val_miou'].append(miou)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optim_state_dict': self.optimizer.state_dict(),
                'round': self.round,
                'history': self.history,
        }, self.model_path)
        self.history['val_miou'].append(miou)
        return miou
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optim_state_dict'])
        if 'round' in checkpoint.keys():
            self.round = checkpoint['round'] + 1
        if 'history' in checkpoint.keys():
            self.history = checkpoint['history']
        mIoU = self.history['val_miou'][-1]
        print(f'Loading pre-trained model, mIoU={mIoU}')
        return self.round, self.history
    
    def plot_history(self):
        plt.plot(self.history['val_miou'][1:], label='Validation mIoU')
        plt.legend()
