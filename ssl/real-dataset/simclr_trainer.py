import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from copy import deepcopy
from loss.nt_xent import NTXentLoss
import re
import os
import shutil
import sys

import numpy as np

import logging
log = logging.getLogger(__file__)

class SimCLRTrainer(object):
    def __init__(self, log_dir, model, optimizer, evaluator, device, params):
        self.model = model
        self.params = params
        self.device = device
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.writer = SummaryWriter(log_dir)
        self.params = params
        self.nt_xent_criterion = NTXentLoss(self.device, params['batch_size'], **params['nce_loss'])

    def _step(self, model, xis, xjs, xs, n_iter):

        # get the representations and the projections
        zis = model(xis)  # [N,C]

        # get the representations and the projections
        zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        if xs is not None:
            # Unaugmented datapoint. 
            zs = model(xs)
            zs = F.normalize(zs, dim=1)
        else:
            zs = None

        loss, loss_intra = self.nt_xent_criterion(zis, zjs, zs)
        return loss, loss_intra

    def train(self, train_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.params["batch_size"] * torch.cuda.device_count(),
                                  num_workers=self.params["num_workers"], drop_last=True, shuffle=False)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        if not os.path.exists(model_checkpoints_folder):
            os.mkdir(model_checkpoints_folder)

        self.save_model(os.path.join(model_checkpoints_folder, 'model_000.pth'))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)

        margin = self.params["grad_combination_margin"]
        if margin is not None:
            matcher = re.compile(r"encoder.(\d+)")
            layers = dict()
            for name, _ in self.model.named_parameters():
                # print(f"{name}: {params.size()}")
                m = matcher.match(name)
                if m is None:
                    l = 10
                else:
                    l = int(m.group(1))
                layers[name] = l
            unique_entries = sorted(list(set(layers.values())))
            series = np.linspace(margin, 1 - margin, len(unique_entries)) 
            l2ratio = dict(zip(unique_entries, series))
            layer2ratio = { name : l2ratio[l] for name, l in layers.items() }

            log.info(f"Gradient margin: {margin}")
            for name, r in layer2ratio.items():
                log.info(f"  {name}: {r}")
        else:
            log.info("No gradient margin")

        n_iter = 0
        alpha = self.params["noise_blend"]

        for epoch_counter in range(self.params['max_epochs']):
            loss_record = []
            suffix = str(epoch_counter).zfill(3)

            # Add noise to weight once in a while
            if alpha > 0:
                for name, p in self.model.named_parameters():
                    with torch.no_grad():
                        if len(p.size()) < 2:
                            continue
                        w = torch.zeros_like(p, device=p.get_device())
                        torch.nn.init.xavier_uniform_(w)
                        p[:] = (1 - alpha) * p[:] + alpha * w

            for (xis, xjs, xs), _ in train_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                if self.nt_xent_criterion.need_unaug_data():
                    xs = xs.to(self.device)
                else:
                    xs = None

                loss, loss_intra = self._step(self.model, xis, xjs, xs, n_iter)

                # if n_iter % self.params['log_every_n_steps'] == 0:
                #     self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                all_loss = loss + loss_intra
                loss_record.append(all_loss.item())

                if margin is not None:
                    # Here we do backward twice for each loss and weight the gradient at different layer differently. 
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)

                    inter_grads = dict()
                    for name, p in self.model.named_parameters():
                        # print(f"{name}: {p.size()}")
                        inter_grads[name] = p.grad.clone()

                    self.optimizer.zero_grad()
                    loss_intra.backward()
                    for name, p in self.model.named_parameters():
                        r = layer2ratio[name]
                        # Lower layer -> high ratio of loss_intra
                        p.grad *= (1 - r)
                        p.grad += inter_grads[name] * r
                else:
                    self.optimizer.zero_grad()
                    all_loss.backward()

                self.optimizer.step()
                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

            log.info(f"Epoch {epoch_counter}: numIter: {n_iter} Loss: {np.mean(loss_record)}")
            if self.evaluator is not None:
                best_acc = self.evaluator.eval_model(deepcopy(self.model))
                log.info(f"Epoch {epoch_counter}: best_acc: {best_acc}")

            if epoch_counter % self.params["save_per_epoch"] == 0:
                # save checkpoints
                self.save_model(os.path.join(model_checkpoints_folder, f'model_{suffix}.pth'))

    def save_model(self, PATH):
        torch.save({
            'online_network_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)
