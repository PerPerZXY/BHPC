import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from loss_functions.nt_xent import NTXentLoss
import os
import shutil
import sys
import pickle
import torch.optim as optim

from datasets.two_dim.NumpyDataLoader import NumpyDataSet
from networks.unet_con import GlobalConUnet, MLP

apex_support = False

import numpy as np

torch.manual_seed(0)

os.environ['CUDA_VISIBLE_DEVICE'] ='0,1'
def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))
#define the prediction head
class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        hidden_dim = in_dim
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class SimCLR(object):

    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        print("use the cpu:",self.device)

        self.writer = SummaryWriter(os.path.join(self.config['save_dir'], 'tensorboard'))
        self.nt_xent_criterion = NTXentLoss(self.device, **config['loss'])

        split_dir = os.path.join(self.config["base_dir"], "splits.pkl")
        data_dir = os.path.join(self.config["base_dir"], "preprocessed")
        print(data_dir)
        with open(split_dir, "rb") as f:
            splits = pickle.load(f)
        tr_keys = splits[0]['train'] + splits[0]['val'] + splits[0]['test']
        val_keys = splits[0]['val']
        self.train_loader = NumpyDataSet(data_dir, target_size=self.config["img_size"], batch_size=self.config["batch_size"],
                                         keys=tr_keys, do_reshuffle=True, mode='simclr')
        self.val_loader = NumpyDataSet(data_dir, target_size=self.config["img_size"], batch_size=self.config["val_batch_size"],
                                         keys=val_keys, do_reshuffle=True, mode='simclr')

        print(len(self.train_loader))
        self.model = GlobalConUnet()
        self.head = MLP(num_class=256)

        self.nt_xent_criterion = NTXentLoss(self.device, **config['loss'])

        # dist.init_process_group(backend='nccl')
        if torch.cuda.device_count() > 1:
            print("Let's use %d GPUs" % torch.cuda.device_count())
            # print("Let's use %d GPUs" % self.device)
            self.model = nn.DataParallel(self.model)
            self.head = nn.DataParallel(self.head)

        self.model.to(self.device)
        self.head.to(self.device)

        self.model = self._load_pre_trained_weights(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), 3e-4, weight_decay=eval(self.config['weight_decay']))
        # self.optimizer = torch.optim.SGD(self.model.parameters(), 1e-4,momentum=0.9, weight_decay=eval(self.config['weight_decay']))

    def set_optimizer(opt, model):
        if opt.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(),
                                  lr=opt.learning_rate,
                                  momentum=opt.momentum,
                                  weight_decay=opt.weight_decay
                                  )
        elif opt.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(),
                                   lr=opt.learning_rate,
                                   weight_decay=opt.weight_decay)
        else:
            raise NotImplementedError("The optimizer is not supported.")
        return optimizer
    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("Running on:", device)
        return device
#TODO：simsam network implement
    def D(self,p,z,queue):  #negative cosine similarity
        z = z.detach()   # stop gradient
        return self.nt_xent_criterion(p, z,queue)


    def _step(self, model, head, xis, xjs, queue):
        model.zero_grad()
        # get the representations and the projections
        ris = model(xis)  # [N,C]
        zis = head(ris)
        # get the representations and the projections
        rjs = model(xjs)  # [N,C]
        zjs = head(rjs)
        #norm
        p = F.normalize(zis,dim=1)
        z = F.normalize(zjs, dim=1)
        # update the queue
        queue = torch.cat((queue,z ), 0)
        if queue.shape[0] > 20*160:  #memory size  20 *batch_size
            queue = queue[160:, :]
        loss = self.D(p,z,queue)/2+self.D(p,z,queue)/2
        # loss = self.nt_xent_criterion(p,z,queue)
        return loss

    def train(self):
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader), eta_min=0,
                                                                    last_epoch=-1)
        flag  = 0
        K = 20*160  # the most biggest number of queue
        queue = None
        if queue is None:
            while True:
                with torch.no_grad():
                    for i, (xis, xjs) in enumerate(self.train_loader):
                        xjs = xjs['data'][0].float().to(self.device)
                        self.model.zero_grad()
                        # get the representations and the projections
                        rjs = self.model(xjs)  # [N,C]
                        zjs = self.head(rjs)
                        zjs.detach()
                        k = torch.div(zjs,torch.norm(zjs,dim=1).reshape(-1,1))
                        if queue is None:
                            queue = k
                        else:
                            if queue.shape[0]<K:
                                queue = torch.cat((queue,k),0)
                            else:
                                flag = 1
                        if flag == 1:
                            break
                if flag == 1:
                    break

        # self.model = torch.load(r"E:\tangcheng\idea2\semi_cotrast_seg-master\save\simclr\Hippocampus\b_120_model.pth")
        for epoch_counter in range(self.config['epochs']):
            print("=====Training Epoch: %d =====" % epoch_counter)
            count = 0  # memory the number of
            for i, (xis, xjs) in enumerate(self.train_loader):

                self.optimizer.zero_grad()

                xis = xis['data'][0].float().to(self.device)
                xjs = xjs['data'][0].float().to(self.device)
                loss = self._step(self.model, self.head, xis, xjs, queue)


                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print("Train:[{0}][{1}][{2}] loss: {loss:.4f}".format(epoch_counter, i, len(self.train_loader),
                                                                          loss=loss.item()))

                loss.backward()
                self.optimizer.step()
                # up
                n_iter += 1

            print("===== Validation =====")
            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(self.val_loader,queue)
                print("Val:[{0}] loss: {loss:.5f}".format(epoch_counter, loss=valid_loss))
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(self.model.state_dict(), os.path.join(self.config['save_dir'],
                                                                     'b_{}_model.pth'.format(self.config["batch_size"])))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)


    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, valid_loader,queue):

        # validation steps
        with torch.no_grad():
            self.model.eval()

            valid_loss = 0.0
            counter = 0
            for (xis, xjs) in valid_loader:
                xis = xis['data'][0].float().to(self.device)
                xjs = xjs['data'][0].float().to(self.device)

                loss = self._step(self.model, self.head, xis, xjs, queue)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        return valid_loss
