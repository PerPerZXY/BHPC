import os
import pickle

import numpy as np
# import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from datasets.two_dim.NumpyDataLoader import NumpyDataSet
from trixi.experiment.pytorchexperiment import PytorchExperiment

from networks.RecursiveUNet import UNet
from networks.unet_con import SupConUnet

from loss_functions.dice_loss import SoftDiceLoss

from loss_functions.metrics import dice_pytorch, SegmentationMetric
from skimage.io import imsave


def save_images(segs, names, root, mode, iter):
    # b, w, h = segs.shape
    for seg, name in zip(segs, names):
        save_path = os.path.join(root, str(iter) + mode + name + '.png')

        # save_path.mkdir(parents=True, exist_ok=True)
        imsave(str(save_path), seg.cpu().numpy())


class MixExperiment(PytorchExperiment):
    """
    The UnetExperiment is inherited from the PytorchExperiment. It implements the basic life cycle for a segmentation
    task with UNet(https://arxiv.org/abs/1505.04597).
    It is optimized to work with the provided NumpyDataLoader.

    The basic life cycle of a UnetExperiment is the same s PytorchExperiment:

        setup()
        (--> Automatically restore values if a previous checkpoint is given)
        prepare()

        for epoch in n_epochs:
            train()
            validate()
            (--> save current checkpoint)

        end()
    """

    def set_loader(self, opt):
        # construct data loader
        pkl_dir = opt.split_dir
        with open(os.path.join(pkl_dir, "splits.pkl"), 'rb') as f:
            splits = pickle.load(f)

        if opt.train_sample == 1:
            tr_keys = splits[opt.fold]['train'] + splits[opt.fold]['val'] + splits[opt.fold]['test']
        else:
            tr_keys = splits[opt.fold]['train']
            tr_size = int(len(tr_keys) * opt.train_sample)
            tr_keys = tr_keys[0:tr_size]

        train_loader = NumpyDataSet(opt.data_dir, target_size=160, batch_size=opt.batch_size,
                                    keys=tr_keys, do_reshuffle=True, mode="supcon")

        return train_loader

    def setup(self):
        pkl_dir = self.config.split_dir
        with open(os.path.join(pkl_dir, "splits.pkl"), 'rb') as f:
            splits = pickle.load(f)

        tr_keys = splits[self.config.fold]['train']
        tr_size = int(len(tr_keys) * self.config.train_sample)
        tr_keys = tr_keys[0:tr_size]
        val_keys = splits[self.config.fold]['val']
        self.test_keys = splits[self.config.fold]['test']
        test_keys = splits[self.config.fold]['test']

        self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')  #

        if self.config.stage == "train" or "test":
            self.train_data_loader = NumpyDataSet(self.config.data_dir, target_size=self.config.img_size,
                                                  batch_size=self.config.batch_size,
                                                  keys=tr_keys, do_reshuffle=True)
        elif self.config.stage == "mix_train":
            self.train_data_loader = self.set_loader(self.config)
            # self.train_data_loader = NumpyDataSet(self.config.data_dir, target_size=self.config.img_size, batch_size=self.config.batch_size,
            #                                   keys=tr_keys, do_reshuffle=True, mode="supcon")

        self.val_data_loader = NumpyDataSet(self.config.data_dir, target_size=self.config.img_size,
                                            batch_size=self.config.batch_size,
                                            keys=val_keys, mode="val", do_reshuffle=True)
        self.test_data_loader = NumpyDataSet(self.config.data_dir, target_size=self.config.img_size,
                                             batch_size=self.config.batch_size,
                                             keys=test_keys, mode="test", do_reshuffle=False)
        # self.model = UNet(num_classes=self.config.num_classes, num_downs=4)
        # initial the alongside model
        self.model = SupConUnet(num_classes=self.config.num_classes)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            # self.model.encoder = nn.DataParallel(self.model.encoder)
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)

        # We use a combination of DICE-loss and CE-Loss in this example.
        # This proved good in the medical segmentation decathlon.
        self.dice_loss = SoftDiceLoss(batch_dice=True, do_bg=False)  # Softmax für DICE Loss!

        # weight = torch.tensor([1, 30, 30]).float().to(self.device)
        self.ce_loss = torch.nn.CrossEntropyLoss()  # Kein Softmax für CE Loss -> ist in torch schon mit drin!
        # self.dice_pytorch = dice_pytorch(self.config.num_classes)

        # If directory for checkpoint is provided, we load it.
        if self.config.do_load_checkpoint:
            if self.config.checkpoint_dir == '':
                print('checkpoint_dir is empty, please provide directory to load checkpoint.')
            else:
                self.load_checkpoint(name=self.config.checkpoint_dir, save_types=("model"))

        if self.config.saved_model_path is not None:
            self.set_model()

        # freeze certain layer if required
        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        # self.optimizer = optim.Adam(parameters, lr=self.config.learning_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        self.save_checkpoint(name="checkpoint_start")
        self.writter = SummaryWriter(self.elog.work_dir)
        # self.writter = tb_logger.Logger(logdir=self.elog.work_dir, flush_secs=2)
        self.elog.print('Experiment set up.')
        self.elog.print(self.elog.work_dir)

    def train(self, epoch):
        if self.config.stage == "mix_train":
            print('=====MIX -- TRAIN=====')
            self.model.train()
            # data = None
            batch_counter = 0

            # self.train_data_loader = self.set_loader(self.config)
            for data_batch in self.train_data_loader:
                self.optimizer.zero_grad()
                data1 = data_batch['data'][0].float().to(self.device)
                target1 = data_batch['seg'][0].long().to(self.device)
            # for idx, data_batch in enumerate(self.train_data_loader):

                # data1 = data_batch[0]['data'][0].float().to(self.device)
                # target1 = data_batch[0]['seg'][0].long().to(self.device)
                #
                # data2 = data_batch[1]['data'][0].float().to(self.device)
                # 对data2 做数据扰动 ：
                # target2 = data_batch[1]['seg'][0].long().to(self.device)
                inputs_1, target_a_1, target_b_1, lam_1 = self.mixup_data(data1, target1, 1.0)
                inputs_2, target_a_2, target_b_2, lam_2 = self.mixup_data(data1, target1, 1.0)

                feature_list1, feature_list2, pred_1, pred_2 = self.model(inputs_1, inputs_2, stage="mix_train")

                pred_softmax_1 = F.softmax(pred_1, dim=-1)  # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.
                pred_softmax_2 = F.softmax(pred_2, dim=-1)

                pred_image_1 = torch.argmax(pred_softmax_1, dim=1)
                pred_image_2 = torch.argmax(pred_softmax_2, dim=1)
                # TODO: each decoder layer loss caclulation
                # loss_kl = 0.0
                loss_kl = F.kl_div(feature_list1[0].softmax(dim=-1).log(), feature_list2[0].softmax(dim=-1),reduction='sum')
                loss_seg = F.kl_div(pred_softmax_1.log(), pred_softmax_2,reduction='mean')
                # for index in range(len(feature_list1)):
                #     loss_kl_1 = F.kl_div(feature_list1[index].softmax(dim=1).log(), feature_list2[index].softmax(dim=1),
                #                    reduction='sum')  # why so large ????  how to slove it
                #     loss_kl = loss_kl+loss_kl_1

                loss_seg_1 = self.mixup_criterian(pred_1, target_a_1, target_b_1, lam_1)
                loss_seg_2 = self.mixup_criterian(pred_2, target_a_2, target_b_2, lam_2)
                # loss = self.dice_loss(pred_softmax, target.squeeze())
                loss = loss_seg_1 + loss_seg_2
                loss.backward()
                self.optimizer.step()

                # Some logging and plotting
                if (batch_counter % self.config.plot_freq) == 0:
                    print('Train: [{0}][{1}/{2}]\t'
                          'loss {loss:.4f}'.format(epoch, batch_counter, len(self.train_data_loader),
                                                   loss=loss.item()))
                    self.writter.add_scalar("train_loss", loss.item(),
                                            epoch * len(self.train_data_loader) + batch_counter)

                batch_counter += 1

            # assert data is not None, 'data is None. Please check if your dataloader works properly'
        elif self.config.stage == "train":
            self.elog.print('=====TRAIN=====')
            self.model.train()

            data = None
            batch_counter = 0
            for data_batch in self.train_data_loader:

                self.optimizer.zero_grad()

                # Shape of data_batch = [1, b, c, w, h]
                # Desired shape = [b, c, w, h]
                # Move data and target to the GPU
                data = data_batch['data'][0].float().to(self.device)
                target = data_batch['seg'][0].long().to(self.device)
                max_value = target.max()
                min_value = target.min()

                inputs, target_a, target_b, lam = self.mixup_data(data, target, 1.0)
                # inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
                # print("inputs.shape:", inputs.shape)
                pred = self.model(inputs, stage="train")
                pred_softmax = F.softmax(pred,dim=1)  # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.
                pred_image = torch.argmax(pred_softmax, dim=1)

                loss = self.mixup_criterian(pred, target_a, target_b, lam)
                # loss = self.dice_loss(pred_softmax, target.squeeze())
                loss.backward()
                self.optimizer.step()

                # Some logging and plotting
                if (batch_counter % self.config.plot_freq) == 0:
                    self.elog.print('Train: [{0}][{1}/{2}]\t'
                                    'loss {loss:.4f}'.format(epoch, batch_counter, len(self.train_data_loader),
                                                             loss=loss.item()))
                    self.writter.add_scalar("train_loss", loss.item(),
                                            epoch * len(self.train_data_loader) + batch_counter)

                batch_counter += 1

            assert data is not None, 'data is None. Please check if your dataloader works properly'

    def test(self, epoch=120):
        metric_val = SegmentationMetric(self.config.num_classes)
        metric_val.reset()
        self.model.eval()

        num_of_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("number of parameters:", num_of_parameters)

        with torch.no_grad():
            for i, data_batch in enumerate(self.test_data_loader):
                data = data_batch['data'][0].float().to(self.device)
                target = data_batch["seg"][0].long().to(self.device)

                output = self.model(data, stage="test")
                pred_softmax = F.softmax(output, dim=1)
                # pred_argmax = torch.argmax(pred_softmax, dim=1)
                # save_images(pred_argmax,str(i),r"/home/labuser2/tangcheng/semi_cotrast_seg-master/save/mixup", "seg", i)
                # save_images(target.squeeze(),str(i),r"/home/labuser2/tangcheng/semi_cotrast_seg-master/save/mixup", "groudtruth", i)
                # save_images(data.squeeze(),str(i),r"/home/labuser2/tangcheng/semi_cotrast_seg-master/save/mixup", "inputimage", i)
                metric_val.update(target.squeeze(dim=1), pred_softmax)
                pixAcc, mIoU, Dice = metric_val.get()
                if (i % self.config.plot_freq) == 0:
                    self.elog.print("Index:%f, mean Dice:%.4f" % (i, Dice))

        _, _, Dice = metric_val.get()
        print("Overall mean dice score is:", Dice)
        with open("result.txt", 'a') as f:
            f.write("epoch:" + str(epoch) + " " + "dice socre:" + str(Dice) + "\n")
        print("Finished test")

    def validate(self, epoch):
        self.elog.print('VALIDATE')
        self.model.eval()

        data = None
        loss_list = []
        dice_list = []

        with torch.no_grad():
            for data_batch in self.val_data_loader:
                data = data_batch['data'][0].float().to(self.device)
                target = data_batch['seg'][0].long().to(self.device)

                pred = self.model(data, stage="test")
                pred_softmax = F.softmax(
                    pred)  # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.

                pred_image = torch.argmax(pred_softmax, dim=1)
                dice_result = dice_pytorch(outputs=pred_image, labels=target, N_class=self.config.num_classes)
                dice_list.append(dice_result)

                loss = self.dice_loss(pred_softmax, target.squeeze())  # self.ce_loss(pred, target.squeeze())
                loss_list.append(loss.item())

        assert data is not None, 'data is None. Please check if your dataloader works properly'

        # dice_list = np.asarray(dice_list)
        # dice_score = np.mean(dice_list, axis=0)
        # self.scheduler.step(np.mean(loss_list))

        self.elog.print('Epoch: %2d Loss: %.4f' % (self._epoch_idx, np.mean(loss_list)))

        self.writter.add_scalar("val_loss", np.mean(loss_list), epoch)
        if epoch >= 90:
            self.test(epoch)

    def set_model(self):
        print("====> start loading model:", self.config.saved_model_path)
        checkpoint = torch.load(self.config.saved_model_path)
        if "model" not in checkpoint.keys():
            state_dict = checkpoint
        else:
            state_dict = checkpoint["model"]
        for k in list(state_dict.keys()):
            if "head" in k:
                del state_dict[k]
        self.model.load_state_dict(state_dict, strict=False)
        print("checkpoint state dict:", state_dict.keys())
        print("model state dict:", self.model.state_dict().keys())
        if self.config.freeze:
            # state_dict = torch.load(self.config.saved_model_path)["model"]
            freeze_list = list(state_dict.keys())
            for name, param in self.model.named_parameters():
                if name in freeze_list:
                    param.requires_grad = False

    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterian(self, pred, target_a, target_b, lam):
        pred_softmax = F.softmax(pred)
        loss1 = self.ce_loss(pred, target_a.squeeze()) + self.dice_loss(pred_softmax, target_a.squeeze())
        loss2 = self.ce_loss(pred, target_b.squeeze()) + self.dice_loss(pred_softmax, target_b.squeeze())
        return lam * loss1 + (1 - lam) * loss2
