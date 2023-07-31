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
from networks.unet_con import SupConUnet,MLP

from loss_functions.dice_loss import SoftDiceLoss

from loss_functions.metrics import dice_pytorch, SegmentationMetric
from skimage.io import imsave
import matplotlib.pyplot as plt
from  sklearn import manifold
import numpy  as np
def get_one_hot(label):
    size = label.size()
    num_class = 8
    onehotsize = (size[0],num_class,size[2],size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(onehotsize)).zero_()
    input_label = input_label.scatter_(1,label,1.0)
    return input_label

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def save_images(segs, names, root, mode, iter):
    # b, w, h = segs.shape
    for seg, name in zip(segs, names):
        save_path = os.path.join(root,str(iter)+mode+name+'.png')

        # save_path.mkdir(parents=True, exist_ok=True)
        imsave(str(save_path), seg.cpu().numpy())
class Pix_Projector2(nn.Module):
    def __init__(self, in_channel, out_channel, class_num, output_size=160):
        super(Pix_Projector2, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.size = output_size
        self.class_num = class_num
        self.features = []
        self.mask = []

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channel * 2, in_channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, feature, fusion_label):
        batch_size = feature.size(0)   # batch, 128, height, width
        f2 = self.conv1(feature)
        f3 = F.interpolate(f2, size=(self.size, self.size), mode="bilinear", align_corners=False)
        label = fusion_label
        onehot_label = get_one_hot(label)

        mask = []
        features=[]
        for i in range(batch_size):
            masked_features = onehot_label[i].unsqueeze(1) * f3[i].unsqueeze(0)
            pool_features = masked_features.sum([-2, -1]) / (onehot_label[i].sum([-2, -1]) + 1e-8).unsqueeze(-1)
            s_classes = label[i].unique()
            for cls in s_classes:
                if cls < self.class_num:
                    features.append(pool_features[cls].unsqueeze(0))
                    mask.append(cls.item())

        if len(features) > 0:
            return torch.cat(features).cuda(), torch.tensor(mask).cuda()
        else:
            return torch.Tensor().cuda(), torch.Tensor().cuda()

def plot_semantic_tsne(model, test_data_loader ,projector,device):
    features = []
    targets = []
    iters = 0
    model.eval()
    with torch.no_grad():
        plt.figure(figsize = (9,9))
        for data_batch in test_data_loader :
            s_images = data_batch['data'][0].float().to(device,dtype=torch.float32)
            s_labels= data_batch['seg'][0].long().to(device,dtype=torch.long) #8x1x160x160
            # 8x8x160x160
            s_feature= model(s_images)
            s_con_feature, s_con_label = projector(s_feature, s_labels.clone())
            features = features + s_con_feature.tolist()
            targets = targets + s_con_label.tolist()
            iters += 1
            print(iters)
            if iters % 100 == 0:
                break

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)

    pooling_features_numpy = np.array(features)
    target_numpy = np.array(targets)
    X_tsne = tsne.fit_transform(pooling_features_numpy)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # ��һ��
    bgindex = np.where(target_numpy == 0)[0]
    plt.scatter(X_norm[bgindex, 0], X_norm[bgindex, 1], c="cornflowerblue", marker=".")
    bgindex = np.where(target_numpy == 1)[0]
    plt.scatter(X_norm[bgindex, 0], X_norm[bgindex, 1], c="darkseagreen", marker=".")
    bgindex = np.where(target_numpy == 2)[0]
    plt.scatter(X_norm[bgindex, 0], X_norm[bgindex, 1], c="plum", marker=".")
    bgindex = np.where(target_numpy == 3)[0]
    plt.scatter(X_norm[bgindex, 0], X_norm[bgindex, 1], c="navajowhite", marker=".")
    bgindex = np.where(target_numpy == 4)[0]
    plt.scatter(X_norm[bgindex, 0], X_norm[bgindex, 1], c="gold", marker=".")
    bgindex = np.where(target_numpy == 5)[0]
    plt.scatter(X_norm[bgindex, 0], X_norm[bgindex, 1], c="lightcoral", marker=".")
    bgindex = np.where(target_numpy == 6)[0]
    plt.scatter(X_norm[bgindex, 0], X_norm[bgindex, 1], c="steelblue", marker=".")
    bgindex = np.where(target_numpy == 7)[0]
    plt.scatter(X_norm[bgindex, 0], X_norm[bgindex, 1], c="r", marker=".")
    plt.legend()
    # plt.title("random")
    plt.xticks([])
    plt.yticks([])
    # plt.savefig('./tsne/affectnet_tsne_norm_new.png')
    plt.show()
class SegExperiment(PytorchExperiment):
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

        self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')    #

        self.train_data_loader = NumpyDataSet(self.config.data_dir, target_size=self.config.img_size, batch_size=self.config.batch_size,
                                              keys=tr_keys, do_reshuffle=True)
        self.val_data_loader = NumpyDataSet(self.config.data_dir, target_size=self.config.img_size, batch_size=self.config.batch_size,
                                            keys=val_keys, mode="val", do_reshuffle=True)
        self.test_data_loader = NumpyDataSet(self.config.data_dir, target_size=self.config.img_size, batch_size=self.config.batch_size,
                                             keys=test_keys, mode="test", do_reshuffle=False)
        # self.psne_loader = NumpyDataSet(self.config.data_dir, target_size=self.config.img_size, batch_size=2,
        #                                      keys=test_keys, mode="test", do_reshuffle=False)
        # self.model = UNet(num_classes=self.config.num_classes, num_downs=4)
        self.model = SupConUnet(num_classes=self.config.num_classes)
        self.projector = Pix_Projector2(in_channel=8, out_channel=self.config.img_size, class_num=self.config.num_classes,
                              output_size=self.config.img_size)

	# self.head = MLP(num_class=160)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            # self.model.encoder = nn.DataParallel(self.model.encoder)
            self.model = nn.DataParallel(self.model)
		    # self.projector = nn.DataParallel(self.projector)
            self.projector = nn.DataParallel(self.projector)
        self.model.to(self.device)
        self.projector.to(self.device)
	# self.projector.to(self.device)
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
        print("learning rate is :",self.config.learning_rate)
        self.optimizer = optim.Adam(parameters, lr=self.config.learning_rate)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate)

        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        self.save_checkpoint(name="checkpoint_start")
        self.writter = SummaryWriter(self.elog.work_dir)
        # self.writter = tb_logger.Logger(logdir=self.elog.work_dir, flush_secs=2)
        self.elog.print('Experiment set up.')
        self.elog.print(self.elog.work_dir)

    def train(self, epoch):
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

            pred = self.model(data)
            pred_softmax = F.softmax(pred, dim=1)  # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.
            pred_image = torch.argmax(pred_softmax, dim=1)

            # loss = self.dice_pytorch(outputs=pred_image, labels=target)
            loss = self.ce_loss(pred, target.squeeze()) + self.dice_loss(pred_softmax, target.squeeze())
            # loss = self.dice_loss(pred_softmax, target.squeeze())
            loss.backward()
            self.optimizer.step()

            # Some logging and plotting
            if (batch_counter % self.config.plot_freq) == 0:
                self.elog.print('Train: [{0}][{1}/{2}]\t'
                                'loss {loss:.4f}'.format(epoch, batch_counter, len(self.train_data_loader),
                                                         loss=loss.item()))
                self.writter.add_scalar("train_loss", loss.item(), epoch * len(self.train_data_loader) + batch_counter)

            batch_counter += 1

        assert data is not None, 'data is None. Please check if your dataloader works properly'

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

                pred = self.model(data)
                pred_softmax = F.softmax(pred)  # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.

                pred_image = torch.argmax(pred_softmax, dim=1)
                dice_result = dice_pytorch(outputs=pred_image, labels=target, N_class=self.config.num_classes)
                dice_list.append(dice_result)

                loss = self.dice_loss(pred_softmax, target.squeeze()) # self.ce_loss(pred, target.squeeze())
                loss_list.append(loss.item())

        assert data is not None, 'data is None. Please check if your dataloader works properly'

        # dice_list = np.asarray(dice_list)
        # dice_score = np.mean(dice_list, axis=0)
        # self.scheduler.step(np.mean(loss_list))

        self.elog.print('Epoch: %2d Loss: %.4f' % (self._epoch_idx, np.mean(loss_list)))

        self.writter.add_scalar("val_loss", np.mean(loss_list), epoch)

    def test(self):
        metric_val = SegmentationMetric(self.config.num_classes)
        metric_val.reset()
        self.model.eval()

        num_of_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("number of parameters:", num_of_parameters)

        with torch.no_grad():
            for i, data_batch in enumerate(self.test_data_loader):
                data = data_batch['data'][0].float().to(self.device)
                target = data_batch["seg"][0].long().to(self.device)
                # print(i)
		        # plot_semantic_tsne(self.model,self.psne_loader,device)
                if i>=100 and i%10==0 :
                    plot_semantic_tsne(self.model,self.test_data_loader,self.projector,device)
                output = self.model(data)
                pred_softmax = F.softmax(output, dim=1)
                pred_argmax = torch.argmax(pred_softmax, dim=1)
                # save_images(pred_argmax,str(i),r"/home/labuser2/tangcheng/semi_cotrast_seg-master/save/random", "seg", i)
                # save_images(target.squeeze(),str(i),r"/home/labuser2/tangcheng/semi_cotrast_seg-master/save/random", "groudtruth", i)
                # save_images(data.squeeze(),str(i),r"/home/labuser2/tangcheng/semi_cotrast_seg-master/save/random", "inputimage", i)
                metric_val.update(target.squeeze(dim=1), pred_softmax)
                pixAcc, mIoU, Dice = metric_val.get()
                if (i % self.config.plot_freq) == 0:
                    self.elog.print("Index:%f, mean Dice:%.4f" % (i, Dice))

        _, _, Dice = metric_val.get()
        with open("result.txt", 'a') as f:
            f.write(str(Dice) + "\n")
        print("Overall mean dice score is:", Dice)
        print("Finished test")

    """
    def test(self):

        self.model.eval()
        data = None
        total_ce = []
        total_cls_dice = []
        total_dice_loss = []

        num_of_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("number of parameters:", num_of_parameters)

        with torch.no_grad():
            for file in self.test_keys:
                np_array = np.load(os.path.join(self.config.data_dir, file))
                data = torch.Tensor(np_array[:, 0:1]).float().to(self.device)
                target = torch.Tensor(np_array[:, 1:]).long().to(self.device)

                slice_num = data.shape[0]
                bsz = self.config.batch_size
                pred = []  # prediction of an entire volume

                iter = slice_num // bsz

                for k in range(iter):
                    if k != iter - 1:
                        data_tmp = data[k*bsz:(k+1)*bsz]
                    else:
                        data_tmp = data[k*bsz:]

                    pred_tmp = self.model(data_tmp)
                    pred.append(pred_tmp)

                pred = torch.cat(pred, dim=0)
                pred_softmax = F.softmax(pred, dim=1)  # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.
                # uncertainty_map = torch.max(pred_softmax, dim=1)[0]
                dice_loss = self.dice_loss(pred_softmax, target.squeeze())
                ce_loss = self.ce_loss(pred, target.squeeze())
                pred_image = torch.argmax(pred_softmax, dim=1)
                cls_dice = dice_pytorch(outputs=pred_image, labels=target, N_class =self.config.num_classes)
                print('ce_loss:%.4f   cls_dice:%s dice_loss:%4f' % (ce_loss.data, cls_dice.data, dice_loss.data))
                total_ce.append(ce_loss)
                total_cls_dice.append(cls_dice[None])
                total_dice_loss.append(dice_loss)

            avg_ce = torch.cat(total_ce).mean()
            avg_cls_dice = torch.cat(total_dice_loss, dim=0).mean(dim=0)
            avg_dice_loss = torch.cat(dice_loss).mean()

            self.elog.print("avg_ce:%.4f  avg_cls_dice:%s  avg_dice_loss:%4f" % (avg_ce.item(), avg_cls_dice.item(),
                                                                                 avg_dice_loss.item()))
            print('test_data loading finished')

        assert data is not None, 'data is None. Please check if your dataloader works properly'
        """

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

