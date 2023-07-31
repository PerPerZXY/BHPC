from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC
# from util import get_gpu_memory_map
class SupConLoss(nn.Module):
    """modified supcon loss for segmentation application, the main difference is that the label for different view
    could be different if after spatial transformation"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None,beta = 0.1):
        # input features shape: [bsz, v, c, w, h]
        # input labels shape: [bsz, v, w, h]
        batch_size = features.shape[0]
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # of size (bsz*v, c, h, w)


        kernels = contrast_feature.permute(0, 2, 3, 1)
        kernels = kernels.reshape(-1, contrast_feature.shape[1], 1, 1)
        # kernels = kernels[non_background_idx]
        #positive loss
        logits = torch.div(F.conv2d(contrast_feature, kernels), self.temperature)  # of size (bsz*v, bsz*v*h*w, h, w)
        logits = logits.permute(1, 0, 2, 3)
        logits = logits.reshape(logits.shape[0], -1)


        if labels is not None:  # to make the label and similarity matrix
            labels = torch.cat(torch.unbind(labels, dim=1), dim=0).float().to(device)

            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)


            bg_bool = torch.eq(labels.squeeze().cpu(), torch.zeros(labels.squeeze().shape))
            non_bg_bool = ~ bg_bool
            non_bg_bool = non_bg_bool.int().to(device)
        else:
            mask = torch.eye(logits.shape[0]//contrast_count).float().to(device)
            mask = mask.repeat(contrast_count, contrast_count)
            # print(mask.shape)

        # mask-out self-contrast cases  ，仅有对角线的元素为1 ，其它为0 .
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(mask.shape[0]).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask # 去掉了对角线的原来像素点，仅仅保留了的同类的像素标签
        neg_mask = 1 - mask   # 其它不同类的像素点为 1
        features1, features2 = contrast_feature[:batch_size, :, :, :], contrast_feature[batch_size:, :, :, :]
        similarity_matrix = torch.cosine_similarity(features1, features2).float().to(device)
        similarity_matrix = torch.cat([similarity_matrix,similarity_matrix],dim=0)
        similarity_matrix = similarity_matrix.contiguous().view(-1, 1)
        similarity_matrix_mask =mask * torch.sqrt(torch.mm(similarity_matrix, similarity_matrix.T))

        label_matrix =mask* torch.sqrt(torch.mm(labels, labels.T))

        sample_hard_mask,sample_hard_similarity = label_matrix,similarity_matrix_mask
        # if sample_hard_similarity>0.99:
        #     weight = 0
        # else:
        #     weight = -beta
        weight = -beta
        # hard positive stradegy
        exp_logits = torch.exp(logits) * logits_mask*torch.exp(sample_hard_similarity*weight)
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        dvm=2
        loss =[]
        region_size = exp_logits.shape[0]//2
        for i in range(dvm):
            for j in range(dvm):
                region_logits = exp_logits[i*region_size:(i+1)*region_size,j*region_size:(j+1)*region_size]
                zero_region = torch.ones_like(region_logits)
                region_logits = torch.cat([torch.cat((region_logits,zero_region),dim=0),
                                           torch.cat((zero_region,zero_region),dim=0)],dim=1)

                neg_logits = torch.exp(logits) * neg_mask
                neg_logits = neg_logits.sum(1, keepdim=True)

                # log_prob = logits - torch.log(exp_logits + neg_logits)
                log_prob = logits - torch.log(region_logits + neg_logits)

        # compute mean of log-likelihood over positive
                mean_log_prob_pos = (sample_hard_mask*mask * log_prob).sum(1) / mask.sum(1)
        # loss
                tem_loss = mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
                if labels is not None:
            # only consider the contrastive loss for non-background pixel
                    tem_loss = (-tem_loss * non_bg_bool).sum() / (non_bg_bool.sum())
                    loss.append(tem_loss)
                else:
                    loss = loss.mean()
        loss= torch.stack(loss).mean()
        return loss


class SupConSegLoss(nn.Module):
    # TODO: only support batch size = 1
    def __init__(self, temperature=0.7,pixel_queue = None):
        super(SupConSegLoss, self).__init__()
        self.temp = temperature
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.pixel_queue = pixel_queue
    def forward(self, features, labels=None):
        # input features: [bsz, c, h ,w], h & w are the image size
        shape = features.shape
        img_size = shape[-1]
        if labels is not None:
            f1, f2 = torch.split(features, [1, 1], dim=1)
            features = torch.cat([f1.squeeze(1), f2.squeeze(1)], dim=0)
            l1, l2 = torch.split(labels, [1, 1], dim=1)
            labels = torch.cat([l1.squeeze(1), l2.squeeze(1)], dim=0)
            # features = features.squeeze(dim=1)
            # labels = labels.squeeze(dim=1)
            bsz = features.shape[0]
            loss = []
            for b in range(bsz):
                # print("Iteration index:", idx, "Batch_size:", b)
                for i in range(img_size):
                    # print("before ith iteration, the consumption memory is:", torch.cuda.memory_allocated() / 1024**2)
                    for j in range(img_size):
                        x = features[b:b + 1, :, i:i + 1, j:j + 1]  # [1,c, 1, 1, 1]
                        x_label = labels[b, i, j] + 1  # avoid cases when label=0
                        if x_label == 1:  # ignore background
                            continue
                        cos_dst = F.conv2d(features, x)  # [2b, 1, 512, 512]
                        cos_dst = torch.div(cos_dst.squeeze(dim=1), self.temp)
                        # print("cos_dst:", cos_dst.max(), cos_dst.min())
                        self_contrast_dst = torch.div((x * x).sum(), self.temp)

                        mask = labels + 1
                        mask[mask != x_label] = 0
                        # if mask.sum() < 5:
                        #    print("Not enough same label pixel")
                        #    continue
                        mask = torch.div(mask, x_label)
                        numerator = (mask * cos_dst).sum() - self_contrast_dst
                        denominator = torch.exp(cos_dst).sum() - torch.exp(self_contrast_dst)
                        # print("denominator:", denominator.item())
                        # print("numerator:", numerator.max(), numerator.min())
                        loss_tmp = torch.log(denominator) - numerator / (mask.sum() - 1)
                        if loss_tmp != loss_tmp:
                            print(numerator.item(), denominator.item())

                        loss.append(loss_tmp)
            if len(loss) == 0:
                loss = torch.tensor(0).float().to(self.device)
                return loss
            loss = torch.stack(loss).mean()
            return loss

        else:
            bsz = features.shape[0]
            loss = []
            for b in range(bsz):
                # print("Iteration index:", idx, "Batch_size:", b)
                tmp_feature = features[b]
                for n in range(tmp_feature.shape[0]):
                    for i in range(img_size):
                        # print("before ith iteration, the consumption memory is:", torch.cuda.memory_allocated() / 1024**2)
                        for j in range(img_size):
                            x = tmp_feature[n:n+1, :, i:i + 1, j:j + 1]  # [c, 1, 1, 1]
                            cos_dst = F.conv2d(tmp_feature, x)  # [2b, 1, 512, 512]
                            cos_dst = torch.div(cos_dst.squeeze(dim=1), self.temp)
                            # print("cos_dst:", cos_dst.max(), cos_dst.min())
                            self_contrast_dst = torch.div((x * x).sum(), self.temp)

                            mask = torch.zeros((tmp_feature.shape[0], tmp_feature.shape[2], tmp_feature.shape[3]),
                                               device=self.device)
                            mask[0:tmp_feature.shape[0], i, j] = 1
                            numerator = (mask * cos_dst).sum() - self_contrast_dst
                            denominator = torch.exp(cos_dst).sum() - torch.exp(self_contrast_dst)
                            # print("numerator:", numerator.max(), numerator.min())
                            loss_tmp = torch.log(denominator) - numerator / (mask.sum() - 1)
                            if loss_tmp != loss_tmp:
                                print(numerator.item(), denominator.item())

                            loss.append(loss_tmp)

            loss = torch.stack(loss).mean()
            return loss


class ContrastCELoss(nn.Module):
    def __init__(self, temperature=0.7, stride=1,weight = 1):
        super(ContrastCELoss, self).__init__()
        self.temp = temperature
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.supconloss = SupConLoss(temperature=self.temp)
        self.stride = stride
        self.weight = weight
    def forward(self, features, labels=None):
        # input features: [bsz, num_view, c, h ,w], h & w are the image size
        features = features[:, :, :, ::self.stride, ::self.stride]  # resample feature maps to reduce memory consumption and running time
        features = features
        shape = features.shape
        img_size = shape[-1]
        if labels is not None:
            labels = labels[:, :, ::self.stride, ::self.stride]
            pass
            if labels.sum() == 0:
                loss = torch.tensor(0).float().to(self.device)
                return loss

            loss = self.supconloss(features, labels)
            # ce_loss = nn.CrossEntropyLoss(features,labels)
            # loss = ce_loss+self.weight*loss
            """
            f1, f2 = torch.split(features, [1, 1], dim=1)
            features = torch.cat([f1.squeeze(1), f2.squeeze(1)], dim=0)
            l1, l2 = torch.split(labels, [1, 1], dim=1)
            labels = torch.cat([l1.squeeze(1), l2.squeeze(1)], dim=0)
            bsz = features.shape[0]
            loss = []
            for b in range(bsz):
                # print("Iteration index:", idx, "Batch_size:", b)
                for i in range(img_size):
                    # print("before ith iteration, the consumption memory is:", torch.cuda.memory_allocated() / 1024**2)
                    for j in range(img_size):
                        x = features[b:b + 1, :, i:i + 1, j:j + 1]  # [c, 1, 1, 1]
                        x_label = labels[b, i, j] + 1  # avoid cases when label=0
                        if x_label == 1:  # ignore background
                            continue
                        cos_dst = F.conv2d(features, x)  # [2b, 1, 512, 512]
                        cos_dst = torch.div(cos_dst.squeeze(dim=1), self.temp)
                        self_contrast_dst = torch.div((x * x).sum(), self.temp)

                        mask = labels + 1
                        mask[mask != x_label] = 0
                        mask = torch.div(mask, x_label)
                        numerator = (mask * cos_dst).sum() - self_contrast_dst
                        denominator = torch.exp(cos_dst).sum() - torch.exp(self_contrast_dst)
                        # print("denominator:", denominator.item())
                        # print("numerator:", numerator.max(), numerator.min())
                        loss_tmp = torch.log(denominator) - numerator / (mask.sum() - 1)
                        if loss_tmp != loss_tmp:
                            print(numerator.item(), denominator.item())

                        loss.append(loss_tmp)

            if len(loss) == 0:
                loss = torch.tensor(0).float().to(self.device)
                return loss
            loss = torch.stack(loss).mean()
            """
            return loss
        else:
            bsz = features.shape[0]
            loss = self.supconloss(features)

            """
            loss = []
            for b in range(bsz):
                # print("Iteration index:", idx, "Batch_size:", b)
                tmp_feature = features[b]
                for n in range(tmp_feature.shape[0]):
                    for i in range(img_size):
                        # print("before ith iteration, the consumption memory is:", torch.cuda.memory_allocated() / 1024**2)
                        for j in range(img_size):
                            x = tmp_feature[n:n+1, :, i:i + 1, j:j + 1]  # [c, 1, 1, 1]
                            cos_dst = F.conv2d(tmp_feature, x)  # [2b, 1, 512, 512]
                            cos_dst = torch.div(cos_dst.squeeze(dim=1), self.temp)
                            # print("cos_dst:", cos_dst.max(), cos_dst.min())
                            self_contrast_dst = torch.div((x * x).sum(), self.temp)

                            mask = torch.zeros((tmp_feature.shape[0], tmp_feature.shape[2], tmp_feature.shape[3]),
                                               device=self.device)
                            mask[0:tmp_feature.shape[0], i, j] = 1
                            numerator = (mask * cos_dst).sum() - self_contrast_dst
                            denominator = torch.exp(cos_dst).sum() - torch.exp(self_contrast_dst)
                            # print("numerator:", numerator.max(), numerator.min())
                            loss_tmp = torch.log(denominator) - numerator / (mask.sum() - 1)
                            if loss_tmp != loss_tmp:
                                print(numerator.item(), denominator.item())

                            loss.append(loss_tmp)

            loss = torch.stack(loss).mean()
            """
            return loss


# class Patch_Region_Loss(nn.Module):
#     """modified supcon loss for segmentation application, the main difference is that the label for different view
#     could be different if after spatial transformation"""
#
#     def __init__(self, temperature=0.07, contrast_mode='all',
#                  base_temperature=0.07):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature
#         self.contrast_mode = contrast_mode
#         self.base_temperature = base_temperature
#
#     def sample_hard_and_weight(self, mat, label):  # min_25
#         device = torch.device('cuda')
#         p_size = 3  # 类别个数， 分割结果，hippo 为 3 ,mmhws 为7
#         sum = torch.tensor([0, 0, 0])
#         for i in range(p_size):
#             for j in range(p_size):
#                 sum[int(label[i, j])] += 1
#         sum = [int(x * 0.25) for x in sum]
#         total = [[], [], []]
#         for i in range(p_size):
#             for j in range(p_size):
#                 x = int(label[i, j])
#                 total[x].append(float(mat[i, j]))
#         standard = [-1, -1, -1]
#         for i in range(3):
#             total[i].sort()
#             if (sum[i] > 0): standard[i] = total[i][sum[i] - 1]
#         result = torch.ones_like(label).float().to(device)
#         for i in range(p_size):
#             for j in range(p_size):
#                 x = int(label[i, j])
#                 result[i, j] = 1 if (mat[i, j] <= standard[x]) else 0
#                 mat[i, j] = mat[i, j] if (mat[i, j] <= standard[x]) else 0
#         return result, mat.detach()
#
#     def notation_matrix(self, sim_matrix):
#         note = torch.ones_like(sim_matrix)
#         for i in range(sim_matrix.shape[0]):
#             for j in range(sim_matrix.shape[1]):
#                 note[i, j] = 1 if sim_matrix[i, j] > 0 else -1
#         return note
#
#     def forward(self, features, labels=None ,queue=None, beta=0.1):
#         # input features shape: [bsz, v, c, w, h]
#         # input labels shape: [bsz, v, w, h]
#         batch_size = features.shape[0]
#         device = (torch.device('cuda')
#                   if features.is_cuda
#                   else torch.device('cpu'))
#
#         if len(features.shape) < 3:
#             raise ValueError('`features` needs to be [bsz, n_views, ...],'
#                              'at least 3 dimensions are required')
#
#         contrast_count = features.shape[1]
#         contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # of size (bsz*v, c, h, w)
#
#         kernels = contrast_feature.permute(0, 2, 3, 1)
#         kernels = kernels.reshape(-1, contrast_feature.shape[1], 1, 1)
#         # kernels = kernels[non_background_idx]
#         # positive loss
#         logits = torch.div(F.conv2d(contrast_feature, kernels), self.temperature)  # of size (bsz*v, bsz*v*h*w, h, w)
#         logits = logits.permute(1, 0, 2, 3)
#         logits = logits.reshape(logits.shape[0], -1)
#         logits_max, _ = torch.max(logits, dim=1, keepdim=True)  # add
#         logits = logits - logits_max.detach()
#
#         if labels is not None:  # to make the label and similarity matrix
#             labels = torch.cat(torch.unbind(labels, dim=1), dim=0).float().to(device)
#
#             labels = labels.contiguous().view(-1, 1)
#             mask = torch.eq(labels, labels.T).float().to(device)
#
#             bg_bool = torch.eq(labels.squeeze().cpu(), torch.zeros(labels.squeeze().shape))
#             non_bg_bool = ~ bg_bool
#             non_bg_bool = non_bg_bool.int().to(device)
#         else:
#             mask = torch.eye(logits.shape[0] // contrast_count).float().to(device)
#             mask = mask.repeat(contrast_count, contrast_count)
#             # print(mask.shape)
#
#         # mask-out self-contrast cases  ，仅有对角线的元素为1 ，其它为0 .
#         logits_mask = torch.scatter(
#             torch.ones_like(mask),
#             1,
#             torch.arange(mask.shape[0]).view(-1, 1).to(device),
#             0
#         )
#
#         mask = mask * logits_mask  # 去掉了对角线的原来像素点，仅仅保留了的同类的像素标签
#         neg_mask = 1 - mask  # 其它不同类的像素点为 1
#         features1, features2 = contrast_feature[:batch_size, :, :, :], contrast_feature[batch_size:, :, :, :]
#         similarity_matrix = torch.cosine_similarity(features1, features2).float().to(device)
#         similarity_matrix = torch.cat([similarity_matrix, similarity_matrix], dim=0)
#         similarity_matrix = similarity_matrix.contiguous().view(-1, 1)
#         similarity_matrix_mask = mask * torch.sqrt(torch.mm(similarity_matrix, similarity_matrix.T))
#
#         label_matrix = mask * torch.sqrt(torch.mm(labels, labels.T))
#
#         sample_hard_mask, sample_hard_similarity = self.sample_hard_and_weight(similarity_matrix_mask, label_matrix)
#
#         # hard positive stradegy
#         # 保留前25% 的正对 给予weight
#         exp_logits = torch.exp(logits) * logits_mask * torch.exp(-sample_hard_similarity * beta)
#
#         # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
#         # if queue is not None:
#         #     neg_logits =
#         neg_logits = torch.exp(logits) * neg_mask
#         neg_logits = neg_logits.sum(1, keepdim=True)
#
#         # update queue
#         memory_size = 100 * logits.shape[0]
#         queue = torch.cat([queue, logits * neg_mask], dim=0)
#         if queue > memory_size:
#             queue = queue[logits.shape[0]:, :]
#
#         log_prob = logits - torch.log(exp_logits + neg_logits)
#
#         # compute mean of log-likelihood over positive
#         mean_log_prob_pos = (sample_hard_mask * mask * log_prob).sum(1) / mask.sum(1)
#
#         # loss
#         loss = - mean_log_prob_pos
#         # loss = loss.view(anchor_count, batch_size).mean()
#         if labels is not None:
#             # only consider the contrastive loss for non-background pixel
#             loss = (loss * non_bg_bool).sum() / (non_bg_bool.sum())
#         else:
#             loss = loss.mean()
#         return loss

class BlockConLoss(nn.Module):
    def __init__(self, temperature=0.7, block_size=16,pixel_queue = None):
        super(BlockConLoss, self).__init__()
        self.block_size = block_size
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.supconloss = SupConLoss(temperature=temperature)
        self.pixel_queue = pixel_queue
    def forward(self, features, labels=None,pixel_queue=None):
        # input features: [bsz, num_view, c, h ,w], h & w are the image size
        shape = features.shape
        img_size = shape[-1]
        div_num = img_size // self.block_size
        if labels is not None:
            loss = []
            # region_features_matrix = torch.ones((div_num,div_num))
            # region_labels_matrix = torch.ones((div_num, div_num))
            # for i in range(div_num-1):
            #     for j in range(div_num-1):
            #         x_strat = i*self.block_size
            #         x_end = self.block_size+self.block_size+i*self.block_size # 16
            #         y_start = j*self.block_size
            #         y_end = self.block_size+self.block_size+j*self.block_size #  16
            #         if x_end < features.shape[3] and y_end < features.shape[3]:
            #             region_features_matrix[i,j] = features[:, :, :, x_strat:x_end,
            #                       y_start:y_end]
            #             region_labels_matrix[i,j] = labels[:, :, x_strat:x_end,y_start:y_end]
            #         elif x_end == features.shape[3]:
            #             region_features_matrix[i, j] = features[:, :, :, x_strat:x_end,
            #                                            y_start:y_end]
            #             region_labels_matrix[i, j+1] = labels[:, :, x_strat:x_end, y_start:y_end]
            #             region_features_matrix[i, j+1] = region_features_matrix[i, j]
            #         elif y_end == features.shape[3]:
            #             region_features_matrix[i, j] = features[:, :, :, x_strat:x_end,
            #                                            y_start:y_end]
            #             region_labels_matrix[i+1, j] = labels[:, :, x_strat:x_end, y_start:y_end]
            #             region_features_matrix[i+1, j] = region_features_matrix[i, j]
            for i in range(div_num):
                # print("Iteration index:", idx, "Batch_size:", b)
                for j in range(div_num):
                    # print("before ith iteration, the consumption memory is:", torch.cuda.memory_allocated() / 1024**2)
                    block_features = features[:, :, :, i*self.block_size:(i+1)*self.block_size,
                                  j*self.block_size:(j+1)*self.block_size]
                    block_labels = labels[:,:, i*self.block_size:(i+1)*self.block_size,
                                  j*self.block_size:(j+1)*self.block_size]
                    if block_labels.sum() == 0:
                        continue
                    # # reapeat the patch 4 times1 ---> 4
                    # block_features =torch.cat([torch.cat((block_features,block_features),dim=3),torch.cat((block_features,block_features),dim=3)],dim=4)
                    # block_labels = torch.cat([torch.cat((block_labels, block_labels), dim=3),
                    #                             torch.cat((block_labels, block_labels), dim=3)], dim=4)

                    tmp_loss = self.supconloss(block_features, block_labels)

                    loss.append(tmp_loss)

            if len(loss) == 0:
                loss = torch.tensor(0).float().to(self.device)
                return loss
            loss = torch.stack(loss).mean()
            return loss

        else:
            loss = []
            for i in range(div_num):
                # print("Iteration index:", idx, "Batch_size:", b)
                for j in range(div_num):
                    # print("before ith iteration, the consumption memory is:", torch.cuda.memory_allocated() / 1024**2)
                    block_features = features[:, :, :, i * self.block_size:(i + 1) * self.block_size,
                                     j * self.block_size:(j + 1) * self.block_size]

                    tmp_loss = self.supconloss(block_features)

                    loss.append(tmp_loss)

            loss = torch.stack(loss).mean()
            return loss