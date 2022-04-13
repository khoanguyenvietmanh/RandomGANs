import numpy as np

import torch

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

BCE = torch.nn.BCELoss()


# Baseline 4 - HGAN
class HGANLoss():

    nadir = -1e9
    nadir_slack = 1.5

    @staticmethod
    def compute_lossd(discriminator, gen_imgs, real_imgs):
        m = discriminator.m
        assert m > 1
        s = 0
        for head_id in range(m):
            real_labels = Tensor(real_imgs.shape[0]).fill_(1.0)
            gen_labels = Tensor(gen_imgs.shape[0]).fill_(0.0)

            real_preds = discriminator(real_imgs, head_id)
            real_loss = BCE(real_preds, real_labels)

            gen_preds = discriminator(gen_imgs, head_id)
            gen_loss = BCE(gen_preds, gen_labels)
            s += real_loss + gen_loss
        return s

    @staticmethod
    def compute_lossg(discriminator, gen_imgs):
        real_labels = Tensor(gen_imgs.shape[0]).fill_(1.0)
        loss_g = 0
        losses_list_float = []
        losses_list_var = []
        m = discriminator.m
        for head_id in range(m):
            losses_list_var.append(BCE(discriminator(gen_imgs, head_id), real_labels))
            losses_list_float.append(losses_list_var[-1].item())

        HGANLoss.update_nadir_point(losses_list_float)

        for i, loss in enumerate(losses_list_var):
            loss_g -= torch.log(HGANLoss.nadir - loss)
        return loss_g

    @staticmethod
    def update_nadir_point(losses_list):
        HGANLoss.nadir = float(np.max(losses_list) * HGANLoss.nadir_slack + 1e-8)
