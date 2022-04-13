import torch

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

BCE = torch.nn.BCELoss()


# Random GAN
# Baseline 1 - Small GAN with one Discriminator
# Baseline 2 - GAN with one Discriminator and has size equal GAN with multiple heads
class RFLoss():
    @staticmethod
    def compute_lossd(discriminator, gen_imgs, real_imgs):
        real_labels = Tensor(real_imgs.shape[0]).fill_(1.0)
        gen_labels = Tensor(gen_imgs.shape[0]).fill_(0.0)

        real_preds = discriminator(real_imgs, -1)
        real_loss = BCE(real_preds, real_labels)

        gen_preds = discriminator(gen_imgs, -1)
        gen_loss = BCE(gen_preds, gen_labels)

        return real_loss + gen_loss
    
    @staticmethod
    def compute_lossg(discriminator, gen_imgs):
        real_labels = Tensor(gen_imgs.shape[0]).fill_(1.0)
        lossg = -1.0 * BCE(real_labels - discriminator(gen_imgs, -1), real_labels)
        return lossg


class GAN1Loss():
    @staticmethod
    def compute_lossd(discriminator, gen_imgs, real_imgs):
        real_labels = Tensor(real_imgs.shape[0]).fill_(1.0)
        gen_labels = Tensor(gen_imgs.shape[0]).fill_(0.0)

        real_preds = discriminator(real_imgs)
        real_loss = BCE(real_preds, real_labels)

        gen_preds = discriminator(gen_imgs)
        gen_loss = BCE(gen_preds, gen_labels)

        return real_loss + gen_loss
    
    @staticmethod
    def compute_lossg(discriminator, gen_imgs):
        real_labels = Tensor(gen_imgs.shape[0]).fill_(1.0)
        lossg = -1.0 * BCE(real_labels - discriminator(gen_imgs), real_labels)
        return lossg
