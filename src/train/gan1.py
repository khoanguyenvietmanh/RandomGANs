import argparse
import logging
import os
import torch
from torch import distributions
from torch import optim

from src import evaluation, inputs
from src.dataset import get_test, get_dataset
from src.models.erik import G, D
from src.losses.gan1 import RFLoss
from src.utils import visualize_generated

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", default="baseline_gan1", type=str)
parser.add_argument("--nepochs", default=100, type=int)
parser.add_argument("--weights", type=str, default='')
parser.add_argument("--it", type=int, default=-1)
args = parser.parse_args()

exp_folder = f'experiments/{args.exp_name}'
if not args.weights:
    os.makedirs(exp_folder, exist_ok=False)
    mode = 'w'
else:
    mode = 'a'
logging.basicConfig(filename=f'{exp_folder}/{args.exp_name}.txt',
                    filemode=mode,
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

logging.info(f'exp_folder: {exp_folder}')
images_folder = os.path.join(exp_folder, 'images')
models_folder = os.path.join(exp_folder, 'models')
os.makedirs(images_folder, exist_ok=True)
os.makedirs(models_folder, exist_ok=True)
logging.info(f'images_folder: {images_folder}')
logging.info(f'models_folder: {models_folder}')

nepochs = args.nepochs
z_dim = 2
test_batch_size = 50000
train_batch_size = 100
npts = 100000
variance = 0.0025
k_value = 25
lr = 1e-4
beta1 = 0.8
beta2 = 0.999


get_data = inputs.get_data_grid
percent_good = evaluation.percent_good_grid
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

zdist = distributions.Normal(torch.zeros(z_dim, device=device),
                             torch.ones(z_dim, device=device))
z_test = zdist.sample((test_batch_size, ))

x_test, y_test = get_test(get_data=get_data,
                          batch_size=test_batch_size,
                          variance=variance,
                          k_value=k_value,
                          device=device)

x_cluster, _ = get_test(get_data=get_data,
                        batch_size=10000,
                        variance=variance,
                        k_value=k_value,
                        device=device)

train_loader = get_dataset(get_data=get_data,
                           batch_size=train_batch_size,
                           npts=npts,
                           variance=variance,
                           k_value=k_value)

N = 1
generator = G()
discriminator = D(True, 1, False)
logging.info(f"generator: {generator}")
logging.info(f"discriminator: {discriminator}")
if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()

g_optimizer = optim.Adam(generator.parameters(),
                         lr=lr,
                         betas=(beta1, beta2))
d_optimizer = optim.Adam(discriminator.parameters(),
                         lr=lr,
                         betas=(beta1, beta2))

it = 0
if args.weights:
    s = torch.load(args.weights)
    generator.load_state_dict(s['generator'])
    discriminator.load_state_dict(s['discriminator'])
    it = args.it + 1
for epoch in range(nepochs):
    for x_real, _ in train_loader:
        z = zdist.sample((train_batch_size, ))
        x_real = x_real.to(device)

        g_optimizer.zero_grad()
        gen_imgs = generator(z)
        lossg = RFLoss.compute_lossg(discriminator, gen_imgs)
        lossg.backward()
        g_optimizer.step()

        d_optimizer.zero_grad()
        g = gen_imgs.detach()
        lossd = RFLoss.compute_lossd(discriminator, g, x_real)
        lossd.backward()
        d_optimizer.step()

        if it % 1000 == 0:
            generator.eval()
            discriminator.eval()
            x_fake = generator(z_test).detach().cpu().numpy()

            visualize_generated(x_fake,
                                x_test.detach().cpu().numpy(), it,
                                images_folder)

            torch.save(
                {
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict()
                },
                os.path.join(models_folder, f'model_{it}.pt'))
            logging.info(os.path.join(models_folder, f'model_{it}.pt'))
            percent, modes, kl = percent_good(x_fake, var=variance)
            log_message = f'[epoch {epoch} it {it}] dloss = {lossd}, gloss = {lossg}, prop_real = {percent}, modes = {modes}, kl = {kl}'
            logging.info(log_message)

        it += 1

logging.info("Completed!")
