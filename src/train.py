import argparse
import logging
import os
import random
import torch
from torch import distributions
from torch import optim
from tqdm import tqdm

from src import evaluation, inputs
from src.dataset import get_test, get_dataset
from src.models.erik import G, D
from src.losses.gan1 import RFLoss
from src.losses.hgan import HGANLoss
from src.utils import visualize_generated

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", default="rfgan", type=str)
parser.add_argument("--weights", type=str, default='')
parser.add_argument("--nepochs", default=100, type=int)
parser.add_argument("--nheads", required=True, type=int)
parser.add_argument("--type", type=str, choices=['gan1', 'hgan', 'rfgan'])
parser.add_argument("--data_type", type=str, required=True, choices=['grid', 'ring'])
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

# logging general information
logging.info(f'\n***************** {args.exp_name.upper()} **************')
for k, v in args._get_kwargs():
    logging.info(f'{k}: {v}')
logging.info('--------------------------------')

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
variance = 0.0025 if args.data_type == 'grid' else 0.0001
k_value = 8 if args.data_type == 'ring' else 25
lr = 1e-4
beta1 = 0.8
beta2 = 0.999

if args.data_type == 'grid':
    get_data = inputs.get_data_grid
    percent_good = evaluation.percent_good_grid
elif args.data_type == 'ring':
    get_data = inputs.get_data_ring
    percent_good = evaluation.percent_good_ring
else:
    raise NotImplementedError()

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

generator = G()
if args.type == 'gan1':
    assert args.nheads == 1
    discriminator = D(True, 1, False)
elif args.type == 'hgan':
    discriminator = D(True, args.nheads, False)
elif args.type == 'rfgan':
    discriminator = D(True, args.nheads, True)
else:
    raise RuntimeError(f"invalid model type: {args.type}")

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
shared_layers_D_otim = torch.optim.Adam(discriminator.shared_layers.parameters(), lr=lr, betas=(beta1, beta2))
heads_D_optim = []
for i in range(args.nheads):
    optim = torch.optim.Adam(getattr(discriminator, f'head_{i}').parameters(), lr=lr, betas=(beta1, beta2))
    heads_D_optim.append(optim)

start_epoch = 0
if args.weights:
    s = torch.load(args.weights)
    generator.load_state_dict(s['generator'])
    discriminator.load_state_dict(s['discriminator'])
    s = args.weights.split('/')[-1].split('.pt')[0].split('_')[-1]
    start_epoch = int(s) + 1
    logging.info(f"Loaded weights at: {args.weights}")

for epoch in range(start_epoch, nepochs):
    logging.info(f"EPOCH: {epoch}")
    for x_real, _ in tqdm(train_loader):
        z = zdist.sample((train_batch_size, ))
        x_real = x_real.to(device)

        if args.type == 'gan1':
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
        elif args.type == 'hgan':
            g_optimizer.zero_grad()
            gen_imgs = generator(z)
            s = gen_imgs
            lossg = HGANLoss.compute_lossg(discriminator, s)
            lossg.backward()
            g_optimizer.step()

            random_seq = list(range(args.nheads))
            random.shuffle(random_seq)
            for head_id in random_seq:
                shared_layers_D_otim.zero_grad()
                heads_D_optim[head_id].zero_grad()
                g = gen_imgs.detach()
                r = x_real
                lossd = HGANLoss.compute_lossd(discriminator, g, r)
                lossd.backward()
                shared_layers_D_otim.step()
                heads_D_optim[head_id].step()
        elif args.type == 'rfgan':
            g_optimizer.zero_grad()
            gen_imgs = generator(z)
            lossg = RFLoss.compute_lossg(discriminator, gen_imgs)
            lossg.backward()
            g_optimizer.step()

            random_seq = list(range(args.nheads))
            random.shuffle(random_seq)
            for head_id in random_seq:
                shared_layers_D_otim.zero_grad()
                heads_D_optim[head_id].zero_grad()
                g = gen_imgs.detach()
                r = x_real
                lossd = RFLoss.compute_lossd(discriminator, g, r)
                lossd.backward()
                shared_layers_D_otim.step()
                heads_D_optim[head_id].step()
        else:
            raise RuntimeError(f"invalid model type: {args.type}")

    generator.eval()
    discriminator.eval()
    x_fake = generator(z_test).detach().cpu().numpy()

    visualize_generated(x_fake,
                        x_test.detach().cpu().numpy(), epoch,
                        images_folder)
    path = os.path.join(models_folder, f'model_{epoch}.pt')
    if os.path.isfile(path):
        raise RuntimeError(f"File existed at: {path}")
    torch.save(
        {
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict()
        },
        path)
    logging.info(f"Saved model at: {path}")

    percent, modes, kl = percent_good(x_fake, var=variance)
    log_message = f'[epoch {epoch}] dloss = {lossd}, gloss = {lossg}, prop_real = {percent}, modes = {modes}, kl = {kl}'
    logging.info(log_message)

logging.info("Completed!")
