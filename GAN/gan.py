# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import save_image

import argparse
import os
import numpy as np

from GAN.utils import device, init_weight

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help="the total epoch for training")
parser.add_argument('--batch_size', type=int, default=128, help="the size of a training batch")
parser.add_argument('--d_lr', type=float, default=2e-4, help="the learning rate for discriminator")
parser.add_argument('--g_lr', type=float, default=2e-4, help="the learning rate for generator")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--noise_dim', type=int, default=100, help="the dimension of input noise")
parser.add_argument('--img_size', type=int, default=28, help="size of each image dimension")
parser.add_argument('--channels', type=int, default=1, help="number of image channels")
parser.add_argument('--save_interval', type=int, default=400, help="the interval of saving some generated images")
args = parser.parse_args()

os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)
img_shape = (args.channels, args.img_size, args.img_size)


class GeneratorNet(nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(args.noise_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
        self.model.apply(init_weight)

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class DiscriminatorNet(nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.model.apply(init_weight)

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


def configure_data():
    os.makedirs("../data/mnist", exist_ok=True)
    loader = DataLoader(
        datasets.MNIST(
            root="../data/mnist",
            train=True,
            download=False,
            transform=transforms.Compose(
                [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True
    )
    return loader


def main():
    writer = SummaryWriter()
    data_loader = configure_data()
    g = GeneratorNet().to(device)
    d = DiscriminatorNet().to(device)
    loss_func = nn.BCELoss().to(device)

    optimizer_g = opt.Adam(g.parameters(), lr=args.g_lr, betas=(args.b1, args.b2))
    optimizer_d = opt.Adam(d.parameters(), lr=args.d_lr, betas=(args.b1, args.b2))

    for e in range(args.epoch):
        total_d_loss, total_r_loss, total_f_loss, total_g_loss = torch.tensor(0.0).to(device), \
                                                                 torch.tensor(0.0).to(device),\
                                                                 torch.tensor(0.0).to(device),\
                                                                 torch.tensor(0.0).to(device)
        for i, (imgs, _) in enumerate(data_loader):
            fake = torch.zeros(imgs.shape[0], 1).to(device)
            real = torch.ones(imgs.shape[0], 1).to(device)
            noise_a = torch.randn(imgs.shape[0], args.noise_dim).to(device)
            noise_b = torch.randn(imgs.shape[0], args.noise_dim).to(device)
            with torch.no_grad():
                gen_pictures_a = g(noise_a)
            gen_pictures_b = g(noise_b)
            real_pictures = imgs.to(device)

            # 训练D
            gen_scores = d(gen_pictures_a)
            real_scores = d(real_pictures)
            optimizer_d.zero_grad()
            r_loss = loss_func(real_scores, real)
            f_loss = loss_func(gen_scores, fake)
            d_loss = r_loss + f_loss
            total_d_loss += d_loss
            total_r_loss += r_loss
            total_f_loss += f_loss
            d_loss.backward()
            optimizer_d.step()

            # 训练G
            optimizer_g.zero_grad()
            # g_loss = -loss_func(d(gen_pictures.detach()), fake)
            g_loss = loss_func(d(gen_pictures_b), real)
            total_g_loss += g_loss
            g_loss.backward()
            optimizer_g.step()

            print(f"[Epoch:{e+1}] [Batch:{i+1}] [D loss:{d_loss}] [G loss:{g_loss}]")

            batchs_done = e * len(data_loader) + i
            if batchs_done % args.save_interval == 0:
                save_image(gen_pictures_a[np.random.randint(0, args.batch_size // 2, size=25)],
                           f"images/{batchs_done}.png", nrow=5, normalize=True)

        writer.add_scalars('loss', {'d_loss_expectation': total_d_loss/len(data_loader),
                                    'real_loss_expectation': total_r_loss/len(data_loader),
                                    'fake_loss_expectation': total_f_loss/len(data_loader),
                                    'g_loss_expectation': total_g_loss/len(data_loader)}, e)
        if (e+1) % 50 == 0:
            torch.save(g.state_dict(), f"models/g{e+1}.pth")
            torch.save(d.state_dict(), f"models/d{e+1}.pth")


if __name__ == "__main__":
    training = False
    if training:
        main()
    else:
        # d = DiscriminatorNet().load_state_dict(torch.load("models/d200.pth"))
        for i in [50, 100, 150, 200]:
            generator = GeneratorNet().to(device)
            generator.load_state_dict(torch.load(f"models/g{i}.pth"))
            noise = torch.randn(100, args.noise_dim).to(device)
            final_images = generator(noise)
            save_image(final_images[:],
                       f"images/final_{i}.png", nrow=10, normalize=True)
