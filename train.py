import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pickle as pkl

from argparse import Namespace

from data_prep import get_dataloader
from model import build_network
from utilities import scale


args = Namespace(
    lr=0.0002,
    beta1=0.5,
    beta2=0.999,  # default value
    train_on_gpu=torch.cuda.is_available(),
    d_conv_dim=128,
    g_conv_dim=128,
    z_size=100,
    n_epochs=2,
    batch_size=512,
    img_size=32
)


def real_loss(D_out):
    """Calculates how close discriminator outputs are to being real.
       param, D_out: discriminator logits
       return: real loss"""
    batch_size = D_out.size(0)
    labels = torch.ones(batch_size)  # real labels = 1

    # move labels to GPU if available
    if args.train_on_gpu:
        labels = labels.cuda()

    # binary cross entropy with logits loss
    criterion = nn.BCEWithLogitsLoss()

    # calculate loss
    loss = criterion(D_out.squeeze(), labels)

    return loss


def fake_loss(D_out):
    """Calculates how close discriminator outputs are to being fake.
       param, D_out: discriminator logits
       return: fake loss"""
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)  # fake labels = 0

    if args.train_on_gpu:
        labels = labels.cuda()

    criterion = nn.BCEWithLogitsLoss()

    # calculate loss
    loss = criterion(D_out.squeeze(), labels)

    return loss


def train(D, G, n_epochs, print_every=50):
    """Trains adversarial networks for some number of epochs
       param, D: the discriminator network
       param, G: the generator network
       param, n_epochs: number of epochs to train for
       param, print_every: when to print and record the models' losses
       return: D and G losses"""

    # move models to GPU

    if args.train_on_gpu:
        D.cuda()
        G.cuda()

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size = 16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, args.z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    # move z to GPU if available
    if args.train_on_gpu:
        fixed_z = fixed_z.cuda()

    # epoch training loop
    for epoch in range(n_epochs):

        # batch training loop
        for batch_i, (real_images, _) in enumerate(celeba_train_loader):

            batch_size = real_images.size(0)
            real_images = scale(real_images)

            # ===============================================
            #         YOUR CODE HERE: TRAIN THE NETWORKS
            # ===============================================

            # 1. Train the discriminator on real and fake images
            d_optimizer.zero_grad()

            if args.train_on_gpu:
                real_images = real_images.cuda()

            D_real = D(real_images)

            d_real_loss = real_loss(D_real)

            z = np.random.uniform(-1, 1, size=(batch_size, args.z_size))
            z = torch.from_numpy(z).float()

            if args.train_on_gpu:
                z = z.cuda()

            fake_images = G(z)
            D_fake = D(fake_images)

            d_fake_loss = fake_loss(D_fake)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # 2. Train the generator with an adversarial loss
            g_optimizer.zero_grad()
            z = np.random.uniform(-1, 1, size=(batch_size, args.z_size))
            z = torch.from_numpy(z).float()

            if args.train_on_gpu:
                z = z.cuda()

            fake_images = G(z)
            D_fake = D(fake_images)

            g_loss = real_loss(D_fake)

            g_loss.backward()
            g_optimizer.step()

            # ===============================================
            #              END OF YOUR CODE
            # ===============================================

            # Print some loss stats
            if batch_i % print_every == 0:
                # append discriminator loss and generator loss
                losses.append((d_loss.item(), g_loss.item()))
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch + 1, n_epochs, d_loss.item(), g_loss.item()))

        ## AFTER EACH EPOCH##
        # this code assumes your generator is named G, feel free to change the name
        # generate and save sample, fake images
        G.eval()  # for generating samples
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train()  # back to training mode

    # Save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    # finally return losses
    return losses


if __name__ == '__main__':
    D, G = build_network(args.d_conv_dim, args.g_conv_dim, args.z_size)
    d_optimizer = optim.Adam(D.parameters(), args.lr, (args.beta1, args.beta2))
    g_optimizer = optim.Adam(G.parameters(), args.lr, (args.beta1, args.beta2))

    celeba_train_loader = get_dataloader(args.batch_size, args.img_size)

    losses = train(D, G, n_epochs=args.n_epochs)