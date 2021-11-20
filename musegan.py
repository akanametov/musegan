"""MuseGAN."""

from typing import Iterable
from torch import Tensor

import time
import argparse
from progress.bar import IncrementalBar

import torch
from torch import nn
from gan.generator import MuseGenerator
from gan.critic import MuseCritic
from gan.utils import initialize_weights
from criterion import WassersteinLoss, GradientPenalty


class MuseGAN(object):
    """MuseGAN.

    Parameters
    ----------
    z_dimension: int, optional (default=32)
        Noise space dimension.
    g_channels: int, optional (default=1024)
        Number of hidden channels for Generator.
    g_features: int, optional (default=1024)
        Number of hidden features for Generator.
    c_channels: int, optional (default=128)
        Number of hidden channels for Critic.
    c_features: int, optional (default=1024)
        Number of hidden features for Critic.
    g_lr: float, optional (default=0.001)
        Learning rate for Generator.
    c_lr: float, optional (default=0.001)
        Learning rate for Critic.
    device: str, optional (default="cuda:0")
        Device.

    """

    def __init__(
        self,
        z_dimension: int = 32,
        g_channels: int = 1024,
        g_features: int = 1024,
        c_channels: int = 128,
        c_features: int = 1024,
        g_lr: float = 0.001,
        c_lr: float = 0.001,
        device: str = "cuda:0",
    ) -> None:
        """Initialize."""
        # generator and optimizer
        self.generator = MuseGenerator(
            z_dimension=z_dimension,
            hid_channels=g_channels,
            hid_features=g_features,
            out_channels=1,
        ).to(device)
        self.generator = self.generator.apply(initialize_weights)
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=g_lr,
            betas=(0.5, 0.9),
        )
        # critic and optimizer
        self.critic = MuseCritic(
            hid_channels=c_channels,
            hid_features=c_features,
            out_features=1,
        ).to(device)
        self.critic = self.critic.apply(initialize_weights)
        self.c_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=c_lr,
            betas=(0.5, 0.9),
        )
        # loss functions and gradient penalty (critic is wasserstein-like gan)
        self.g_criterion = WassersteinLoss().to(device)
        self.c_criterion = WassersteinLoss().to(device)
        self.c_penalty = GradientPenalty().to(device)
        self.device = device
        # dictionary to save history
        self.data = {
            "g_loss": [],
            "c_loss": [],
            "cf_loss": [],
            "cr_loss": [],
            "cp_loss": [],
        }
        print('MuseGAN initialized.')

    def train(
        self,
        dataloader: Iterable,
        epochs: int = 500,
        batch_size: int = 64,
        display_epoch: int = 10
    ) -> None:
        """Train GAN.

        Parameters
        ----------
        dataloader: Iterable
            Dataloader.
        epochs: int, (default=500)
            Number of epochs.
        batch_size: int, (default=64)
            Batch size.
        display_epoch: int, (default=10)
            Display step.

        """
        # alpha parameter for mixing images
        self.alpha = torch.rand((batch_size, 1, 1, 1, 1)).requires_grad_().to(self.device)
        for epoch in range(epochs):
            ge_loss, ce_loss = 0, 0
            cfe_loss, cre_loss, cpe_loss = 0, 0, 0
            start = time.time()
            bar = IncrementalBar(f'[Epoch {epoch+1}/{epochs}]', max=len(dataloader))
            for real in dataloader:
                real = real.to(self.device)
                # train Critic
                cb_loss = 0
                cfb_loss, crb_loss, cpb_loss = 0, 0, 0
                for _ in range(5):
                    # create random `noises`
                    cords = torch.randn(batch_size, 32).to(self.device)
                    style = torch.randn(batch_size, 32).to(self.device)
                    melody = torch.randn(batch_size, 4, 32).to(self.device)
                    groove = torch.randn(batch_size, 4, 32).to(self.device)
                    # forward to generator
                    self.c_optimizer.zero_grad()
                    with torch.no_grad():
                        fake = self.generator(cords, style, melody, groove).detach()
                    # mix `real` and `fake` melody
                    realfake = self.alpha * real + (1. - self.alpha) * fake
                    # get critic's `fake` loss
                    fake_pred = self.critic(fake)
                    fake_target = - torch.ones_like(fake_pred)
                    fake_loss = self.c_criterion(fake_pred, fake_target)
                    # get critic's `real` loss
                    real_pred = self.critic(real)
                    real_target = torch.ones_like(real_pred)
                    real_loss = self.c_criterion(real_pred, real_target)
                    # get critic's penalty
                    realfake_pred = self.critic(realfake)
                    penalty = self.c_penalty(realfake, realfake_pred)
                    # sum up losses
                    closs = fake_loss + real_loss + 10 * penalty
                    # retain graph
                    closs.backward(retain_graph=True)
                    # update critic parameters
                    self.c_optimizer.step()
                    # devide by number of critic updates in the loop (5)
                    cfb_loss += fake_loss.item() / 5
                    crb_loss += real_loss.item() / 5
                    cpb_loss += 10 * penalty.item() / 5
                    cb_loss += closs.item() / 5

                cfe_loss += cfb_loss / len(dataloader)
                cre_loss += crb_loss / len(dataloader)
                cpe_loss += cpb_loss / len(dataloader)
                ce_loss += cb_loss / len(dataloader)

                # train generator
                self.g_optimizer.zero_grad()
                # create random `noises`
                cords = torch.randn(batch_size, 32).to(self.device)
                style = torch.randn(batch_size, 32).to(self.device)
                melody = torch.randn(batch_size, 4, 32).to(self.device)
                groove = torch.randn(batch_size, 4, 32).to(self.device)
                # forward to generator
                fake = self.generator(cords, style, melody, groove)
                # forward to critic (to make prediction)
                fake_pred = self.critic(fake)
                # get generator loss (idea is to fool critic)
                gb_loss = self.g_criterion(fake_pred, torch.ones_like(fake_pred))
                gb_loss.backward()
                # update critic parameters
                self.g_optimizer.step()
                ge_loss += gb_loss.item() / len(dataloader)
                bar.next()
            bar.finish()
            end = time.time()
            tm = (end - start)
            # save history
            self.data['g_loss'].append(ge_loss)
            self.data['c_loss'].append(ce_loss)
            self.data['cf_loss'].append(cfe_loss)
            self.data['cr_loss'].append(cre_loss)
            self.data['cp_loss'].append(cpe_loss)
            # display losses
            if epoch % 10 == 0:
                print("[Epoch %d/%d] [G loss: %.3f] [D loss: %.3f] ETA: %.3fs" % (
                    epoch + 1,
                    epochs,
                    ge_loss,
                    ce_loss,
                    tm
                ))
                print(f"[C loss | (fake: {cfe_loss:.3f}, real: {cre_loss:.3f}, penalty: {cpe_loss:.3f})]")
