"""Trainer."""

from typing import Iterable

import torch
from torch import nn
from tqdm.notebook import tqdm
from utils import WassersteinLoss, GradientPenalty


class Trainer():
    """Trainer."""

    def __init__(
        self,
        generator,
        critic,
        g_optimizer,
        c_optimizer,
        device: str = "cuda:0",
    ) -> None:
        """Initialize."""
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer
        self.g_criterion = WassersteinLoss().to(device)
        self.c_criterion = WassersteinLoss().to(device)
        self.c_penalty = GradientPenalty().to(device)
        self.device = device

    def train(
        self,
        dataloader: Iterable,
        epochs: int = 500,
        batch_size: int = 64,
        repeat: int = 5,
        display_step: int = 10,
    ) -> None:
        """Start training process."""
        self.alpha = torch.rand((batch_size, 1, 1, 1, 1)).requires_grad_().to(self.device)
        self.data = {
            "gloss": [],
            "closs": [],
            "cfloss": [],
            "crloss": [],
            "cploss": [],
        }
        for epoch in tqdm(range(epochs)):
            e_gloss = 0
            e_cfloss = 0
            e_crloss = 0
            e_cploss = 0
            e_closs = 0
            for real in dataloader:
                real = real.to(self.device)
                # Train Critic
                b_closs = 0
                b_cfloss = 0
                b_crloss = 0
                b_cploss = 0
                for _ in range(repeat):
                    cords = torch.randn(batch_size, 32).to(self.device)
                    style = torch.randn(batch_size, 32).to(self.device)
                    melody = torch.randn(batch_size, 4, 32).to(self.device)
                    groove = torch.randn(batch_size, 4, 32).to(self.device)

                    self.c_optimizer.zero_grad()
                    with torch.no_grad():
                        fake = self.generator(cords, style, melody, groove).detach()
                    realfake = self.alpha * real + (1. - self.alpha) * fake

                    fake_pred = self.critic(fake)
                    real_pred = self.critic(real)
                    realfake_pred = self.critic(realfake)
                    fake_loss = self.c_criterion(fake_pred, - torch.ones_like(fake_pred))
                    real_loss = self.c_criterion(real_pred, torch.ones_like(real_pred))
                    penalty = self.c_penalty(realfake, realfake_pred)
                    closs = fake_loss + real_loss + 10 * penalty
                    closs.backward(retain_graph=True)
                    self.c_optimizer.step()
                    b_cfloss += fake_loss.item() / repeat
                    b_crloss += real_loss.item() / repeat
                    b_cploss += 10 * penalty.item() / repeat
                    b_closs += closs.item() / repeat
                e_cfloss += b_cfloss / len(dataloader)
                e_crloss += b_crloss / len(dataloader)
                e_cploss += b_cploss / len(dataloader)
                e_closs += b_closs / len(dataloader)
                # Train Generator
                self.g_optimizer.zero_grad()
                cords = torch.randn(batch_size, 32).to(self.device)
                style = torch.randn(batch_size, 32).to(self.device)
                melody = torch.randn(batch_size, 4, 32).to(self.device)
                groove = torch.randn(batch_size, 4, 32).to(self.device)

                fake = self.generator(cords, style, melody, groove)
                fake_pred = self.critic(fake)
                b_gloss = self.g_criterion(fake_pred, torch.ones_like(fake_pred))
                b_gloss.backward()
                self.g_optimizer.step()
                e_gloss += b_gloss.item() / len(dataloader)

            self.data['gloss'].append(e_gloss)
            self.data['closs'].append(e_closs)
            self.data['cfloss'].append(e_cfloss)
            self.data['crloss'].append(e_crloss)
            self.data['cploss'].append(e_cploss)
            if epoch % display_step == 0:
                print(f"Epoch {epoch}/{epochs} | Generator loss: {e_gloss:.3f} | Critic loss: {e_closs:.3f}")
                print(f"(fake: {e_cfloss:.3f}, real: {e_crloss:.3f}, penalty: {e_cploss:.3f})")
