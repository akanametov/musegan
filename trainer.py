import torch
from torch import nn
from tqdm.notebook import tqdm
from criterion import WassersteinLoss, GradientPenalty


class Trainer():
    def __init__(self, generator, critic, g_optimizer, c_optimizer, device='cuda:0'):
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer  
        self.g_criterion=WassersteinLoss().to(device)
        self.c_criterion=WassersteinLoss().to(device)
        self.c_penalty=GradientPenalty().to(device)
        
    def train(self, dataloader, epochs=500, N=64, repeat=5, display_step=10, device='cuda:0'):
        self.alpha = torch.rand((64, 1, 1, 1, 1)).requires_grad_().to(device)
        self.data={'gloss':[], 'closs':[], 'cfloss':[], 'crloss':[], 'cploss':[]}
        for epoch in tqdm(range(epochs)):
            e_gloss=0
            
            e_cfloss=0
            e_crloss=0
            e_cploss=0
            e_closs=0
            for real in dataloader:
                real = real.to(device)
                # Train Critic
                b_closs=0
                b_cfloss=0
                b_crloss=0
                b_cploss=0
                for _ in range(repeat):
                    cords = torch.randn(N, 32).to(device)
                    style = torch.randn(N, 32).to(device)
                    melody = torch.randn(N, 4, 32).to(device)
                    groove = torch.randn(N, 4, 32).to(device)
                    
                    self.c_optimizer.zero_grad()
                    with torch.no_grad():
                        fake = self.generator(cords, style, melody, groove).detach()
                    realfake = self.alpha* real + (1. - self.alpha)* fake

                    fake_pred = self.critic(fake)
                    real_pred = self.critic(real)
                    realfake_pred = self.critic(realfake)
                    fake_loss = self.c_criterion(fake_pred, -torch.ones_like(fake_pred))
                    real_loss = self.c_criterion(real_pred,  torch.ones_like(real_pred))
                    penalty = self.c_penalty(realfake, realfake_pred)
                    closs = fake_loss + real_loss + 10* penalty 
                    closs.backward(retain_graph=True)
                    self.c_optimizer.step()
                    b_cfloss += fake_loss.item()/repeat
                    b_crloss += real_loss.item()/repeat
                    b_cploss += 10* penalty.item()/repeat
                    b_closs += closs.item()/repeat
                e_cfloss += b_cfloss/len(dataloader) 
                e_crloss += b_crloss/len(dataloader)
                e_cploss += b_cploss/len(dataloader)
                e_closs += b_closs/len(dataloader)
                # Train Generator
                self.g_optimizer.zero_grad()
                cords = torch.randn(N, 32).to(device)
                style = torch.randn(N, 32).to(device)
                melody = torch.randn(N, 4, 32).to(device)
                groove = torch.randn(N, 4, 32).to(device)
                
                fake = self.generator(cords, style, melody, groove)
                fake_pred = self.critic(fake)
                b_gloss = self.g_criterion(fake_pred, torch.ones_like(fake_pred))
                b_gloss.backward()
                self.g_optimizer.step()
                e_gloss += b_gloss.item()/len(dataloader)
                
            self.data['gloss'].append(e_gloss)
            self.data['closs'].append(e_closs)
            self.data['cfloss'].append(e_cfloss)
            self.data['crloss'].append(e_crloss)
            self.data['cploss'].append(e_cploss)
            if epoch% display_step==0:
                print(f'Epoch {epoch}/{epochs} | Generator loss: {e_gloss:.3f} | '\
                      + f'Critic loss: {e_closs:.3f} (fake: {e_cfloss:.3f}, real: {e_crloss:.3f}, penalty: {e_cploss:.3f})')