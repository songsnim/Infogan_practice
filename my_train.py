import torch
import torchvision
import time

from my_utils import *
from config import *
from torch.utils import tensorboard

def train(gen, critic, enc, train_loader, opt_gen, opt_critic, epoch, device):
    start = time.time()
    fixed_noise = torch.randn(BATCH_SIZE, Z_DIM).to(device)
    mse = nn.MSELoss()  
    gen.train()
    critic.train()
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(train_loader):
        real = real.to(device)
        encoded = enc(real).to(device)
        # curr_batch_size = 100
        
        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        # critic을 CRITIC_ITERATIONS번 역전파 할 때, gen은 1번 역전파 한다.
        for _ in range(CRITIC_ITERATIONS):
            latent = torch.randn(BATCH_SIZE, Z_DIM).to(device)
            fake = gen(encoded, latent)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[D(G(z))]
        gen_fake = critic(fake).reshape(-1)
        loss_recon = mse(real, fake)
        loss_fake = -torch.mean(gen_fake)
        loss_gen = loss_recon + loss_fake
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses
        if batch_idx % 100 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch}/{EPOCHS}] Batch {batch_idx}/{len(train_loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )
            with torch.no_grad():
                encoded = enc(real).to(device)
                fake = gen(encoded,fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                
    end = time.time()
    print(end - start)
    return loss_gen, loss_critic, img_grid_real, img_grid_fake