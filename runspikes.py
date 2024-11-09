import torch
from modules.pivae import VAE
spikes = torch.as_tensor(spikes)  # Spikes shape = (2295, 180, 182)
num_trials, seq_len, num_channel = spikes.shape
held_in = 148
t_in = 140
ratio = 0.5
num_train = int(ratio*num_trials)
spike_train = spikes[:num_train]
vae = VAE(num_channel, dim_z=num_channel//4, dim_u=seq_len+1, gen_nodes=128)
# trainning steps, save model
# epochs = 50
# batch_size = 16
# num_batch = num_train//batch_size
# u_in = torch.cat([torch.arange(seq_len, dtype=torch.int).reshape(-1, 1).unsqueeze(0)]*batch_size)
# for e in range(epochs):
#     tl = 0.0
#     for i in range(num_batch):
#         bstart = i*batch_size
#         x_in = spike_train[i*batch_size: (i+1)*batch_size]
#         loss, rates = vae.train_step(x_in, u_in)
#         tl+=loss
#     print(f"Epoch {e}, loss={loss/num_train}")

# torch.save(vae, "pivae50.pt")

# validation steps
vae = torch.load("pivae50.pt")
u_in = torch.cat([torch.arange(seq_len, dtype=torch.int).reshape(-1, 1).unsqueeze(0)]*num_train)
rates = vae(spike_train, u_in)[3]
sp = spike_train.detach().cpu().numpy()
rates = rates.detach().cpu().numpy()
from nlb_tools.evaluation import bits_per_spike
print(bits_per_spike(rates, sp))
