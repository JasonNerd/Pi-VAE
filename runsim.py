## 1. data generation (you should only run it once)
import numpy as np
from pi_vae import *
from util import *

length = 10000
n_cls = 5
n_dim = 100
z_true, u_true, mean_true, lam_true = simulate_data(length, n_cls, n_dim)
np.random.seed(777)
x_true = np.random.poisson(lam_true)
np.savez('../data/sim/sim_100d_poisson_disc_label.npz', u=u_true, z=z_true, x=x_true, lam=lam_true, mean=mean_true)

length = 15000
n_dim = 100
z_true, u_true, mean_true, lam_true = simulate_cont_data_diff_var(length, n_dim)
np.random.seed(777)
x_true = np.random.poisson(lam_true)
np.savez('../data/sim/sim_100d_poisson_cont_label.npz', u=u_true, z=z_true, x=x_true, lam=lam_true, mean=mean_true)

## train a model
def train():
    # 1. load data
    import numpy as np
    dat = np.load('../data/sim/sim_100d_poisson_disc_label.npz');
    u_true = dat['u'];
    z_true = dat['z'];
    x_true = dat['x'];

    x_all = x_true.reshape(50,200,-1);
    u_all = u_true.reshape(50,200,-1);
    x_all = torch.as_tensor(x_all, dtype=torch.float32)
    u_all = torch.as_tensor(u_all)
    print(x_all.shape)
    print(u_all.shape)
    # 2. define model
    vae = VAE(
        x_all[0].shape[-1], 
        2, 
        5, 
        gen_nodes=60, 
        n_blk=2, 
        mdl='poisson', 
        disc=True, 
        learning_rate=5e-4
    )
    # print(vae)
    print("--------------")
    epochs = 10
    for e in range(epochs):
        ttl = 0.
        for i in range(x_all.shape[0]):
            loss = vae.train_step(x_all[i], u_all[i])
            break
            ttl+=loss
        break
        print(f"Epoch {e} loss={ttl/x_all.shape[0]}")
train()

## obviously, it can not deal with batch data
