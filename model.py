import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def slice_func(x, start, size):
    return x[:, start:start+size]

def perm_func(x, ind):
    return x.index_select(-1, torch.as_tensor(ind, device=x.device))

def squeeze_func(x):
    return x.squeeze(1)

def softplus_func(x):
    return F.softplus(x)

def sigmoid_func(x):
    return torch.sigmoid(x)

def clip_func(x, min_value=1e-7, max_value=1e7):
    return torch.clamp(x, min_value, max_value)

def trans_func(x0, x1, x2):
    return x0 * torch.exp(x1) + x2

def sum_func(x):
    return torch.sum(-x, dim=-1, keepdim=True)

def clamp_func(x):
    return 0.1 * torch.tanh(x)

def sampling(z_mean, z_log_var):
    batch = z_mean.size(0)
    dim = z_mean.size(1)
    epsilon = torch.randn(batch, dim, device=z_mean.device)
    return z_mean + torch.exp(0.5 * z_log_var) * epsilon

def compute_posterior(z_mean, z_log_var, lam_mean, lam_log_var):
    post_mean = (z_mean / (1 + torch.exp(z_log_var - lam_log_var))) + (lam_mean / (1 + torch.exp(lam_log_var - z_log_var)))
    post_log_var = z_log_var + lam_log_var - torch.log(torch.exp(z_log_var) + torch.exp(lam_log_var))
    return post_mean, post_log_var

class FirstNFlowLayer(nn.Module):
    def __init__(self, dim_x, dim_z=2, min_gen_nodes=30):
        super(FirstNFlowLayer, self).__init__()
        gen_nodes = max(min_gen_nodes, dim_x // 4)
        n_nodes = [gen_nodes, gen_nodes, dim_x - dim_z]
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(dim_z, n_nodes[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(n_nodes[0], n_nodes[1]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(n_nodes[1], n_nodes[2]))
        
    def forward(self, z_input):
        output = z_input
        for layer in self.layers:
            output = layer(output)
        output = torch.cat([z_input, output], dim=-1)
        return output

class AffineCouplingLayer(nn.Module):
    def __init__(self, dim_x, min_gen_nodes=30, dd=None):
        super(AffineCouplingLayer, self).__init__()
        DD = dim_x
        if dd is None:
            dd = DD // 2
        
        self.dd = dd
        self.dim_x = dim_x
        self.clamp_func = clamp_func
        self.trans_func = trans_func
        self.sum_func = sum_func
        self.slice_func = slice_func
        
        n_nodes = [max(min_gen_nodes, DD // 4), max(min_gen_nodes, DD // 4), 2 * (DD - dd) - 1]
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(dd, n_nodes[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(n_nodes[0], n_nodes[1]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(n_nodes[1], n_nodes[2]))
        
    def forward(self, layer_input):
        """layer_input.shape[-1] == dim_x"""
        x_input1 = self.slice_func(layer_input, 0, self.dd)
        x_input2 = self.slice_func(layer_input, self.dd, self.dim_x - self.dd)
        st_output = x_input1
        
        for layer in self.layers:
            st_output = layer(st_output)
        
        s_output = self.slice_func(st_output, 0, self.dim_x - self.dd - 1)
        t_output = self.slice_func(st_output, self.dim_x - self.dd - 1, self.dim_x - self.dd)
        s_output = self.clamp_func(s_output)
        s_output = torch.cat([s_output, self.sum_func(s_output)], dim=-1)
        trans_x = self.trans_func(x_input2, s_output, t_output)
        output = torch.cat([trans_x, x_input1], dim=-1)
        return output

class AffineCouplingBlock(nn.Module):
    def __init__(self, dim_x, min_gen_nodes=30, dd=None):
        super(AffineCouplingBlock, self).__init__()
        self.layers = nn.ModuleList([AffineCouplingLayer(dim_x, min_gen_nodes, dd) for _ in range(2)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DecodeNFlowFunc(nn.Module):
    def __init__(self, dim_x, dim_z, n_blk, mdl, min_gen_nodes=30, dd=None):
        super(DecodeNFlowFunc, self).__init__()
        self.first_nflow_layer = FirstNFlowLayer(dim_x, dim_z, min_gen_nodes)
        self.affine_coupling_blocks = nn.ModuleList()
        self.permute_indices = []
        
        for ii in range(n_blk):
            np.random.seed(ii)
            self.permute_indices.append(torch.tensor(np.random.permutation(dim_x)))
            self.affine_coupling_blocks.append(AffineCouplingBlock(dim_x, min_gen_nodes, dd))
        
        if mdl == 'poisson':
            self.final_activation = softplus_func
    
    def forward(self, z_input):
        output = self.first_nflow_layer(z_input)
        for ii in range(len(self.affine_coupling_blocks)):
            output = perm_func(output, self.permute_indices[ii])
            output = self.affine_coupling_blocks[ii](output)
        if hasattr(self, 'final_activation'):
            output = self.final_activation(output)
        return output

class DecodeFunc(nn.Module):
    def __init__(self, dim_x, dim_z, gen_nodes, mdl):
        super(DecodeFunc, self).__init__()
        n_nodes = [gen_nodes, gen_nodes, dim_x]
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(dim_z, n_nodes[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(n_nodes[0], n_nodes[1]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(n_nodes[1], n_nodes[2]))

        if mdl == 'poisson':
            self.final_activation = softplus_func
    
    def forward(self, z_input):
        output = z_input
        for layer in self.layers:
            output = layer(output)
        if hasattr(self, 'final_activation'):
            output = self.final_activation(output)
        return output

class EncodeFunc(nn.Module):
    def __init__(self, dim_x, dim_z, gen_nodes):
        super(EncodeFunc, self).__init__()
        n_nodes = [gen_nodes, gen_nodes, dim_z]
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(dim_x, n_nodes[0]))
        self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(n_nodes[0], n_nodes[1]))
        self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(n_nodes[1], n_nodes[2]))
    
    def forward(self, x_input):
        output = x_input
        for layer in self.layers:
            output = layer(output)
        return output

class ZPriorNN(nn.Module):
    def __init__(self, dim_z, dim_u):
        super(ZPriorNN, self).__init__()
        n_hidden_nodes_in_prior = 20
        n_nodes = [n_hidden_nodes_in_prior, n_hidden_nodes_in_prior, 2 * dim_z]
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(dim_u, n_nodes[0]))
        self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(n_nodes[0], n_nodes[1]))
        self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(n_nodes[1], n_nodes[2]))
    
    def forward(self, u_input):
        output = u_input
        for layer in self.layers:
            output = layer(output)
        lam_mean = output[:, :output.shape[-1] // 2]
        lam_log_var = output[:, output.shape[-1] // 2:]
        return lam_mean, lam_log_var

class ZPriorDisc(nn.Module):
    def __init__(self, dim_z, num_u):
        """num_u: number of different labels u"""
        super(ZPriorDisc, self).__init__()
        self.embedding = nn.Embedding(num_u, dim_z)
    
    def forward(self, u_input):
        lam_mean = squeeze_func(self.embedding(u_input).squeeze(1))
        lam_log_var = squeeze_func(self.embedding(u_input).squeeze(1))
        return lam_mean, lam_log_var

class VAE(nn.Module):
    def __init__(self, dim_x, dim_z, dim_u, gen_nodes, n_blk=None, min_gen_nodes_decoder_nflow=30, mdl='poisson', disc=True, learning_rate=5e-4):
        super(VAE, self).__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.mdl = mdl
        self.disc = disc
        
        if disc:
            self.z_prior = ZPriorDisc(dim_z, dim_u)
        else:
            self.z_prior = ZPriorNN(dim_z, dim_u)
        
        self.encoder = EncodeFunc(dim_x, dim_z, gen_nodes)
        self.decoder = DecodeFunc(dim_x, dim_z, gen_nodes, mdl) if n_blk is None else DecodeNFlowFunc(dim_x, dim_z, n_blk, mdl, min_gen_nodes_decoder_nflow)
        self.obs = nn.Linear(1, self.dim_x, bias=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, x_input, u_input):
        lam_mean, lam_log_var = self.z_prior(u_input)
        z_mean = self.encoder(x_input)
        z_log_var = self.encoder(x_input)
        post_mean, post_log_var = compute_posterior(z_mean, z_log_var, lam_mean, lam_log_var)
        z_sample = sampling(post_mean, post_log_var)
        fire_rate = self.decoder(z_sample)
        
        if self.mdl == 'gaussian':
            one_tensor = torch.ones((1, 1), device=x_input.device)
            obs_log_var = self.obs(one_tensor)
            vae_outputs = (post_mean, post_log_var, z_sample, fire_rate, lam_mean, lam_log_var, z_mean, z_log_var, obs_log_var)
        else:
            vae_outputs = (post_mean, post_log_var, z_sample, fire_rate, lam_mean, lam_log_var, z_mean, z_log_var)
        
        return vae_outputs
    
    def loss_function(self, x_input, vae_outputs):
        post_mean, post_log_var, z_sample, fire_rate, lam_mean, lam_log_var, z_mean, z_log_var, *extra = vae_outputs
        
        if self.mdl == 'poisson':
            obs_loglik = torch.sum(fire_rate - x_input * torch.log(fire_rate), dim=-1)
        elif self.mdl == 'gaussian':
            obs_log_var = extra[0]
            obs_loglik = torch.sum((fire_rate - x_input)**2 / (2 * torch.exp(obs_log_var)) + (obs_log_var / 2), dim=-1)
        
        kl_loss = 1 + post_log_var - lam_log_var - ((torch.square(post_mean - lam_mean) + torch.exp(post_log_var)) / torch.exp(lam_log_var))
        kl_loss = torch.sum(kl_loss, dim=-1)
        kl_loss *= -0.5
        
        vae_loss = torch.mean(obs_loglik + kl_loss)
        return vae_loss
    
    def train_step(self, x_input, u_input):
        self.optimizer.zero_grad()
        vae_outputs = self.forward(x_input, u_input)
        loss = self.loss_function(x_input, vae_outputs)
        loss.backward()
        self.optimizer.step()
        return loss.item()
