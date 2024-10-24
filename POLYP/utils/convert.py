import torch
import torch.nn as nn


class AdaBN(nn.BatchNorm2d):
    def __init__(self, in_ch):
        super(AdaBN, self).__init__(in_ch)
        self.g_syn_mu = torch.zeros((1, in_ch, 1, 1))
        self.g_syn_var = torch.zeros((1, in_ch, 1, 1))
        self.sample_num = 0

    def update_stats(self, src_mu, src_var, cur_mu, cur_var):
        syn_mu = 0.5 * src_mu + 0.5 * cur_mu
        syn_var = 0.5 * src_var + 0.5 * cur_var

        self.g_syn_mu = (self.g_syn_mu * self.sample_num + syn_mu) / (self.sample_num + 1)
        self.g_syn_var = (self.g_syn_var * self.sample_num + syn_var) / (self.sample_num + 1)
        self.sample_num += 1

    def get_alpha(self, src_mu, src_var, cur_mu, cur_var):
        d_cur_syn = (cur_mu - self.g_syn_mu).abs() + (cur_var - self.g_syn_var).abs()
        d_cur_src = (cur_mu - src_mu).abs() + (cur_var - src_var).abs()

        weight = torch.cat((d_cur_src, d_cur_syn), dim=0).squeeze().softmax(0)
        return 1.0 - weight[0], (1.0 - weight[1]) / 2, (1.0 - weight[1]) / 2

    def get_mu_var(self, x):
        C = x.shape[1]

        cur_mu = x.mean((0, 2, 3), keepdims=True).detach()
        cur_var = x.var((0, 2, 3), keepdims=True).detach()
        src_mu = self.running_mean.view(1, C, 1, 1)
        src_var = self.running_var.view(1, C, 1, 1)

        src_alpha, syn_alpha, cur_alpha = self.get_alpha(src_mu, src_var, cur_mu, cur_var)
        src_alpha, syn_alpha, cur_alpha = src_alpha.view(1, C, 1, 1), syn_alpha.view(1, C, 1, 1), cur_alpha.view(1, C,
                                                                                                                 1, 1)
        self.update_stats(src_mu, src_var, cur_mu, cur_var)

        new_mu = src_alpha * src_mu + syn_alpha * self.g_syn_mu + cur_alpha * cur_mu
        new_var = src_alpha * src_var + syn_alpha * self.g_syn_var + cur_alpha * cur_var
        return new_mu, new_var

    def forward(self, x):
        C = x.shape[1]
        self.g_syn_mu = self.g_syn_mu.to(x.device)
        self.g_syn_var = self.g_syn_var.to(x.device)

        new_mu, new_var = self.get_mu_var(x)

        new_sig = (new_var + self.eps).sqrt()
        new_x = ((x - new_mu) / new_sig) * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
        return new_x
