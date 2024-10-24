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


def convert_encoder_to_target(net, norm, start=0, end=5, verbose=True, bottleneck=False, input_size=512, n=5):
    def convert_norm(old_norm, new_norm, num_features, idx, fea_size):
        norm_layer = new_norm(num_features).to(net.conv1.weight.device)
        if hasattr(norm_layer, 'load_old_dict'):
            info = 'Converted to : {}'.format(norm)
            norm_layer.load_old_dict(old_norm)
        elif hasattr(norm_layer, 'load_state_dict'):
            state_dict = old_norm.state_dict()
            info = norm_layer.load_state_dict(state_dict, strict=False)
        else:
            info = 'No load_old_dict() found!!!'
        if verbose:
            print(info)
        return norm_layer

    layers = [0, net.layer1, net.layer2, net.layer3, net.layer4]

    idx = 0
    for i, layer in enumerate(layers):
        if not (start <= i < end):
            continue
        if i == 0:
            net.bn1 = convert_norm(net.bn1, norm, net.bn1.num_features, idx, fea_size=input_size // 2)
            idx += 1
        else:
            down_sample = 2 ** (1 + i)

            for j, block in enumerate(layer):
                block.bn1 = convert_norm(block.bn1, norm, block.bn1.num_features, idx,
                                         fea_size=input_size // down_sample)
                idx += 1
                block.bn2 = convert_norm(block.bn2, norm, block.bn2.num_features, idx,
                                         fea_size=input_size // down_sample)
                idx += 1
                if bottleneck:
                    block.bn3 = convert_norm(block.bn3, norm, block.bn3.num_features, idx,
                                             fea_size=input_size // down_sample)
                    idx += 1
                if block.downsample is not None:
                    block.downsample[1] = convert_norm(block.downsample[1], norm, block.downsample[1].num_features, idx,
                                                       fea_size=input_size // down_sample)
                    idx += 1
    return net


def convert_decoder_to_target(net, norm, start=0, end=5, verbose=True, input_size=512, n=5):
    def convert_norm(old_norm, new_norm, num_features, idx, fea_size):
        norm_layer = new_norm(num_features).to(old_norm.weight.device)
        if hasattr(norm_layer, 'load_old_dict'):
            info = 'Converted to : {}'.format(norm)
            norm_layer.load_old_dict(old_norm)
        elif hasattr(norm_layer, 'load_state_dict'):
            state_dict = old_norm.state_dict()
            info = norm_layer.load_state_dict(state_dict, strict=False)
        else:
            info = 'No load_old_dict() found!!!'
        if verbose:
            print(info)
        return norm_layer

    layers = [net[0], net[1], net[2], net[3], net[4]]

    idx = 0
    for i, layer in enumerate(layers):
        if not (start <= i < end):
            continue
        if i == 4:
            net[4] = convert_norm(layer, norm, layer.num_features, idx, input_size)
            idx += 1
        else:
            down_sample = 2 ** (4 - i)
            layer.bn = convert_norm(layer.bn, norm, layer.bn.num_features, idx, input_size // down_sample)
            idx += 1
    return net
