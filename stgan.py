# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
#
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""AttGAN, generator, and discriminator."""

import torch
import torch.nn as nn
from nn import LinearBlock, Conv2dBlock, ConvTranspose2dBlock
from torchsummary import summary


# This architecture is for images of 128x128
# In the original AttGAN, slim.conv2d uses padding 'same'
MAX_DIM = 64 * 16  # 1024


def _concat(x, attr):
    n, state_dim, h, w = x.size()
    att_dim = attr.size()[1]
    attr = attr.view((n, att_dim, 1, 1)).expand((n, att_dim, h, w))
    return torch.cat([x, attr], 1)


class ConvGRUCell(nn.Module):
    def __init__(self, n_attrs, in_dim, out_dim, kernel_size=3):
        super(ConvGRUCell, self).__init__()
        self.n_attrs = n_attrs
        self.upsample = nn.ConvTranspose2d(
            in_dim * 2 + n_attrs, out_dim, 4, 2, 1, bias=False)
        self.reset_gate = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size,
                      1, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size,
                      1, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid()
        )
        self.hidden = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size,
                      1, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Tanh()
        )

    def forward(self, input, old_state, attr):
        n, _, h, w = old_state.size()
        attr = attr.view((n, self.n_attrs, 1, 1)).expand(
            (n, self.n_attrs, h, w))
        state_hat = self.upsample(torch.cat([old_state, attr], 1))
        r = self.reset_gate(torch.cat([input, state_hat], dim=1))
        z = self.update_gate(torch.cat([input, state_hat], dim=1))
        new_state = r * state_hat
        hidden_info = self.hidden(torch.cat([input, new_state], dim=1))
        output = (1 - z) * state_hat + z * hidden_info
        return output, new_state


class Generator(nn.Module):
    def __init__(self, attr_dim, conv_dim=64, n_layers=5, shortcut_layers=2,
                 stu_kernel_size=3, use_stu=True, one_more_conv=True):
        super(Generator, self).__init__()
        self.n_attrs = attr_dim
        self.n_layers = n_layers
        self.shortcut_layers = min(shortcut_layers, n_layers - 1)
        self.use_stu = use_stu

        self.encoder = nn.ModuleList()
        in_channels = 3
        for i in range(self.n_layers):
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_channels, conv_dim * 2 ** i, 4, 2, 1, bias=False),
                nn.BatchNorm2d(conv_dim * 2 ** i),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ))
            in_channels = conv_dim * 2 ** i

        if use_stu:
            self.stu = nn.ModuleList()
            for i in reversed(range(self.n_layers - 1 - self.shortcut_layers,
                                    self.n_layers - 1)):
                self.stu.append(ConvGRUCell(
                    self.n_attrs,
                    conv_dim * 2 ** i,
                    conv_dim * 2 ** i,
                    stu_kernel_size))

        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            if i < self.n_layers - 1:
                if i == 0:
                    self.decoder.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            conv_dim * 2 ** (self.n_layers - 1) + attr_dim,
                            conv_dim * 2 ** (self.n_layers - 1), 4, 2, 1,
                            bias=False),
                        nn.BatchNorm2d(in_channels),
                        nn.ReLU(inplace=True)
                    ))
                elif i <= self.shortcut_layers:     # not <
                    self.decoder.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            conv_dim * 3 * 2 ** (
                                self.n_layers - 1 - i) + attr_dim,
                            conv_dim * 2 ** (self.n_layers - 1 - i), 4, 2, 1,
                            bias=False),
                        nn.BatchNorm2d(
                            conv_dim * 2 ** (self.n_layers - 1 - i)),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    self.decoder.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            conv_dim * 2 ** (self.n_layers - i),
                            conv_dim * 2 ** (self.n_layers - 1 - i), 4, 2, 1,
                            bias=False),
                        nn.BatchNorm2d(
                            conv_dim * 2 ** (self.n_layers - 1 - i)),
                        nn.ReLU(inplace=True)
                    ))
            else:
                in_dim = conv_dim * 3 + attr_dim if self.shortcut_layers == \
                    self.n_layers - 1 else conv_dim * 2
                if one_more_conv:
                    self.decoder.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            in_dim, conv_dim // 4, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(conv_dim // 4),
                        nn.ReLU(inplace=True),

                        nn.ConvTranspose2d(
                            conv_dim // 4, 3, 3, 1, 1, bias=False),
                        nn.Tanh()
                    ))
                else:
                    self.decoder.append(nn.Sequential(
                        nn.ConvTranspose2d(in_dim, 3, 4, 2, 1, bias=False),
                        nn.Tanh()
                    ))

    def encode(self, x):
        z = x
        zs = []
        for layer in self.encoder:
            z = layer(z)
            zs.append(z)
        return zs

    def decode(self, zs, a):
        out = zs[-1]
        n, _, h, w = out.size()
        attr = a.view((n, self.n_attrs, 1, 1)).expand((n, self.n_attrs, h, w))
        out = self.decoder[0](torch.cat([out, attr], dim=1))
        stu_state = zs[-1]

        # propagate shortcut layers
        for i in range(1, self.shortcut_layers + 1):
            if self.use_stu:
                stu_out, stu_state = self.stu[
                    i - 1](zs[-(i + 1)], stu_state, a)
                out = torch.cat([out, stu_out], dim=1)
                out = _concat(out, a)  # inject layers
                out = self.decoder[i](out)
            else:
                out = torch.cat([out, zs[-(i + 1)]], dim=1)
                out = _concat(out, a)  # inject layers
                out = self.decoder[i](out)

        # propagate non-shortcut layers
        for i in range(self.shortcut_layers + 1, self.n_layers):
            out = self.decoder[i](out)

        return out

    def forward(self, x, a=None, mode='enc-dec'):
        if mode == 'enc-dec':
            assert a is not None, 'No given attribute.'
            return self.decode(self.encode(x), a)
        if mode == 'enc':
            return self.encode(x)
        if mode == 'dec':
            assert a is not None, 'No given attribute.'
            return self.decode(x, a)
        raise Exception('Unrecognized mode: ' + mode)


class Discriminator(nn.Module):
    def __init__(self, image_size=128, attr_dim=10, conv_dim=64, fc_dim=1024,
                 n_layers=5):
        super(Discriminator, self).__init__()
        layers = []
        in_channels = 3
        for i in range(n_layers):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, conv_dim * 2 ** i, 4, 2, 1),
                nn.InstanceNorm2d(conv_dim * 2 ** i,
                                  affine=True, track_running_stats=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ))
            in_channels = conv_dim * 2 ** i
        self.conv = nn.Sequential(*layers)
        feature_size = image_size // 2**n_layers
        self.fc_adv = nn.Sequential(
            nn.Linear(conv_dim * 2 ** (n_layers - 1) * feature_size ** 2,
                      fc_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(fc_dim, 1)
        )
        self.fc_att = nn.Sequential(
            nn.Linear(conv_dim * 2 ** (n_layers - 1) * feature_size ** 2,
                      fc_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(fc_dim, attr_dim),
        )

    def forward(self, x):
        y = self.conv(x)
        y = y.view(y.size()[0], -1)
        logit_adv = self.fc_adv(y)
        logit_att = self.fc_att(y)
        return logit_adv, logit_att


import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim


class STGAN():
    def __init__(self, args):
        self.mode = args.mode
        self.gpu = args.gpu
        self.multi_gpu = args.multi_gpu if 'multi_gpu' in args else False
        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2
        self.lambda_3 = args.lambda_3
        self.lambda_gp = args.lambda_gp

        self.G = Generator(args.n_attrs,
                           conv_dim=args.conv_dim,
                           n_layers=args.n_layers,
                           shortcut_layers=args.shortcut_layers,
                           stu_kernel_size=args.stu_kernel_size,
                           use_stu=args.use_stu,
                           one_more_conv=args.one_more_conv)
        self.G.train()
        if self.gpu:
            self.G.cuda()
        summary(self.G, [(3, args.img_size, args.img_size),
                         (args.n_attrs,)], batch_size=4, use_gpu=self.gpu)

        self.D = Discriminator(image_size=args.img_size,
                               attr_dim=len(args.attrs),
                               conv_dim=args.conv_dim,
                               fc_dim=args.dis_fc_dim,
                               n_layers=args.dis_layers)
        self.D.train()
        if self.gpu:
            self.D.cuda()
        summary(self.D, [(3, args.img_size, args.img_size)],
                batch_size=4, use_gpu=self.gpu)

        if self.multi_gpu:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        self.optim_G = optim.Adam(
            self.G.parameters(), lr=args.lr, betas=args.betas)
        self.optim_D = optim.Adam(
            self.D.parameters(), lr=args.lr, betas=args.betas)

    def set_lr(self, lr):
        for g in self.optim_G.param_groups:
            g['lr'] = lr
        for g in self.optim_D.param_groups:
            g['lr'] = lr

    def trainG(self, img_a, att_a, att_a_, att_b, att_b_):
        for p in self.D.parameters():
            p.requires_grad = False

        zs_a = self.G(img_a, mode='enc')
        img_fake = self.G(zs_a, att_b_, mode='dec')
        img_recon = self.G(zs_a, att_a_, mode='dec')
        d_fake, dc_fake = self.D(img_fake)

        if self.mode == 'wgan':
            gf_loss = -d_fake.mean()
        if self.mode == 'lsgan':  # mean_squared_error
            gf_loss = F.mse_loss(d_fake, torch.ones_like(d_fake))
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            gf_loss = F.binary_cross_entropy_with_logits(
                d_fake, torch.ones_like(d_fake))
        gc_loss = F.binary_cross_entropy_with_logits(dc_fake, att_b)
        gr_loss = F.l1_loss(img_recon, img_a)
        g_loss = gf_loss + self.lambda_2 * gc_loss + self.lambda_1 * gr_loss

        self.optim_G.zero_grad()
        g_loss.backward()
        self.optim_G.step()

        errG = {
            'g_loss': g_loss.item(), 'gf_loss': gf_loss.item(),
            'gc_loss': gc_loss.item(), 'gr_loss': gr_loss.item()
        }
        return errG

    def trainD(self, img_a, att_a, att_a_, att_b, att_b_):
        for p in self.D.parameters():
            p.requires_grad = True

        img_fake = self.G(img_a, att_b_).detach()
        d_real, dc_real = self.D(img_a)
        d_fake, dc_fake = self.D(img_fake)

        def gradient_penalty(f, real, fake=None):
            def interpolate(a, b=None):
                if b is None:  # interpolation in DRAGAN
                    beta = torch.rand_like(a)
                    b = a + 0.5 * a.var().sqrt() * beta
                alpha = torch.rand(a.size(0), 1, 1, 1)
                alpha = alpha.cuda() if self.gpu else alpha
                inter = a + alpha * (b - a)
                return inter
            x = interpolate(real, fake).requires_grad_(True)
            pred = f(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            grad = autograd.grad(
                outputs=pred, inputs=x,
                grad_outputs=torch.ones_like(pred),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad = grad.view(grad.size(0), -1)
            norm = grad.norm(2, dim=1)
            gp = ((norm - 1.0) ** 2).mean()
            return gp

        if self.mode == 'wgan':
            wd = d_real.mean() - d_fake.mean()
            df_loss = -wd
            df_gp = gradient_penalty(self.D, img_a, img_fake)
        if self.mode == 'lsgan':  # mean_squared_error
            df_loss = F.mse_loss(d_real, torch.ones_like(d_fake)) + \
                F.mse_loss(d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img_a)
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            df_loss = F.binary_cross_entropy_with_logits(
                d_real, torch.ones_like(d_real)) + \
                F.binary_cross_entropy_with_logits(
                    d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img_a)
        dc_loss = F.binary_cross_entropy_with_logits(dc_real, att_a)
        d_loss = df_loss + self.lambda_gp * df_gp + self.lambda_3 * dc_loss

        self.optim_D.zero_grad()
        d_loss.backward(retain_graph=True)
        self.optim_D.step()

        errD = {
            'd_loss': d_loss.item(), 'df_loss': df_loss.item(),
            'df_gp': df_gp.item(), 'dc_loss': dc_loss.item()
        }
        return errD

    def train(self):
        self.G.train()
        self.D.train()

    def eval(self):
        self.G.eval()
        self.D.eval()

    def save(self, path):
        states = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'optim_D': self.optim_D.state_dict()
        }
        torch.save(states, path)

    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.G.load_state_dict(states['G'])
        if 'D' in states:
            self.D.load_state_dict(states['D'])
        if 'optim_G' in states:
            self.optim_G.load_state_dict(states['optim_G'])
        if 'optim_D' in states:
            self.optim_D.load_state_dict(states['optim_D'])

    def saveG(self, path):
        states = {
            'G': self.G.state_dict()
        }
        torch.save(states, path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', dest='img_size', type=int, default=128)
    parser.add_argument('--shortcut_layers',
                        dest='shortcut_layers', type=int, default=1)
    parser.add_argument('--inject_layers',
                        dest='inject_layers', type=int, default=0)
    parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
    parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
    parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64)
    parser.add_argument('--dis_fc_dim', dest='dis_fc_dim',
                        type=int, default=1024)
    parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=5)
    parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=5)
    parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5)
    parser.add_argument('--enc_norm', dest='enc_norm',
                        type=str, default='batchnorm')
    parser.add_argument('--dec_norm', dest='dec_norm',
                        type=str, default='batchnorm')
    parser.add_argument('--dis_norm', dest='dis_norm',
                        type=str, default='instancenorm')
    parser.add_argument('--dis_fc_norm', dest='dis_fc_norm',
                        type=str, default='none')
    parser.add_argument('--enc_acti', dest='enc_acti',
                        type=str, default='lrelu')
    parser.add_argument('--dec_acti', dest='dec_acti',
                        type=str, default='relu')
    parser.add_argument('--dis_acti', dest='dis_acti',
                        type=str, default='lrelu')
    parser.add_argument('--dis_fc_acti', dest='dis_fc_acti',
                        type=str, default='relu')
    parser.add_argument('--mode', dest='mode', default='wgan',
                        choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--lr', dest='lr', type=float,
                        default=0.0002, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    args.n_attrs = 13
    args.betas = (args.beta1, args.beta2)
    arrgan = AttGAN(args)
