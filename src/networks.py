import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from .GlobalAttention import GlobalAttentionGeneral as ATT_NET

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)



class InpaintGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),
        )

        blocks = []
        for _ in range(residual_blocks):
            block = MMSFM(dim=256, ratio=2, kernel_size=[3,5,7], dilation=[2,2,2], use_spectral_norm=False)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.dc1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
        )
        self.dc2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
        )
        self.dc3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )
        self.ca_net = CA_NET()
        self.att = ATT_NET(64, 256)
        self.sent_fc = nn.Sequential(
            nn.Linear(in_features=100 * 1, out_features=256 * 16 * 16),
            nn.BatchNorm1d(256 * 16 * 16),
            nn.ReLU(True)
        )

        self.sent_up1 = nn.Sequential(  
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        self.sent_conv = nn.Sequential(
            nn.Conv2d(in_channels=64+64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.sent_up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True) 
        )

        self.fushion1 = MTIFB(in_channel=256, out_channel=256)
        self.fushion2 = MTIFB(in_channel=128, out_channel=128)
        if init_weights:
            self.init_weights()

    def forward(self, x, word_embs, sent_emb, text_mask):
        x = self.encoder(x)
        x = self.middle(x)
        sent_emb1, _, _ = self.ca_net(sent_emb)
        sent_emb1 = self.sent_fc(sent_emb1).view(-1, 256, 16, 16)
        sent_emb1 = self.sent_up1(sent_emb1)
        self.att.applyMask(text_mask)
        context_word, _ = self.att(sent_emb1, word_embs)
        sent_emb1 = self.sent_conv(torch.cat((sent_emb1, context_word), 1))
        x = self.dc1(self.fushion1(x, sent_emb1))
        sent_emb2 = self.sent_up2(sent_emb1)
        x = self.dc2(self.fushion2(x, sent_emb2))
        x = self.dc3(x)
        x = (torch.tanh(x) + 1) / 2

        return x

class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class MSFM(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=2, use_spectral_norm=False):
        super(MSFM, self).__init__()
        self.dim = dim
        self.conv_block_1 = nn.Sequential(    
            spectral_norm(nn.Conv2d(in_channels=dim//2, out_channels=dim//2, kernel_size=1, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim//2, track_running_stats=False),
            nn.ReLU(True),
        )

        self.conv_block_2_1 = nn.Sequential(
            nn.ReflectionPad2d((dilation*(kernel_size-1))//2),
            spectral_norm(nn.Conv2d(in_channels=dim//2, out_channels=dim//2, kernel_size=kernel_size, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim//2, track_running_stats=False),
            nn.ReLU(True),
        )
        self.conv_block_2_2 = nn.Sequential(    
            nn.ReflectionPad2d(kernel_size//2),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim//2, kernel_size=kernel_size, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim//2, track_running_stats=False),
            nn.ReLU(True),
        )
        self.conv_block_2_3 = nn.Sequential(    
            nn.ReflectionPad2d(kernel_size//2),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim//2, kernel_size=kernel_size, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim//2, track_running_stats=False),     
            nn.ReLU(True),       
        )

        self.confusion = nn.Sequential(    
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),           
        )

    def forward(self, x):
        input1 = x[:, :(self.dim//2), ...]
        input2 = x[:, (self.dim//2):, ...]
        output_1 = self.conv_block_1(input1)
        output_2_1 = self.conv_block_2_1(input2)
        output_2_2 = self.conv_block_2_2(torch.cat([output_2_1, output_1],1))
        output_2_3 = self.conv_block_2_3(torch.cat([output_2_2, output_1],1))
        output = self.confusion(torch.cat([output_2_3, output_1],1)) 

        return output

class MMSFM(nn.Module):
    def __init__(self, dim, ratio=2, kernel_size=[3,5], dilation=[2,2], use_spectral_norm=False):
        super(MMSFM, self).__init__()
        self.channel_reduce = nn.Sequential(    
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim//ratio, kernel_size=1, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim//ratio, track_running_stats=False),
            nn.ReLU(True),
        )
        self.branch_num = len(kernel_size)
        self.MSFM_blocks = nn.ModuleList([])
        for i in range(self.branch_num):
            MSFM_block = MSFM(dim//ratio, kernel_size[i], dilation[i], use_spectral_norm)
            self.MSFM_blocks.append(MSFM_block)

        self.channel_expand = nn.Sequential(    
            spectral_norm(nn.Conv2d(in_channels=dim//ratio*self.branch_num, out_channels=dim, kernel_size=1, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),           
        )
    
    def forward(self, x):
        x1 = self.channel_reduce(x)
        MSFM_outputs = []
        for i in range(self.branch_num):
            MSFM_output = self.MSFM_blocks[i](x1)
            MSFM_outputs.append(MSFM_output)
        x2 = self.channel_expand(torch.cat(MSFM_outputs, dim=1))
        out = x2 + x
        return out

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class MTIFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MTIFB, self).__init__()
        self.share = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.scale = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.offset = nn.Conv2d(in_channel, out_channel, 1, 1, 0)

    def forward(self, image_features, sentence_features):
        share_out = self.share(sentence_features)
        scale_mask = F.sigmoid(self.scale(share_out))
        offset_mask = self.offset(self.pool(share_out))
        out = image_features * scale_mask + offset_mask
        return out

class CA_NET(nn.Module):
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = 256
        self.c_dim = 100
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.relu = nn.ReLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar
    
    