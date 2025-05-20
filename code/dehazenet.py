
from  transfer import Transfernet
from transformer import CSWin

from MIRNet import *

class netDC(nn.Module):
    def __init__(self):
        super(netDC, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, 1, 0, bias=False),
        )
        self.fc = nn.Linear(16, 8)#(144,4)#(16, 8)#(144,1)

    def forward(self, input):
        output = self.layer1(input)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output.view(-1)

class UNet_emb(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3,bias=False):
        """Initializes U-Net."""

        super(UNet_emb, self).__init__()
        self.embedding_dim = 3

        self.conv1 = nn.Conv2d(128*3, 128, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64*3, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32*3, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.in_chans = 3
        self.ReLU=nn.ReLU(inplace=True)

        self.PPM1 = PPM(16, 4, bins=(1,2,3,4))
        self.PPM2 = PPM(32, 8, bins=(1,2,3,4))
        self.PPM3 = PPM(64, 16, bins=(1,2,3,4))
        # self.PPM4 = PPM(128, 32, bins=(1,2,3,4))


        self.MSRB2=MSRB(64, 3, 1, 2,bias)
        self.MSRB3=MSRB(32, 3, 1, 2,bias)
        self.MSRB4=MSRB(16, 3, 1, 2,bias)

        self.swin_1 =CSWin(patch_size=2, embed_dim=32, depth=[4, 4, 4, 4],
              split_size=[1, 3, 5, 7], num_heads=[8, 16, 32, 32], mlp_ratio=4., qkv_bias=True, qk_scale=None,
              drop_rate=0., attn_drop_rate=0.,
              drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, use_chk=False)


        self.E_block1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        )

        self.E_block2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False))

        self.E_block3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False))
        
        self.E_block4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True))#


        self._block3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2))
        
        self._block4 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2))
        
        self._block5= nn.Sequential(
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2))


        self._block7= nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(16, out_channels, 3, stride=1, padding=1))

        self.convS1= nn.Conv2d(128 * 2, 128, 3, stride=1, padding=1)
        self.transformerlayer_coarse = Transfernet(d_model=128, nhead=4, dim_feedforward=256)#TransformerBlock(dim=128,  num_heads=1, ffn_expansion_factor=2.66, bias=False,  LayerNorm_type='WithBias') #

        self.task_query4 = nn.Parameter(torch.randn(1, 128, 32, 32))


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def Encoder(self, x):
        """Through encoder, then decoder by adding U-skip connections. """
        swin_in = x  # 96,192,384,768
        content = []

        swin_input_1 = self.E_block1(swin_in)  # 32
        swin_input_1 = self.PPM1(swin_input_1)
        swin_out_1 = self.swin_1(swin_in, swin_input_1)  #
        content.append(torch.cat((  swin_input_1, swin_out_1[0]), dim=1))

        swin_input_2 = self.E_block2(swin_input_1)  # 64
        swin_input_2 = self.PPM2(swin_input_2)
        content.append(torch.cat((swin_input_2, swin_out_1[1]), dim=1))

        swin_input_3 = self.E_block3(swin_input_2)  # 128
        swin_input_3 = self.PPM3(swin_input_3)
        content.append(torch.cat((swin_input_3, swin_out_1[2]), dim=1))

        swin_input_4 = self.E_block4(swin_input_3)  # 128
        style= torch.cat((swin_input_4,swin_out_1[3]),dim=1)
        style = self.convS1(style)


        return style,content

    def Decoder(self, style,content,s=1):
        batch, channel3, h3, w3 = style.size(0), style.size(1), style.size(2), style.size(3)
        if s==1 :
            task_q4 = style
            task_qb4 = task_q4
        else:
            task_q4 = self.task_query4
            task_qb4 = task_q4.repeat(batch, 1, 1, 1)  # self.convd(cat(content[4],self.task_query4),1)#

        task_q4 = F.interpolate(task_qb4 , size=(h3, w3), mode='bilinear', align_corners=True)

        swin_out_12 = (task_q4).flatten(2).permute(2, 0, 1)#self.conv1_1
        swin_input_4=style  #(swin_out_1[3])#self.conv1_1
        swin_input_44 =swin_input_4.flatten(2).permute(2, 0, 1)
        swin_input_s = self.transformerlayer_coarse(swin_out_12, swin_input_44)# query, key/value
        swin_input_s= swin_input_s.view(batch, channel3, h3, w3)

        # swin_input_3 =  self.conv1_1 (content[2])
        concat3 = torch.cat((  content[2], swin_input_s), dim=1)#content[2],  # 256+256+256==512
        decoder_3 = self.ReLU(self.conv1(concat3))  # 256
        upsample3 = self._block3(decoder_3)  # 128
        upsample3 = self.MSRB2(upsample3)

        # swin_input_2 = self.conv2_1(content[1])#self.conv2_1
        concat2 = torch.cat((content[1], upsample3), dim=1)#content[1],  # 128+128+128=256
        decoder_2 = self.ReLU(self.conv2(concat2))  # 128
        upsample4 = self._block4(decoder_2)  # 64
        upsample4 = self.MSRB3(upsample4)

        # swin_input_1 =self.conv3_1(content[0])# self.conv3_1
        concat1 = torch.cat(( content[0], upsample4), dim=1) #content[0], # 64+64+64=128
        decoder_1 = self.ReLU(self.conv3(concat1))  # 64
        upsample5 = self._block5(decoder_1)  # 32
        upsample5 = self.MSRB4(upsample5)

        decoder_0 = self.ReLU(self.conv4(upsample5 ))  # 48

        result = self._block7(decoder_0)  # 23

        return result,task_q4



           
class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)


    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))   #what is f(x)
        return torch.cat(out, 1)         
