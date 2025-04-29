import torch.nn as nn
import torch

"""
Further improvements to do: make padding dynamic
"""

class EncoderBlock3D(nn.Module):

    def __init__(self, in_num_ch, out_num_ch, kernel_size=3, conv_act='leaky_relu', dropout=0, pooling=nn.AvgPool3d):
        super(EncoderBlock3D, self).__init__()
        if conv_act == 'relu':
            conv_act_layer = nn.ReLU(inplace=True)
        else:
            conv_act_layer = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Sequential(
                        nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1), #convolution
                        nn.BatchNorm3d(out_num_ch), #batch normalization
                        conv_act_layer, #activation function as above
                        nn.Dropout3d(dropout), #dropout, if specificied
                        pooling(2)) #pooling layer, can be average or max pooling
        self.init_model()

    def init_model(self): #weight initialization
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv3d):
                for name, weight in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_normal_(weight)
                    if 'bias' in name:
                        nn.init.constant_(weight, 0.0)

    def forward(self, x): #forqard pass acccoring to __init__ above
        return self.conv(x)
    


class Encoder3D(nn.Module):
    """
    Building multiple convolutional blocks together for feature extraction.
    """
    def __init__(self, in_num_ch=1, num_block=4, inter_num_ch=16, kernel_size=3, conv_act='leaky_relu', dropout = 0,  pooling=nn.AvgPool3d): #add dropout argument vs original LILAC
        """
        inter_num_ch: base number of output channels for the first block.
        """
        
        super(Encoder3D, self).__init__()

        conv_blocks = []
        for i in range(num_block):
            if i == 0: # initial block
                conv_blocks.append(EncoderBlock3D(in_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=dropout,  pooling=pooling))
            elif i == (num_block-1): # last block: compress features back to inter_num_ch as bottleneck
                conv_blocks.append(EncoderBlock3D(inter_num_ch * (2 ** (i - 1)), inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=dropout,  pooling=pooling))
            else:
                conv_blocks.append(EncoderBlock3D(inter_num_ch * (2 ** (i - 1)), inter_num_ch * (2 ** (i)), kernel_size=kernel_size, conv_act=conv_act, dropout=dropout,  pooling=pooling))

        self.conv_blocks = nn.Sequential(*conv_blocks)


    def forward(self, x): #feed forward through all convolutional blocks

        for cb in self.conv_blocks:
            x = cb(x)

        return x
    


class LILAC(nn.Module):
    def __init__(self, args):
        """
        Args:
            image_size: size of the input image
            channels: number of channels in the input image
            n_of_blocks: number of convolutional blocks
            initial_channel: number of channels in the first convolutional block
            dropout: dropout rate
            pooling: pooling layer
            optional_meta: number of additional features
        """
        super().__init__()
        self.optional_meta_dim = len(args.optional_meta) if args.optional_meta else 0 #number of additional features (if any)
        self.feature_image = torch.tensor(args.image_size) // (2 ** args.n_of_blocks) #size of each feature map after all convolutional blocks
        self.feature_channel = args.initial_channel #number of channels after all convolutional blocks
        self.encoder = Encoder3D(in_num_ch=args.channels, num_block=args.n_of_blocks, inter_num_ch=args.initial_channel, dropout=args.dropout, pooling=args.pooling) #CNN for feature extraction
        self.linear = nn.Linear(self.feature_channel * self.feature_image.prod().item() + int(self.optional_meta_dim), 1, bias=False) #full connected layer w/o bias

    def forward(self, x1, x2, meta = None): #feed-forward
        f1 = self.encoder(x1) #feature extraction for the first image
        f2 = self.encoder(x2) #feature extraction for the second image
        fd = f2 - f1 #feature difference
        fd = fd.view(x1.size(0), -1) #flatten the feature map
        if self.optional_meta_dim == 0:
            return self.linear(fd) #fully connected layer
        else:
            fd = torch.cat((fd, meta), dim=1)
            return self.linear(fd)