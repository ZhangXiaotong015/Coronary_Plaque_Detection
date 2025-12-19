import torch
import torch.nn as nn

class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret

class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


def Conv2CoordConv(backbone):
    model = backbone.body
    model.conv1 = CoordConv(7,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
    model.layer1[0].conv1 = CoordConv(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer1[0].conv2 = CoordConv(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.layer1[0].conv3 = CoordConv(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer1[1].conv1 = CoordConv(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer1[1].conv2 = CoordConv(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.layer1[1].conv3 = CoordConv(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer1[2].conv1 = CoordConv(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer1[2].conv2 = CoordConv(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.layer1[2].conv3 = CoordConv(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    model.layer2[0].conv1 = CoordConv(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer2[0].conv2 = CoordConv(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.layer2[0].conv3 = CoordConv(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer2[1].conv1 = CoordConv(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer2[1].conv2 = CoordConv(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.layer2[1].conv3 = CoordConv(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer2[2].conv1 = CoordConv(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer2[2].conv2 = CoordConv(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.layer2[2].conv3 = CoordConv(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer2[3].conv1 = CoordConv(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer2[3].conv2 = CoordConv(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.layer2[3].conv3 = CoordConv(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    model.layer3[0].conv1 = CoordConv(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer3[0].conv2 = CoordConv(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.layer3[0].conv3 = CoordConv(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer3[1].conv1 = CoordConv(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer3[1].conv2 = CoordConv(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.layer3[1].conv3 = CoordConv(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer3[2].conv1 = CoordConv(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer3[2].conv2 = CoordConv(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.layer3[2].conv3 = CoordConv(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer3[3].conv1 = CoordConv(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer3[3].conv2 = CoordConv(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.layer3[3].conv3 = CoordConv(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer3[4].conv1 = CoordConv(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer3[4].conv2 = CoordConv(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.layer3[4].conv3 = CoordConv(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer3[5].conv1 = CoordConv(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer3[5].conv2 = CoordConv(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.layer3[5].conv3 = CoordConv(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    model.layer4[0].conv1 = CoordConv(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer4[0].conv2 = CoordConv(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.layer4[0].conv3 = CoordConv(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer4[1].conv1 = CoordConv(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer4[1].conv2 = CoordConv(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.layer4[1].conv3 = CoordConv(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer4[2].conv1 = CoordConv(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer4[2].conv2 = CoordConv(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.layer4[2].conv3 = CoordConv(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)

    # for name, para in model.named_parameters():
    #     print(name)
    #     if 'conv' in name:
    #         key_list = name.split('.')[:-1]

    # for layer in model.named_modules():
    #     if isinstance(layer[1],nn.Conv2d):
            # print(layer)

    return backbone