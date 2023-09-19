# copied and modified from https://idiomaticprogrammers.com/post/unet-architecture/

from torch import nn
import torch
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, strides, padding):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernal_size, strides, padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernal_size, strides, padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, strides, padding):
        super(ResidualConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernal_size, strides, padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernal_size, strides, padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return x + self.conv(x)


# Residual UNet
class RUNet(nn.Module):
    def __init__(self, in_channels, num_segmentations=1, features=[32, 64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        conv_bottom = ResidualConv(
            in_channels=features[-1],
            out_channels=features[-1],
            kernal_size=3,
            strides=1,
            padding=1
        )
        conv_out = DoubleConv(
            in_channels=features[-1],
            out_channels=features[-1] * 2,
            kernal_size=3,
            strides=1,
            padding=1
        )
        self.bottleneck = nn.Sequential(
            conv_bottom,
            conv_bottom,
            conv_bottom,
            conv_bottom,
            conv_out
        )

        self.output = nn.Conv2d(
            in_channels=features[0],
            out_channels=num_segmentations,
            kernel_size=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_channels_iter = in_channels
        for feature in features:
            self.downs.append(DoubleConv(
                in_channels=in_channels_iter,
                out_channels=feature,
                kernal_size=3,
                strides=1,
                padding=1
            ))
            in_channels_iter = feature

        for feature in reversed(features):
            up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2,
                    padding=0
                ),
                DoubleConv(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernal_size=3,
                    padding=1,
                    strides=1
                )
            )

            self.ups.append(up)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(len(self.ups)):
            x = self.ups[i][0](x)  # Pass through ConvTranspose first

            skip_connection = skip_connections[i]

            # If the height and width of output tensor and skip connection
            # is not same then resize the tensor
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            # Concat the output tensor with skip connection
            concat_x = torch.cat((skip_connection, x), dim=1)

            # Pass the concatinated tensor through DoubleCOnv
            x = self.ups[i][1](concat_x)

        return self.output(x)


class UNet(nn.Module):
    def __init__(
            self, 
            in_channels,
            num_segmentations=1, 
            features=[64, 128, 256, 512], 
            extras=""
        ):
        
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.dwt_channel = None
        self.in_channels = in_channels

        if 'I' in extras:
            extra_channels = [0, 3, 3, 3]
            self.dwt_channel = extras.find('I')
        else:
            extra_channels = [0, 0, 0, 0]

        self.bottleneck = DoubleConv(
            in_channels=features[-1],
            out_channels=features[-1] * 2,
            kernal_size=3,
            strides=1,
            padding=1
        )

        self.output = nn.Conv2d(
            in_channels=features[0],
            out_channels=num_segmentations,
            kernel_size=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_channels_iter = in_channels
        for feature, extra in zip(features, extra_channels):
            self.downs.append(DoubleConv(
                in_channels=in_channels_iter + extra,
                out_channels=feature,
                kernal_size=3,
                strides=1,
                padding=1
            ))
            in_channels_iter = feature

        for feature in reversed(features):
            up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2,
                    padding=0
                ),
                DoubleConv(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernal_size=3,
                    padding=1,
                    strides=1
                )
            )

            self.ups.append(up)

    def nan_assert(x):
        assert not torch.isnan(grad).any(), ""

    def forward(self, x):
        using_dwt_extractor = self.dwt_channel is not None
        # Prepare extractors
        if using_dwt_extractor:
            dwt = x[:, self.dwt_channel:self.dwt_channel+1]
            dwt_extras = self.__parse_dwt(dwt)

        # Get main channels
        x = x[:, :self.in_channels]
        
        skip_connections = []
        for down in self.downs:
            if using_dwt_extractor:
                e = dwt_extras.pop(0)
                if e is not None:
                    x = torch.cat((x, e), dim=1)
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(len(self.ups)):
            x = self.ups[i][0](x)  # Pass through ConvTranspose first

            skip_connection = skip_connections[i]

            # If the height and width of output tensor and skip connection
            # is not same then resize the tensor
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            # Concat the output tensor with skip connection
            concat_x = torch.cat((skip_connection, x), dim=1)

            # Pass the concatinated tensor through DoubleCOnv
            x = self.ups[i][1](concat_x)

        return self.output(x)

    @staticmethod
    def __parse_dwt(channel, level=3):
        h = channel.shape[2]
        layers = [None]
        for i in range(level):
            h = h // 2
            LH = TF.crop(channel, 0, h, h, h)
            HL = TF.crop(channel, h, 0, h, h)
            HH = TF.crop(channel, h, h, h, h)
            layer = torch.cat([LH, HL, HH], dim=1)
            layers.append(layer)
        return layers


def init_weights(model, init_func, *params, **kwargs):
    for p in model.parameters():
        init_func(p, *params, **kwargs)

def test():
    image = torch.randn((32, 3, 161, 161))
    model = UNet(in_channels=3)
    out = model(image)
    print(image.shape, out.shape)
    assert out.shape == (32, 1, 161, 161)


if __name__ == "__main__":
    test()