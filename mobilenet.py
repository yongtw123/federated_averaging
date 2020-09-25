from torch import nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None, relu_layer=None):
        padding = (kernel_size - 1) // 2
        layers = [nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False)]

        if norm_layer is not None:
            layers.append(norm_layer(out_planes))

        if relu_layer is not None:
            layers.append(relu_layer(inplace=True))
        
        super(ConvBNReLU, self).__init__(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None, quant_friendly=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        relu_layer = nn.ReLU

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, relu_layer=relu_layer))
        # dw
        if quant_friendly:
            layers.append(ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim))
        else:
            layers.append(ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer, relu_layer=relu_layer))
        
        layers.extend([
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
				 in_channels=3,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None,
                 quant_friendly=False):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        relu_layer = nn.ReLU

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(in_channels, input_channel, stride=2, norm_layer=None if quant_friendly else norm_layer, relu_layer=relu_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, quant_friendly=quant_friendly))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, relu_layer=relu_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class Conv_dw_quant(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, use_relu6=True):
        super().__init__()
        
        relu_layer = nn.ReLU6 if use_relu6 else nn.ReLU
        layers = [
            nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding, bias=False, groups=in_channel),
            ConvBNReLU(in_channel, out_channel, kernel_size=1, stride=1,
                       norm_layer=nn.BatchNorm2d, relu_layer=relu_layer)
        ]

        self.model = nn.Sequential(*layers)
    
    def forward(self, input):
        return self.model(input)

class Conv_dw_Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, use_relu6=True):
        super().__init__()
        
        relu_layer = nn.ReLU6 if use_relu6 else nn.ReLU
        self.layers = [
            nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding, bias=False, groups=in_channel),
            nn.BatchNorm2d(in_channel),
            relu_layer(inplace=True),
            ConvBNReLU(in_channel, out_channel, kernel_size=1, stride=1,
                       norm_layer=nn.BatchNorm2d, relu_layer=relu_layer)
        ]

        self.model = nn.Sequential(*self.layers)
    
    def forward(self, input):
        return self.model(input)


class MobileNetV1(nn.Module):
    def __init__(self, num_classes, in_channels=3, use_relu6=True, quant_friendly=False):
        super().__init__()
        
        self.num_classes = num_classes
        Block = Conv_dw_quant if quant_friendly else Conv_dw_Conv
        
        
        self.model = nn.Sequential(
            ConvBNReLU(in_channels, 32, stride=2, 
                       norm_layer=None if quant_friendly else nn.BatchNorm2d,
                       relu_layer=nn.ReLU6 if use_relu6 else nn.ReLU),
            Block(32, 64, kernel_size=3, stride=1, use_relu6=use_relu6),
            Block(64, 128, kernel_size=3, stride=2, use_relu6=use_relu6),
            Block(128, 128, kernel_size=3, stride=1, use_relu6=use_relu6),
            Block(128, 256, kernel_size=3, stride=2, use_relu6=use_relu6),
            Block(256, 256, kernel_size=3, stride=1, use_relu6=use_relu6),
            Block(256, 512, kernel_size=3, stride=2, use_relu6=use_relu6),

            Block(512, 512, kernel_size=3, stride=1, use_relu6=use_relu6),
            Block(512, 512, kernel_size=3, stride=1, use_relu6=use_relu6),
            Block(512, 512, kernel_size=3, stride=1, use_relu6=use_relu6),
            Block(512, 512, kernel_size=3, stride=1, use_relu6=use_relu6),
            Block(512, 512, kernel_size=3, stride=1, use_relu6=use_relu6),

            Block(512, 1024, kernel_size=3, stride=2, use_relu6=use_relu6),
            Block(1024, 1024, kernel_size=3, stride=1, use_relu6=use_relu6)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)
        
    def forward(self, input):
        x = self.model(input)
        x = self.avg_pool(x)
        x = x.view(-1, 1024)
        out = self.fc(x)
        return out


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def mobilenet_v2q(**kwargs):
    return MobileNetV2(quant_friendly=True, **kwargs)


def mobilenet_v1(**kwargs):
    return MobileNetV1(**kwargs)


def mobilenet_v1q(**kwargs):
    return MobileNetV1(use_relu6=False, quant_friendly=True, **kwargs)