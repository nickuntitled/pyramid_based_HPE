import torch
import torch.nn as nn
import math
import torch.nn.functional as F

import utils

if __name__ == '__main__':
    from efficientnet import EfficientNetV2, MBConvConfig, MBConv, get_efficientnet_v2_structure, load_from_zoo
else:
    from model.efficientnet import EfficientNetV2, MBConvConfig, MBConv, get_efficientnet_v2_structure, load_from_zoo

class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, out_channels = 256) -> None:
        super().__init__(
            # Replace ASPP with smaller
            ASPP(in_channels, [1,2], out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation) -> None:
        super().__init__()
        modules = []
        
        # Replace 1x1 conv instead of 3x3 conv when dilation = 1 to improve the accuracy of HPE
        if dilation > 1:
            modules.append(nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False))
        else:
            modules.append(nn.Conv2d(in_channels, out_channels, 1, bias=False))

        modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.SiLU())
        
        self.artous_convolution = nn.Sequential(*modules)
        
        # Channel Attention
        self.dilation = dilation

        k_size = 3
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # ---
        # self.adjust_module = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        # Use the below code to improve the accuracy of HPE 
        self.adjust_module = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # ----

        self.bypass_module = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Artous Attention
        out = self.artous_convolution(x)

        bypass = self.avgpool(x)
        bypass = self.adjust_module(bypass)
        bypass = self.bypass_module(bypass.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        bypass = self.sigmoid(bypass)

        # Combine with Channel Attention
        out = out * bypass.expand_as(out)
        return out

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.SiLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)

class Pyramid(nn.Module):
    def __init__(self, channel_arr):
        super().__init__()
        
        self.up1 = nn.Sequential(
            nn.Conv2d(channel_arr[-1], channel_arr[-1], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_arr[-1]),
            nn.SiLU(),
            nn.Conv2d(channel_arr[-1], channel_arr[-2], kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(channel_arr[-2]),
            nn.SiLU()
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(channel_arr[-2], channel_arr[-2], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_arr[-2]),
            nn.SiLU(),
            nn.Conv2d(channel_arr[-2], channel_arr[-3], kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(channel_arr[-3]),
            nn.SiLU()
        )
        
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(channel_arr[-3], channel_arr[-3], kernel_size = 2, stride = 2, padding = 0),
            nn.BatchNorm2d(channel_arr[-3]),
            nn.SiLU()
        )
        
        self.up3 = nn.Sequential(
            nn.Conv2d(channel_arr[-3], channel_arr[-3], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_arr[-3]),
            nn.SiLU(),
            nn.Conv2d(channel_arr[-3], channel_arr[-4], kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(channel_arr[-4]),
            nn.SiLU()
        )
        
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(channel_arr[-4], channel_arr[-4], kernel_size = 2, stride = 2, padding = 0),
            nn.BatchNorm2d(channel_arr[-4]),
            nn.SiLU()
        )
        
        self.down0 = nn.Sequential(
            nn.Conv2d(channel_arr[-4], channel_arr[-4], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_arr[-4]),
            nn.SiLU(),
            nn.Conv2d(channel_arr[-4], channel_arr[-4], kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(channel_arr[-4]),
            nn.SiLU()
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(channel_arr[-4], channel_arr[-3], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_arr[-3]),
            nn.SiLU(),
            nn.Conv2d(channel_arr[-3], channel_arr[-3], kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(channel_arr[-3]),
            nn.SiLU()
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(channel_arr[-3], channel_arr[-2], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_arr[-2]),
            nn.SiLU(),
            nn.Conv2d(channel_arr[-2], channel_arr[-2], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_arr[-2]),
            nn.SiLU()
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(channel_arr[-2], channel_arr[-1], kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(channel_arr[-1]),
            nn.SiLU(),
            nn.Conv2d(channel_arr[-1], channel_arr[-1], kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(channel_arr[-1]),
            nn.SiLU()
        )

    def forward(self, arr_input):
        input1 = arr_input[3]
        input2 = arr_input[2]
        input3 = arr_input[1]
        input4 = arr_input[0]

        output1 = output_int1 = self.up1(input1)           

        output2 = output_int2 = self.up2(output1 + input2)         
        output2 = self.upsample2(output2)

        output3 = output_int3 = self.up3(output2 + input3)         
        output3 = self.upsample3(output3)

        output = self.down0(output3 + input4)                   
        output = self.down1(output + output_int3)                      
        output = self.down2(output + output_int2)                     
        output = self.down3(output + output_int1)

        return output

class Model(EfficientNetV2):
    # 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, network = 'efficientnet_v2_l', 
        dropout=0.1, stochastic_depth=0.2, start_size = 224):
        residual_config = [MBConvConfig(*layer_config) for layer_config in get_efficientnet_v2_structure(network)]
        super(Model, self).__init__(residual_config, 1280, 21843, 
            dropout=dropout, stochastic_depth=stochastic_depth, block=MBConv, act_layer=nn.SiLU)

        self.network = network

        # Resemble FPN layers
        # block
        # s
        # 1: 24, 1/2, 1/2       (block 1)           64      56
        # 2: 48, 1/4, 1/4       (block 5)           32      28
        # 3: 64, 1/8, 1/8       (block 9)           16      14
        # 4: 128, 1/16, 1/16    (block 15)          8       7
        # 5: 160, 1/16, 1/16    (block 24)          8       7
        # 6: 256, 1/32, 1/32    (block 39)          4
        # m:
        # 1: 24, 1/2, 1/2       (block 2)
        # 2: 48, 1/4, 1/4       (block 7)
        # 3: 80, 1/8, 1/8       (block 12)
        # 4: 160, 1/16, 1/16    (block 19)
        # 5: 176, 1/16, 1/16    (block 33)
        # 6: 304, 1/32, 1/32    (block 51)
        # 7: 512, 1/32, 1/32    (block 56)
        # l
        # 1: 32, 1/2, 1/2, block 3
        # 2: 64, 1/4, 1/4, block 10
        # 3: 96, 1/8, 1/8, block 17
        # 4: 192, 1/16, 1/16,   block 27
        # 5: 224, 1/16, 1/16,   block 46
        # 6: 384, 1/32, 1/32,   block 71
        # 7: 640, 1/32, 1/32,   block 78
        self.network_size = network.split('_')[-1]
        if network == 'efficientnet_v2_s':
            channel_arr = [48, 64, 128, 160] #, 256]
            self.limitation = 25
        elif network == 'efficientnet_v2_m':
            channel_arr = [48, 80, 160, 176] #, 304]
            self.limitation = 34
        else:
            channel_arr = [64, 96, 192, 224] #, 384]
            self.limitation = 47
        
        self.side = Pyramid(channel_arr)
        aggregate_channel = channel_arr[-1]
        self.aspp = DeepLabHead(aggregate_channel, 256)
        
        # Reduce Value
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_yaw_coarse = nn.Linear(256, 10)
        self.fc_yaw_shift  = nn.Linear(256, 20)  
        self.fc_pitch_coarse = nn.Linear(256, 10)
        self.fc_pitch_shift = nn.Linear(256, 20) 
        self.fc_roll_coarse = nn.Linear(256, 10)
        self.fc_roll_shift = nn.Linear(256, 20) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.stem(x)
        idx = 0

        out = []
        frame_id = 0
        for i,stage in enumerate(self.blocks[:self.limitation]):
            x = stage(x)

            if self.network == 'efficientnet_v2_s':
                if i == 5 or i == 9 or i == 15 or i == 24:
                    out.append(x)
                    frame_id += 1
            elif self.network == 'efficientnet_v2_m':
                if i == 7 or i == 12 or i == 19 or i == 33:
                    out.append(x)
                    frame_id += 1
            else:
                if i == 10 or i == 17 or i == 27 or i == 46:
                    out.append(x)
                    frame_id += 1

        output = self.side(out)
        output = self.aspp(output)
        output = self.avgpool(output) 
        output = output.view(output.size(0), -1)
        
        pre_yaw = self.fc_yaw_coarse(output)
        pre_yaw_shift = self.fc_yaw_shift(output)
        pre_pitch = self.fc_pitch_coarse(output)
        pre_pitch_shift = self.fc_pitch_shift(output)
        pre_roll = self.fc_roll_coarse(output)
        pre_roll_shift = self.fc_roll_shift(output)

        return pre_yaw, pre_yaw_shift, pre_pitch, pre_pitch_shift, pre_roll, pre_roll_shift #pre_yaw, pre_pitch, pre_roll

def nice_model(pretrained=False, network = 'efficientnet_v2_s', start_size = 224):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    def remove_head(model):
        model.blocks = model.blocks[:model.limitation]
        del model.head
        return model

    model = Model(network = network, start_size = start_size)

    if pretrained:
        if network != 'efficientnet_v2_l':
            if network == 'efficientnet_v2_s':
                pretrained = 'pretrained/pretrained_effv2_s.pkl'
            elif network == 'efficientnet_v2_m':
                pretrained = 'pretrained/pretrained_effv2_m.pkl'
            
            model = remove_head(model)
            saved_state_dict = torch.load(pretrained, map_location = 'cuda')
            model.load_state_dict(saved_state_dict['model'] if 'model' in saved_state_dict else saved_state_dict)
        else:
            load_from_zoo(model, f'{ network }_in21k')
            model = remove_head(model)
    else:
        model = remove_head(model)
        
    return model
