from torch import nn
import torch


class TemporalTransimitionLayer(nn.Module):
    def __init__(self, input_channel, output_channel, depths=[1, 3, 6]):
        super(TemporalTransimitionLayer, self).__init__()
        self.norm = nn.BatchNorm3d(input_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv1=nn.Conv3d(input_channel, output_channel, (depths[0], 1, 1))
        self.conv2=nn.Conv3d(input_channel, output_channel,
                             (depths[1], 3, 3), padding = (0, 1, 1))
        self.conv3=nn.Conv3d(input_channel, output_channel,
                             (depths[2], 3, 3), padding = (0, 1, 1))
        self.pool=nn.AvgPool3d((2, 2, 2), stride = 2)

    def forward(self, x):
        print('ttl input', x.size())
        x = self.relu(self.norm(x))
        x1=self.conv1(x)
        x2=self.conv2(x)
        x3=self.conv3(x)
        print(x1.size(), x2.size(), x3.size())
        output=torch.cat((x1, x2, x3), 2)
        print('#######', output.size())
        output=self.pool(output)
        print('#######', output.size())
        return output
