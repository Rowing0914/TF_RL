import torch.nn as nn
import torch.nn.functional as F


# the convolution layer of deepmind
class deepmind(nn.Module):
    def __init__(self):
        super(deepmind, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        # start to do the init...
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
        # init the bias...
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 7 * 7)

        return x


# in the initial, just the nature CNN
class net(nn.Module):
    def __init__(self, num_actions):
        super(net, self).__init__()
        # define the network
        self.cnn_layer = deepmind()
        self.fc1 = nn.Linear(32 * 7 * 7, 256)
        self.action_value = nn.Linear(256, num_actions)

    def forward(self, inputs):
        x = self.cnn_layer(inputs / 255.0)
        x = F.relu(self.fc1(x))
        action_value_out = self.action_value(x)
        return action_value_out
