import torch
import torch.nn
import torch.nn.functional as F


class nn2048(torch.nn.Module):
    def __init__(self):
        super(nn2048, self).__init__()
        self.model_name = str(type(self))
        self.conv1 = torch.nn.Conv2d(16, 128, kernel_size=(2, 1), padding=0, bias=True)
        self.conv2 = torch.nn.Conv2d(16, 128, kernel_size=(1, 2), padding=0, bias=True)

        self.conv11 = torch.nn.Conv2d(128, 128, kernel_size=(2, 1), padding=0, bias=True)
        self.conv12 = torch.nn.Conv2d(128, 128, kernel_size=(1, 2), padding=0, bias=True)
        self.conv21 = torch.nn.Conv2d(128, 128, kernel_size=(2, 1), padding=0, bias=True)
        self.conv22 = torch.nn.Conv2d(128, 128, kernel_size=(1, 2), padding=0, bias=True)

        self.fc1 = torch.nn.Linear(128 * 34, 1024)
        self.fc2 = torch.nn.Linear(1024, 128)
        self.fc3 = torch.nn.Linear(128, 4)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x = torch.cat([self.conv11(x1).view(x.size()[0], -1),
                       self.conv12(x1).view(x.size()[0], -1),
                       self.conv21(x2).view(x.size()[0], -1),
                       self.conv22(x2).view(x.size()[0], -1)], dim=1)

        x = self.fc1(F.relu(x))
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))

        return x

    def next(self, x):
        with torch.no_grad():
            x = self.forward(x)
            nextstep = torch.argmax(x, dim=1)
        return nextstep
