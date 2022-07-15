import torch

class Discriminator(torch.nn.Module):
    def __init__(self,):
        super(Discriminator,self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=1),
            # torch.nn.MaxPool3d(2),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
            # torch.nn.MaxPool3d(2),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )

        self.linear = torch.nn.Linear(2048, 128)
        self.logit = torch.nn.Linear(128,1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = x.view(-1,1,128,128,128)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.view(x.size(0),-1)
        x = self.relu(self.linear(x))
        x = self.logit(x)
        x = self.sigmoid(x)
        return x


if __name__=="__main__":
    net = Discriminator()
    x = torch.rand(2,1,128,128,128)

    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))

    print(net(x).shape)