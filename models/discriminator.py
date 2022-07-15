import torch

class Discriminator(torch.nn.Module):
    def __init__(self,):
        super(Discriminator,self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32,kernel_size=5),
            torch.nn.MaxPool3d(2),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(32,64,kernel_size=3),
            torch.nn.MaxPool3d(2),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(64,128,kernel_size=3),
            torch.nn.MaxPool3d(1),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(128,256,kernel_size=3),
            torch.nn.MaxPool3d(1),
            torch.nn.ReLU()
        )
        self.linear = torch.nn.Linear(2048, 128)
        self.logit = torch.nn.Linear(128,1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = x.view(-1,1,32,32,32)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0),-1)
        x = self.relu(self.linear(x))
        x = self.logit(x)
        x = self.sigmoid(x)
        return x


if __name__=="__main__":
    net = Discriminator()
    x = torch.rand(2,1,32,32,32)

    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))

    print(net(x).shape)