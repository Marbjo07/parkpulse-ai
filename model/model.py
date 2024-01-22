import torchvision
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.model = torchvision.models.resnet50(torchvision.models.ResNet50_Weights.DEFAULT)
        self.model = nn.Sequential(*(list(self.model.children())[:-2]))
        
        print(self.model)

        self.model.append(nn.Sequential(nn.Flatten(), nn.LazyLinear(512), nn.Linear(512, 1), nn.Sigmoid()))

    def forward(self, x):
        return self.model(x)