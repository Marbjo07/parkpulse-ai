import torch
import torch.nn as nn

from torchsummary import summary


model_save_file = "./model/pretrained_model.pth"
from model.model import *

model = torch.load(model_save_file)
model.eval()

summary(model, input_size=(3, 128, 128))

test_input = torch.rand(5, 3, 128, 128).cuda()

print(model(test_input))

torch.save(nn.Sequential(*(list(model.model.children())[:8])), f="./model/part1.pth")
torch.save(nn.Sequential(*(list(model.model.children())[8:])), f="./model/part2.pth")


del model

model = nn.Sequential(torch.load("./model/part1.pth"), torch.load("./model/part2.pth"))
print(model(test_input))