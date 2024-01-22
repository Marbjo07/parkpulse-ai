import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data import dataloader
from random import random

class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, train, transform=None):
        super(PatchDataset, self).__init__()
        self.data_dir = data_dir
        self.files = []
        for sub_dir in os.listdir(data_dir):
            sub_dir += "/train" if train else "/test"
            dir = os.path.join(data_dir, sub_dir)
            files_in_dir = os.listdir(dir)
            
            # Concatenate sub_dir to each file
            files_in_dir = [os.path.join(sub_dir, file) for file in files_in_dir]
            
            self.files += files_in_dir

        self.transform = transform
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.data_dir,
                                self.files[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.ones((1)) if "car" in self.files[idx] else torch.zeros((1))

        return image, label

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        import torchvision
        self.model = torchvision.models.resnet50(torchvision.models.ResNet50_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-2]))
        
        print(self.model)

        self.model.append(nn.Sequential(nn.Flatten(), nn.LazyLinear(512), nn.ReLU(), nn.Linear(512, 1), nn.Sigmoid()))

    def forward(self, x):
        return self.model(x)
    
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    eval_criterion = nn.L1Loss(reduction="sum")
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += eval_criterion(output, target).item()  # sum up batch loss

    test_loss /= float(len(test_loader))
    test_loss /= float(batch_size)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., rate=0.5):
        self.std = std
        self.mean = mean
        self.rate = rate

    def __call__(self, tensor):
        if random() < self.rate:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            return tensor
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

if __name__ == "__main__":
    device = torch.device("cuda")
    lr = 0.003
    gamma = 0.9
    momentum = 0.7
    batch_size = 32
    EPOCHS = 100
    log_interval = 5
    dry_run = False
    image_size = 128

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=(0, 360)),
        transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5)),
        ])

    data_root = "./dataset"

    dataset_train = PatchDataset(data_root, train=True, transform=transform)
    dataset_test = PatchDataset(data_root, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5)),
]))

    train_loader = dataloader.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, persistent_workers=True)
    test_loader = dataloader.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, drop_last=True)

    print(len(dataset_train.files))
    print(len(dataset_test.files))


    model = Net()
    model = model.to(device)
    print(model)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.BCELoss()

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma, verbose=True)

    for epoch in range(EPOCHS):
        p_bar = tqdm(train_loader, nrows=10)
        for batch_idx, (data, target) in enumerate(p_bar):
            data, target = data.to(device), target.to(device)


            output = model(data)

            optimizer.zero_grad()
            loss = criterion(output.view(-1, 1), target.view(-1, 1))

            loss.backward()

            optimizer.step()

            if batch_idx % log_interval == 0:
                p_bar.set_description_str('Train Epoch: {} Loss: {:.6f}'.format(epoch, loss.item()))
                if dry_run:
                    break
        torch.save(model, f"./models/model{epoch}.pth")
        test(model, device, test_loader)
        if epoch < 20:
            scheduler.step()
