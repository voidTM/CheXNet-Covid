# encoding: utf-8

"""
The main CheXNet model implementation.
"""


import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


CKPT_PATH = 'model.pth.tar'
DATA_DIR = './data'
BATCH_SIZE = 20
EPOCHS = 25
N_CLASSES = 2


def main():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    print("using", device)
    # initialize and load the model
    model = DenseNet121(N_CLASSES).to(device)
    model = torch.nn.DataParallel(model).to(device)

    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")
        return



    data_transform = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor()
    ])
    
    dataset = datasets.ImageFolder(data_dir, data_transform)

    train_set, test_set = split_dataset(dataset, 0.5)

    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2, pin_memory=True)

    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2, pin_memory=True)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    
    model = train_model(model, train_loader, criterion, optimizer, scheduler)

    torch.save(model, "chexnet_t1.pth")


def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=25):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i in range(epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            scheduler.step()

            # keep track of loss and correct
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)


        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))


    return model



def split_dataset(dataset, val_split= 0.5):
    set_size = len(dataset)

    val = int(set_size * val_split)
    train = set_size - val
    
    
    datasets = torch.utils.data.random_split(dataset, [train, val])
    return datasets



def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


if __name__ == '__main__':
    main()