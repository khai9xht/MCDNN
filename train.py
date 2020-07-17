import torch
import torch.nn as nn
import time
import copy
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch.optim as optim
from model import cnn, mdcnn


def train(model, datasets, criterion, optimizer, scheduler=None, epochs=10):
    """
    function: train neural network model normally
    parameters input:
        -model: model structure  --torch.nn.Module
        -datasets: data for training  --dictionary
            +structure: {'train': torch.utils.data.Dataset, 'val': torch.utils.data.Dataset}
        -criterion: type of loss (CrossEntropyLoss(output, target), MSELoss(output, target), vv)
        -optimizer: optimize loss (SGD, Adam, vv) --torch.optim
        -scheduler: adjust parameter(ltabearning_rate) of optimizer --torch.optim
        -epochs: number of epochs to train
    """

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=32,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    min_loss = 100
    since = time.time()

    for epoch in range(epochs):
        print('Epoch {}/{}:'.format(epoch+1, epochs))

        epoch_loss = {}
        epoch_acc = {}
        start = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in dataloaders[phase]:

                # labels_onehot = labels.numpy()
                # labels_onehot = (np.arange(10) == labels_onehot[:,None]).astype(np.float32)
                # labels_onehot = torch.from_numpy(labels_onehot).to(device)

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # track history when training
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    # labels = labels.view(1, -1)
                    loss = criterion(outputs, labels)

                    # backward + optimize
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics( one-hot-coding)
                running_loss += loss.item() * inputs.size(0)  # inputs.size(0) = batchsize
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train' and scheduler != None and epoch < 500:
                scheduler.step()

            epoch_loss[phase] = running_loss / dataset_sizes[phase]
            epoch_acc[phase] = running_corrects.double() / dataset_sizes[phase]

            if phase == 'val' and epoch_acc[phase] > best_acc:
                best_acc = epoch_acc[phase]
                min_loss = epoch_loss[phase]
                best_model_wts = copy.deepcopy(model.state_dict())
        time_per_epoch = time.time() - start
        print('time: {:.0f}m{:.0f}s    train_loss: {:.4f}    train_acc: {:.4f}    val_loss: {:.4f}    val_acc: {:.4f}'.format(
            time_per_epoch//60, time_per_epoch % 60, epoch_loss['train'], epoch_acc['train'], epoch_loss['val'], epoch_acc['val']
        ))
        print()

    time_elapsed = time.time() - since
    print('\n'+'_'*20)
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f} \n loss: {:.4f}'.format(best_acc, min_loss))

    # load or save model here( example load model)
    model.load_state_dict(best_model_wts)
    return model


if __name__ == 'main':
    transforms = transforms.Compose([
        transforms.Resize((12, 12)),
        transforms.ToTensor()
    ])
    train_dataset = datasets.MNIST(
        'data', train=True, transform=transforms, download=True)
    test_dataset = datasets.MNIST(
        'data', train=False, transform=transforms, download=True)
    model = cnn.cnn10x10()
    datasets = {
        'train': train_dataset,
        'val': test_dataset
    }
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.993)
    epochs = 800

    model = train(model, datasets, criterion, optimizer, scheduler, epochs)
