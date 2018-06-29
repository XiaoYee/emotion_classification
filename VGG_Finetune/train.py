import torch
import torchvision
from torch import nn
import torchvision.transforms as transforms
from VGG_Face_torch import VGG_Face_torch
import argparse
import torch.optim as optim
from torch.autograd import Variable



parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 16)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=30, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

kwargs = {'num_workers': 4, 'pin_memory': True}

print(args)


transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                 transforms.Resize(256),
                                #  transforms.RandomResizedCrop((224,224), scale=(0.875, 1.125), ratio=(1.0, 1.0)),
                                #  transforms.CenterCrop(224),
                                 transforms.RandomCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.507395516207, ),(0.255128989415, ))
                                ])

transform_test  = transforms.Compose([transforms.Resize(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.507395516207, ),(0.255128989415, ))
                                ])


train_data = torchvision.datasets.ImageFolder('./train',transform=transform_train)
test_data = torchvision.datasets.ImageFolder('./test',transform=transform_test)
                                            
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data,   batch_size=args.test_batch_size, shuffle=False, **kwargs)


class average_meter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class VGG_Net(nn.Module):
    def __init__(self, model):
        super(VGG_Net, self).__init__()

        self.pre_model = nn.Sequential(*list(model.children())[:-1])
        # self.dropout = nn.Dropout(p=0.8)
        self.classifier = nn.Linear(4096, 7)

    def forward(self, x):
        x = self.pre_model(x)
        # x = self.dropout(x)
        x = self.classifier(x)

        return x 


model_emotion = VGG_Face_torch
model_emotion.load_state_dict(torch.load('VGG_Face_torch.pth'))
model = VGG_Net(model_emotion).cuda()


loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum ,weight_decay= 0.0005,nesterov=True)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=20, gamma=0.1)



def train(epoch):
    
    losses = average_meter()
    accuracy = average_meter()
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        output = model(data)
        loss = loss_function(output, target)
        losses.update(loss.data[0], data.size(0))

        pred = output.data.max(1)[1]
        prec = pred.eq(target.data).cpu().sum()
        accuracy.update(float(prec) / data.size(0), data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {}\t'
                  'Batch: [{:5d}/{:5d} ({:3.0f}%)]\t'                     
                  'Loss: {:.6f}'.format(
                      epoch, batch_idx * len(data), len(train_data),
                      100. * batch_idx / len(train_loader), losses.val))
            print('Training accuracy:', accuracy.val )


def test():
    losses = average_meter()
    accuracy = average_meter()

    model.eval()

    for data, target in test_loader:

        data, target = Variable(data,volatile=True).cuda(), Variable(target,volatile=True).cuda()
        output = model(data)

        loss = loss_function(output, target)
        losses.update(loss.data[0], data.size(0))

        pred = output.data.max(1)[1]
        prec = pred.eq(target.data).cpu().sum()
        accuracy.update(float(prec) / data.size(0), data.size(0))

    print('\nTest: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        losses.avg, int(accuracy.sum), len(test_data), 100. * accuracy.avg))

    return accuracy.avg


def main():

    best_model = model
    best_accuray = 0.0

    for epoch in range(1, args.epochs + 1):

        scheduler.step()
        train(epoch)
        val_accuracy = test()

        if best_accuray < val_accuracy:
            best_model   = model
            best_accuray = val_accuracy


    print ("The best model has an accuracy of " + str(best_accuray))
    torch.save(best_model.state_dict(), 'best.model')


if __name__ == '__main__':
    main()
