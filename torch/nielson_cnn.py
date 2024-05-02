import torch
import torchvision


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # first convolutional layer with 1 input channel,
        # 3 output features, square kernel size 5, and
        # a stride of size 1.
        self.conv1 = torch.nn.Conv2d(1, 3, 5, 1)

        # fully-connected layer
        self.fc = torch.nn.Linear(3*12*12, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.MaxPool2d(2)(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = torch.nn.functional.softmax(x, dim=1)
        return output


def train(model, training_loader, optimizer):
    model.train()  # ?
    for data, target in training_loader:
        optimizer.zero_grad()  # ?
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()  # ?
        optimizer.step()  # ?


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            # sum up batch loss
            test_loss += torch.nn.functional.nll_loss(
                output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# Setup the model, import the data, and let'er rip.

net = Net()
mnist_dataset = torchvision.datasets.MNIST(root='data/', download=True,
                                           train=True,
                                           transform=torchvision.transforms.ToTensor())

training_data, validation_data = torch.utils.data.random_split(mnist_dataset,
                                                               [50000, 10000])

mini_batch_size = 10
EPOCHS = 60
eta = 0.1

training_loader = torch.utils.data.DataLoader(
    training_data, mini_batch_size, shuffle=True)

validation_loader = torch.utils.data.DataLoader(
    validation_data, 10000, shuffle=False)

# loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=eta)


def main():
    for _ in range(EPOCHS):
        train(net, training_loader, optimizer)
        test(net, validation_loader)

    PATH = "models/mnist_cnn.pt"
    torch.save(net.state_dict(), PATH)

# random_data = torch.rand((1, 1, 28, 28))
# result = net(random_data)
# print(result)


if __name__ == "__main__":
    main()
