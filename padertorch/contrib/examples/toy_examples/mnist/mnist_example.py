"""
A basic example using the mnist database using sacred and using either
the padertorch trainer or a simple rudimentary training function.

Use the following command:
    python mnist_example.py with logfilepath=/path/to/dir/ epochs=500

Other options for sacred are available, refer to the config() function.
"""
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import sacred

import padertorch
import padertorch.train.optimizer as pt_opt


ex = sacred.Experiment()


class FeedForwardNet(nn.Module):
    """A Simple 2-Layer Perceptron Neural Network."""

    def __init__(self, layer_size=800):
        super().__init__()
        self.layer_size = layer_size
        self.fc1 = nn.Linear(28*28, self.layer_size)
        self.fc2 = nn.Linear(self.layer_size, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PadertorchModel(padertorch.base.Model):

    loss_function = nn.CrossEntropyLoss()

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs):
        images = inputs[0]
        out = self.net(images)

        return dict(
            prediction=out
        )

    def review(self, inputs, outputs):

        loss = self.loss_function(
            outputs["prediction"],
            inputs[1]
        )

        _, predicted = torch.max(outputs["prediction"], 1)

        acc = (predicted == inputs[1]).sum().item() / len(inputs[1])

        return dict(
            loss=loss,
            scalars={'Accuracy': acc},
            images={'image': inputs[0][0]},
            texts={
                'label': str(inputs[1][0]),
                'predicted': str(predicted[0]),
            },
        )


def cnt_params(net):
    res = 0
    for l in net.parameters():
        l = list(l.size())
        tmp = 1
        for i in l:
            tmp *= i
        res += tmp
    return res


def train(net, trainloader, gpu=False):
    device = None

    if gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        net.to(device)

    # Set loss function and training algorithm
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    accum_loss = 0.0
    for it, (images, labels) in enumerate(trainloader, 0):
        if gpu:
            images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        accum_loss += loss.item()

        # CPU is slow, use this to still get some kind of feedback
        # if it % 2500 == 2499:
        #     print("Iteration %d" % (it+1))

    return accum_loss
        

def validate(net, testloader, gpu=False):
    device = None
    
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda")

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            if device:
                images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct


@ex.config
def config():
    use_pt = True       # Use trainer from pt
    layer_size = 800    # Variable hidden layer size
    epochs = 2          # Use resume=True to train for more epochs
    resume = False      # PT: Continue from checkpoints
    logfilepath = os.environ['STORAGE_ROOT'] + '/mnist'  # PT creates log files


@ex.automain
def main(layer_size, epochs, logfilepath, resume, use_pt):

    tf = transforms.Compose([transforms.ToTensor()])

    # Get Datasets
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=tf)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=tf)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)

    # Setup net and model
    net = FeedForwardNet(layer_size=layer_size)

    if use_pt:
        # Use trainer from padertorch library
        model = PadertorchModel(net=net)

        trainer = padertorch.trainer.Trainer(
            model,
            logfilepath,
            pt_opt.SGD(),
            stop_trigger=(epochs, 'epoch'),
            #checkpoint_trigger=(5000, 'iteration'),
            summary_trigger=(1000, 'iteration'),
        )
        trainer.register_validation_hook(validation_iterator=testloader)
        try:
            trainer.train(trainloader, resume=resume)
        except Exception:
            print('#' * 1000)
            raise

    else:
        # Use our rudimentary trainer
        print("#Parameters =", cnt_params(net))

        start = time.time()
        for i in range(epochs):
            start_it = time.time()
            accum_loss = train(net, trainloader, gpu=True)
            correct = validate(net, testloader, gpu=True)
            print("[Epoch %d] Average Loss: %.3f; Correct on Test: %.2f%%; "
                  "Iteration: %3ds; Since Start: %3ds" %
                  ((i+1),
                   accum_loss / 15000,
                   100 * correct / 10000.0,
                   (time.time()-start_it),
                   (time.time() - start)))
