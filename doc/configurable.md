# Configurable: What is it good for?

The main idea of the configurable object is to simplyfy the initialization
of a neural network model from a config dictionary.
A neural network usually consists of multiple modules with inter-dependent
configurations which are combined in the main model.
An example model taken from the
[pytorch tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)
looks like this:
```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
```
Instantiating the modules inside of the model makes it more difficult to
change parameter in the module or leads to an unnecessary high amount of
input parameters for the model init.
Therefore, we assume that the already initialized modules are the input
to the model:
```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, conv1, pool, conv2, fc1, fc2, fc3):
        super().__init__()
        self.conv1 = conv1
        self.pool = pool
        self.conv2 = conv2
        self.fc1 = fc1
        self.fc2 = fc2
        self.fc3 = fc3

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net(nn.Conv2d(3, 6, 5), nn.MaxPool2d(2, 2), nn.Conv2d(6, 16, 5),
          nn.Linear(16 * 5 * 5, 120), nn.Linear(120, 84), nn.Linear(84, 10))
```
Using Configurable we can initialize Net from a dictionary:
 
```python
import torch.nn as nn
import torch.nn.functional as F
import padertorch as pt

class Net(nn.Module):
    def __init__(self, conv1, pool, conv2, fc1, fc2, fc3):
        super().__init__()
        self.conv1 = conv1
        self.pool = pool
        self.conv2 = conv2
        self.fc1 = fc1
        self.fc2 = fc2
        self.fc3 = fc3

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

config = dict(
    factory=Net,
    conv1=dict(
        factory=nn.Conv2d,
        in_channels=3,
        out_channels=6,
        kernel_size=5
    ),
    pool=dict(
        factory=nn.MaxPool2d,
        kernel_size=2,
        stride=2
    ),
    conv2=dict(
        factory=nn.Conv2d,
        in_channels=6,
        out_channels=16,
        kernel_size=5
    ),
    fc1=dict(
        factory=nn.Linear,
        in_features=16*5*5,
        out_features=120
    ),
    fc2=dict(
        factory=nn.Linear,
        in_features=120,
        out_features=84
    ),
    fc3=dict(
        factory=nn.Linear,
        in_features=84,
        out_features=10
    ),
)
net = pt.Configurable.from_config(config)
```
This initialization looks more complicated than the two possiblities above,
but it has major upsites.
The main benefit is that we can now easily adjust the parameters from outside
of the model.
Another benefit is, that this initialization allows to easily save the used
configuration for example by writting the config dictionary to a json file.
However, the current config does not include all parameters used in the model
since there are some default values in the modules which are not specified in
the config dict.
If Net is a sub-class of Configurable (for example by replacing nn.Module with
padertorch.Module) we can get a config with all parameters in the model by
calling ```Net.get_config(config)```:
```python
import torch.nn as nn
import torch.nn.functional as F
import padertorch as pt

class Net(pt.Module):
    def __init__(self, conv1, pool, conv2, fc1, fc2, fc3):
        super().__init__()
        self.conv1 = conv1
        self.pool = pool
        self.conv2 = conv2
        self.fc1 = fc1
        self.fc2 = fc2
        self.fc3 = fc3

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

config = dict(
    factory=Net,
    conv1=dict(
        factory=nn.Conv2d,
        in_channels=3,
        out_channels=6,
        kernel_size=5
    ),
    pool=dict(
        factory=nn.MaxPool2d,
        kernel_size=2,
        stride=2
    ),
    conv2=dict(
        factory=nn.Conv2d,
        in_channels=6,
        out_channels=16,
        kernel_size=5
    ),
    fc1=dict(
        factory=nn.Linear,
        in_features=16*5*5,
        out_features=120
    ),
    fc2=dict(
        factory=nn.Linear,
        in_features=120,
        out_features=84
    ),
    fc3=dict(
        factory=nn.Linear,
        in_features=84,
        out_features=10
    ),
)
full_config = Net.get_config(config)
```




## Configurable examples
