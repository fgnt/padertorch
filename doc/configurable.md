# Configurable: What is it good for?

The main idea of the `Configurable` object is to simplyfy the initialization
of a neural network model from a config dictionary.
A neural network usually consists of multiple modules with inter-dependent
configurations which are combined in the main model.
An simple model inspired by the model in
[pytorch tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)
looks like this:
```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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
    def __init__(self, fc1, fc2):
        super().__init__()
        self.fc1 = fc1
        self.fc2 = fc2

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net(nn.Linear(120, 84), nn.Linear(84, 10))
```
Using`Configurable` we can initialize `Net` from a dictionary:
 
```python
config = dict(
    factory=Net,
    fc1=dict(
        factory=nn.Linear,
        in_features=120,
        out_features=84
    ),
    fc2=dict(
        factory=nn.Linear,
        in_features=84,
        out_features=10
    ),
)
net = pt.Configurable.from_config(update)
```
This initialization looks more complicated than the two possiblities above,
but it has major upsites.
The main benefit is that we can now easily adjust the parameters from outside
of the model.
Another benefit is, that this initialization allows to easily save the used
configuration for example by writting the config dictionary to a json file.
However, the current config does not include all parameters used in the model
since there are some default values in the modules which are not specified in
the config dictionary.
We can get a config with all parameters in the model by
calling `pt.Configurable.get_config`:
```python
update = dict(
    factory=Net,
    fc1=dict(
        factory=nn.Linear,
        in_features=120,
        out_features=84
    ),
    fc2=dict(
        factory=nn.Linear,
        in_features=84,
        out_features=10
    ),
)
full_config = pt.Configurable.get_config(update)
```
The full config now includes all input parameters to the sub-modules:
```
{'factory': 'Net',
 'fc1': {'factory': 'torch.nn.modules.linear.Linear',
  'in_features': 120,
  'out_features': 84,
  'bias': True},
 'fc2': {'factory': 'torch.nn.modules.linear.Linear',
  'in_features': 84,
  'out_features': 10,
  'bias': True}}
```
The key ```factory``` is a fixed signifier for a class representer and should
 not be used as a parameter name in combination with `Configurable`. 


## Additonal perks of using Configurable 
