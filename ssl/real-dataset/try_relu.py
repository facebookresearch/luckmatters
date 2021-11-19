import torch
import torch.nn as nn
import random
import torch.optim as optim
import numpy as np

class Model(nn.Module):
    def __init__(self, d, num_hidden):
        super(Model, self).__init__()

        self.extractor = nn.Linear(d, num_hidden)
        self.relu = nn.ReLU()
        self.predictor = nn.Linear(num_hidden, num_hidden)

    def forward(self, x1, x2):
        output1 = self.relu(self.extractor(x1)) 
        output2 = self.relu(self.extractor(x2)) 

        # Directly send the output to predictor and compute loss. 
        branch1 = self.predictor(output1)
        branch2 = output2.detach()

        branch1 = branch1 / branch1.norm(dim=1, keepdim=True)
        branch2 = branch2 / branch2.norm(dim=1, keepdim=True)

        return branch1, branch2

class Generator:
    def __init__(self, d):
        # check ReLU case. 
        self.d = d

        # d choose 2, create an over-complete basis. 
        self.K = self.d * (self.d - 1) // 2
        self.Zs = torch.zeros(self.d, self.K)
        cnt = 0
        for i in range(d):
            for j in range(i + 1, d):
                self.Zs[i, cnt] = 1
                self.Zs[j, cnt] = 1
                cnt += 1

        # Treat the second part as the noisy component.
        self.signal_part = self.Zs[:, self.K//2:]
        self.noise_part = self.Zs[:, :self.K//2]

        print("Signal part:")
        print(self.signal_part)

        print("Noise part:")
        print(self.noise_part)

    def generate(self, batchsize):
        indices = list(range(self.K // 2))

        random.shuffle(indices) 
        x = self.signal_part[:, indices[:batchsize]].t()

        random.shuffle(indices) 
        x1 = x + 0.5*self.noise_part[:, indices[:batchsize]].t() 

        random.shuffle(indices) 
        x2 = x + 0.5*self.noise_part[:, indices[:batchsize]].t() 

        return x1, x2


# construct a very simple neural network and train with DirectPred
d = 10

generator = Generator(d)
num_hidden = 2 * generator.K
model = Model(d, num_hidden)

with torch.no_grad():
    model.extractor.bias[:] = - 0.05 * torch.rand(num_hidden)

loss_func = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.0)

batchsize = 8

for t in range(10000):
    optimizer.zero_grad()

    x1, x2 = generator.generate(batchsize)
    branch1, branch2 = model(x1, x2)

    loss = loss_func(branch1, branch2)
    if t % 100 == 0:
        print(f"{t}: {loss.item()}")
    loss.backward() 
    optimizer.step()

# Check whether the extractor weight is aligned with the features. 
# #type of input x #response of nodes. 
response_signal = model.extractor(generator.signal_part.t())
response_noise = model.extractor(generator.noise_part.t())

response_signal = (response_signal >= 0).float()
response_noise = (response_noise >= 0).float()

signal_noise_diff = response_signal.mean(dim=0) - response_noise.mean(dim=0)

all_response = torch.cat([response_signal, response_noise], dim=0)
sum_response = all_response.sum(dim=0)
one_to_other_diff = all_response - sum_response[None, :] / (sum_response.size(0) - 1)

print("signal_noise_diff")
print(signal_noise_diff)

print("one_to_other_diff")
print(one_to_other_diff)

print(model.extractor)
import pdb
pdb.set_trace()
