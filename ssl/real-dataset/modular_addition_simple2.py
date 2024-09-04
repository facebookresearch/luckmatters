import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# Define the modular addition function
def modular_addition(x, y, mod):
    return (x + y) % mod

def generate_dataset(M):
    data = []
    for x in range(M):
        for y in range(M):
            z = modular_addition(x, y, M)
            data.append((x, y, z))
    return data

nll_criterion = nn.CrossEntropyLoss().cuda()

def compute_loss(outputs, labels, loss_type):
    loss = 0
    for i, o in enumerate(outputs):
        if loss_type == "nll":
            loss = loss + nll_criterion(o, labels[:,i])
        elif loss_type == "mse":
            o_zero_mean = o - o.mean(dim=1, keepdim=True)
            loss = loss + o_zero_mean.pow(2).sum(dim=1).mean() - 2 * o_zero_mean.gather(1, labels[:,i].unsqueeze(1)).mean() + 1 - 1.0 / o.shape[1] 
        else:
            raise RuntimeError(f"Unknown loss! {args.loss}")

    return loss

def test_model(model, X_test, y_test, loss_type):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        corrects = [None] * len(outputs)
        for (i, o) in enumerate(outputs):
            _, predicted = torch.max(o.data, 1)
            corrects[i] = (predicted == y_test[:,i]).sum().item() / y_test.size(0)

        loss = compute_loss(outputs, y_test, loss_type).item()

    return corrects, loss

# Define the neural network model
class ModularAdditionNN(nn.Module):
    def __init__(self, M, hidden_size, activation="sqr", use_bn=False):
        super(ModularAdditionNN, self).__init__()
        self.embedding = nn.Embedding(M, M).requires_grad_(False)
        with torch.no_grad():
            self.embedding.weight[:] = torch.eye(M, M)
            
        self.layera = nn.Linear(M, hidden_size, bias=False)
        self.layerb = nn.Linear(M, hidden_size, bias=False)
        self.layerc = nn.Linear(hidden_size, M, bias=False)

        if use_bn:
            self.bn = nn.BatchNorm1d(hidden_size)
            self.use_bn = True
        else:
            self.use_bn = False

        self.relu = nn.ReLU()
        self.activation = activation
        self.M = M
    
    def forward(self, x):
        y1 = self.embedding(x[:,0])
        y2 = self.embedding(x[:,1]) 
        # x = torch.relu(self.layer1(x))
        x = self.layera(y1) + self.layerb(y2) 
        if self.use_bn:
            x = self.bn(x)

        if self.activation == "sqr": 
            x = x.pow(2)
        elif self.activation == "relu":
            x = self.relu(x)
        else:
            raise RuntimeError(f"Unknown activation = {self.activation}")
            
        return [self.layerc(x)]

    def normalize(self):
        with torch.no_grad():
            self.layera.weight[:] -= self.layera.weight.mean(dim=1, keepdim=True) 
            self.layerb.weight[:] -= self.layerb.weight.mean(dim=1, keepdim=True) 
            self.layerc.weight[:] -= self.layerc.weight.mean(dim=0, keepdim=True) 

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("-M", type=int, default=127)
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--optim", choices=["adam", "sgd"], default="adam")
parser.add_argument("--activation", choices=["relu", "sqr"], default="sqr")
parser.add_argument("--num_epochs", type=int, default=3000)
parser.add_argument("--test_size", type=float, default=0.7)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--use_bn", action="store_true")
parser.add_argument("--loss", choices=["nll", "mse"], default="nll")
parser.add_argument("--save_interval", type=int, default=1000)
parser.add_argument("--eval_interval", type=int, default=100)
parser.add_argument("--normalize", action="store_true")

args = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(args.seed)

# Hyperparameters
# sqr activation, no bn
# learning_rate = 0.0002
# sqr activation, no bn and with MSE
# learning_rate = 0.001
# sqr activation, with bn
# learning_rate = 0.002

# Generate dataset
dataset = generate_dataset(args.M)
dataset_size = len(dataset)

# Prepare data for training and testing
X = torch.LongTensor(dataset_size, 2)
# Use 
labels = torch.LongTensor(dataset_size, 1)

for i, (x, y, z) in enumerate(dataset):
    X[i, 0] = x
    X[i, 1] = y
    labels[i] = z

y = labels

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

X_train = X_train.cuda()
X_test = X_test.cuda()
y_train = y_train.cuda()
y_test = y_test.cuda()

# Initialize the model, loss function, and optimizer
model = ModularAdditionNN(args.M, args.hidden_size, activation=args.activation, use_bn=args.use_bn)

model = model.cuda()


if args.optim == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
elif args.optim == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
else:
    raise RuntimeError(f"Unknown optimizer! {args.optim}")

# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=3e-4)

results = []

# Training loop
for epoch in range(args.num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train)
    # loss = criterion(outputs, y_train)
    loss = compute_loss(outputs, y_train, args.loss)
    
    # Backward and optimize
    loss.backward()
    optimizer.step()

    if args.normalize:
        model.normalize()

    if (epoch+1) % args.eval_interval == 0:
        print(f'Epoch [{epoch}/{args.num_epochs}], Loss: {loss.item():.4f}')

        # Test the model
        train_accuracies, train_loss = test_model(model, X_train, y_train, args.loss)
        test_accuracies, test_loss = test_model(model, X_test, y_test, args.loss)

        train_acc = train_accuracies[0]
        test_acc = test_accuracies[0]

        print(f"Train Accuracy/Loss: {train_acc}/{train_loss}")
        print(f"Test Accuracy/Loss: {test_acc}/{test_loss}\n")

        results.append(dict(epoch=epoch, train_acc=train_acc, test_acc=test_acc, train_loss=train_loss, test_loss=test_loss))

    if (epoch+1) % args.save_interval == 0:
        filename = f"model{epoch:05}_train{train_acc:.2f}_loss{train_loss:.4f}_test{test_acc:.2f}_loss{test_loss:.4f}.pt" 

        data = dict(model=model.state_dict(), results=results) 

        torch.save(data, filename)

