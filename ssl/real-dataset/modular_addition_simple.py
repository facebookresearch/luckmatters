import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.model_selection import train_test_split

# Define the modular addition function
def modular_addition(x, y, mod):
    return (x * y) % mod

def generate_dataset(M):
    data = []
    for x in range(M):
        for y in range(M):
            z = modular_addition(x, y, M)
            data.append((x, y, z))
    return data

def test_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        corrects = [None] * len(outputs)
        for (i, o) in enumerate(outputs):
            _, predicted = torch.max(o.data, 1)
            corrects[i] = (predicted == y_test[:,i]).sum().item() / y_test.size(0)

    return corrects

# Define the neural network model
class ModularAdditionNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_sizes, d, embedding_layer_fixed=True):
        super(ModularAdditionNN, self).__init__()
        self.embedding = nn.Embedding(input_size * 2, d).requires_grad_(not embedding_layer_fixed)
        if embedding_layer_fixed and input_size * 2 == d:
            print("Setting embedding to identity matrix")
            with torch.no_grad():
                self.embedding.weight[:] = torch.eye(d, d)
            
        self.layer1 = nn.Linear(d, hidden_size)
        output_layers = [ nn.Linear(hidden_size, ss) for ss in output_sizes ]
        self.output_layers = nn.ModuleList(output_layers)

        self.input_size = input_size
    
    def forward(self, x):
        f = self.embedding(x[:,0]) + self.embedding(x[:,1] + self.input_size) 
        x = torch.relu(self.layer1(f))
        return [ l(x) for l in self.output_layers ]

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("-d", type=int, default=32)
parser.add_argument("-M", type=int, default=127)
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--emb_fixed", action="store_true")
parser.add_argument("--num_epochs", type=int, default=3000)

args = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(args.seed)

# Hyperparameters
learning_rate = 0.02
test_size = 0.7

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

X_train = X_train.cuda()
X_test = X_test.cuda()
y_train = y_train.cuda()
y_test = y_test.cuda()

# Initialize the model, loss function, and optimizer
model = ModularAdditionNN(args.M, args.hidden_size, [args.M], d=args.d, embedding_layer_fixed=args.emb_fixed)
model = model.cuda()

criterion = nn.CrossEntropyLoss().cuda()
# optimizer = optim.SGD(model.parameters(), lr=3, momentum=0.9, weight_decay=2e-4)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=3e-4)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# Training loop
model.train()
for epoch in range(args.num_epochs):
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train)
    # loss = criterion(outputs, y_train)
    loss = 0
    for i, o in enumerate(outputs):
        loss = loss + criterion(o, y_train[:,i])
    
    # Backward and optimize
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {loss.item():.4f}')

        # Test the model
        train_accuracy = test_model(model, X_train, y_train)
        test_accuracy = test_model(model, X_test, y_test)

        print(f"Train Accuracy: {train_accuracy}")
        print(f"Test Accuracy: {test_accuracy}\n")

        model.train()

torch.save(model.state_dict(), "model.pt")
