import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
torch.manual_seed(0)

# Define the modular addition function
def modular_addition(x, y, mod):
    return (x + y) % mod
    # return (x * x - y * y + mod) % mod

# Generate dataset
# def generate_dataset(M):
#     data = []
#     for x in range(M):
#         for y in range(M):
#             z = modular_addition(x, y, M)
#             data.append((x, y, z))
#     return data

def generate_dataset2(M1, M2):
    data = []
    for x1 in range(M1):
        for y1 in range(M1):
            z1 = modular_addition(x1, y1, M1)
            for x2 in range(M2):
                for y2 in range(M2):
                    z2 = modular_addition(x2, y2, M2)
                    data.append(((x1, x2, y1, y2, z1, z2), (x1 * M2 + x2, y1 * M2 + y2, z1 * M2 + z2)))
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
    def __init__(self, input_size, hidden_size, output_sizes, d):
        super(ModularAdditionNN, self).__init__()
        self.embedding = nn.Linear(input_size, d)
        self.layer1 = nn.Linear(d, hidden_size)
        self.layer2 = nn.Linear(hidden_size, d)
        output_layers = [ nn.Linear(d, ss) for ss in output_sizes ]
        self.output_layers = nn.ModuleList(output_layers)
    
    def forward(self, x):
        x = torch.relu(self.layer1(self.embedding(x)))
        x = self.layer2(x)
        return [ l(x) for l in self.output_layers ]

# Hyperparameters

hidden_size = 512 
num_epochs = 2000
learning_rate = 0.02
test_size = 0.5

# Generate dataset
M1 = 23
M2 = 7

M = M1 * M2
dataset = generate_dataset2(M1, M2)
dataset_size = len(dataset)

# Prepare data for training and testing
r1 = 0.3
r2 = 0.4
X = torch.zeros(dataset_size, M)
# Use 
labels = torch.LongTensor(dataset_size)
labels2 = torch.LongTensor(dataset_size, 2)

for i, ((x1, x2, y1, y2, z1, z2), (x, y, z)) in enumerate(dataset):
    X[i, x] = r1
    X[i, y] = r2
    labels[i] = z
    labels2[i, 0] = z1
    labels2[i, 1] = z2

y = labels
y2 = labels2

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test, y2_train, y2_test = train_test_split(X, y, y2, test_size=test_size, random_state=0)

X_train = X_train.cuda()
X_test = X_test.cuda()
y_train = y_train.cuda()
y_test = y_test.cuda()

y2_train = y2_train.cuda()
y2_test = y2_test.cuda()

# Initialize the model, loss function, and optimizer
model = ModularAdditionNN(M, hidden_size, [M1, M2], d=32)
model = model.cuda()

criterion = nn.CrossEntropyLoss().cuda()
# optimizer = optim.SGD(model.parameters(), lr=0.6, momentum=0.9, weight_decay=2e-4)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=3e-4)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# Training loop
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train)
    # loss = criterion(outputs, y_train)
    loss = 0
    for i, o in enumerate(outputs):
        loss = loss + criterion(o, y2_train[:,i])
    
    # Backward and optimize
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
train_accuracy = test_model(model, X_train, y2_train)
test_accuracy = test_model(model, X_test, y2_test)

print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

torch.save(model.state_dict(), "model.pt")
