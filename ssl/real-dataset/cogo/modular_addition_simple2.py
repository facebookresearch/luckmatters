from collections import defaultdict
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from modular_addition_load import process_one

import hydra
import itertools

import math
import logging
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)

import common_utils

def compute_diag_off_diag_avg(kernel):
    diagonal_avg = kernel.abs().diag().mean().item()
    off_diag_avg = (kernel.abs().sum() - kernel.abs().diag().sum()) / (kernel.shape[0] * (kernel.shape[0] - 1))
    return diagonal_avg, off_diag_avg

def fit_diag_11(kernel):
    # fit the diagonal to be 1 and the off-diagonal to be 1/10
    diagonal_mean = kernel.diag().mean().item()
    off_diag_mean = (kernel.sum() - kernel.diag().sum()) / (kernel.shape[0] * (kernel.shape[0] - 1))
    estimated_kernel = (diagonal_mean - off_diag_mean) * torch.eye(kernel.shape[0]).to(kernel.device) + off_diag_mean 
    return torch.norm(estimated_kernel - kernel) / torch.norm(kernel)

# Define the modular addition function
def modular_addition(xs, ys, mods):
    return tuple( (x + y) % mod for x, y, mod in zip(xs, ys, mods) )

def generate_dataset(M):
    if isinstance(M, int):
        orders = [M]
    else: 
        orders = [ int(v) for v in M.split("x") ]

    def flattern(xs):
        return sum( x * order for x, order in zip(xs[1:], orders[:-1]) ) + xs[0]

    data = []
    for x in itertools.product(*(range(v) for v in orders)):
        for y in itertools.product(*(range(v) for v in orders)):
            # 
            z = modular_addition(x, y, orders)
            # flattern them
            data.append((flattern(x), flattern(y), flattern(z)))
            # print(x, y, z, data[-1])
    return data, math.prod(orders)

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
            raise RuntimeError(f"Unknown loss! {loss_type}")

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

class StatsTracker:
    def __init__(self):
        self.stats = defaultdict(dict)

    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def update(self, **kwargs):
        # Convert any 0-order tensor to scalar
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor) and len(v.size()) == 0:
                v = v.item()
            self.stats[self.epoch][k] = v

    def save(self, filename):
        torch.save(self.stats, filename)

# Define the neural network model
class ModularAdditionNN(nn.Module):
    def __init__(self, M, hidden_size, activation="sqr", use_bn=False, inverse_mat_layer_reg=None):
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
        self.inverse_mat_layer_reg = inverse_mat_layer_reg
    
    def forward(self, x, Y=None, stats_tracker=None):
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

        self.x_before_layerc = x.clone()

        if stats_tracker is not None:
            x_zero_mean = x - x.mean(dim=0, keepdim=True)
            # Use simple matrix inversion
            kernel = x_zero_mean.t() @ x_zero_mean 
            diag_avg, off_diag_avg = compute_diag_off_diag_avg(kernel)
            log.warning(f"~F^t ~F: diag_avg = {diag_avg}, off_diag_avg = {off_diag_avg}, off_diag_avg / diag_avg = {off_diag_avg / diag_avg}")

            kernel2 = x @ x.t()
            dist_from_ideal = fit_diag_11(kernel2)
            # zero mean
            kernel2 = kernel2 - kernel2.mean(dim=0, keepdim=True)
            diag_avg2, off_diag_avg2 = compute_diag_off_diag_avg(kernel2)
            log.warning(f"F F^t: diag_avg = {diag_avg2}, off_diag_avg = {off_diag_avg2}, off_diag_avg / diag_avg = {off_diag_avg2 / diag_avg2}, distance from ideal, {dist_from_ideal}")

            # backpropagated gradient norm
            residual = Y - self.layerc(x)
            backprop_grad_norm = torch.norm(residual @ self.layerc.weight) 
            log.warning(f"Backpropagated gradient norm: {backprop_grad_norm}")

            stats_tracker.update(**{
                "~F^t~F_off_diag_avg": off_diag_avg,
                "~F^t~F_diag_avg": diag_avg,
                "FF^t_dist_from_ideal": dist_from_ideal,
                "FF^t_off_diag_avg": off_diag_avg2,
                "FF^t_diag_avg": diag_avg2,
                "dF_norm": backprop_grad_norm,
            })

        if self.inverse_mat_layer_reg is not None and Y is not None:
            use_svd = True
            update_weightc = True
            with torch.no_grad():
                # Compute the matrix that maps input to target
                # X [bs, d]
                # Y [bs, d_out]
                # we want to find W so that X W = Y, where W = [d, d_out] 

                if use_svd:
                    # Compute the SVD of input 
                    U, s, Vt = torch.linalg.svd(x, full_matrices=False)
                    # Then we invert to get W.  
                    # 
                    # self.layerc.weight[:] = (Vt.t() @ ((U.t() @ Y) / (s[:,None] + self.inverse_mat_layer_reg))).t()
                    reg_diag = s / (s.pow(2) + self.inverse_mat_layer_reg)
                    log.warning(f"Using SVD, singular value [min, max] are {s.min(), s.max()}, inverse_mat_layer_reg is {self.inverse_mat_layer_reg}")
                    if update_weightc:
                        self.layerc.weight[:] = (Vt.t() @ ((U.t() @ Y) * reg_diag[:,None])).t()
                else:
                    kernel = x.t() @ x
                    # Check if the kernel scale is the same as the self.inverse_mat_layer_reg
                    kernel_scale = kernel.diag().mean().item()
                    log.warning(f"Kernel scale is {kernel_scale}, inverse_mat_layer_reg is {self.inverse_mat_layer_reg}")
                    if update_weightc:
                        self.layerc.weight[:] = (torch.linalg.inv(kernel + self.inverse_mat_layer_reg * torch.eye(x.shape[1]).to(x.device)) @ x.t() @ Y).t()

        return [self.layerc(x)]

    def normalize(self):
        with torch.no_grad():
            self.layera.weight[:] -= self.layera.weight.mean(dim=1, keepdim=True) 
            self.layerb.weight[:] -= self.layerb.weight.mean(dim=1, keepdim=True) 
            self.layerc.weight[:] -= self.layerc.weight.mean(dim=0, keepdim=True) 

    def scale_down_top(self):
        # Scale down the top layer
        with torch.no_grad():
            cnorm = self.layerc.weight.norm() 
            if cnorm > 1:
                self.layerc.weight[:] /= cnorm


@hydra.main(config_path="config", config_name="dyn_madd.yaml")
def main(args):
    # Set random seed for reproducibility
    log.info(common_utils.print_info(args))
    common_utils.set_all_seeds(args.seed)
    # torch.manual_seed(args.seed)

    # Generate dataset
    dataset, group_order = generate_dataset(args.M)
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

    if args.load_dataset_split is not None:
        # Load dataset
        data = torch.load(args.load_dataset_split)
        train_indices = data["train_indices"]
        test_indices = data["test_indices"]

        X_train = X[train_indices, :]
        y_train = y[train_indices]

        X_test = X[test_indices, :]
        y_test = y[test_indices]
    
    else:
        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

    X_train = X_train.cuda()
    X_test = X_test.cuda()
    y_train = y_train.cuda()
    y_test = y_test.cuda()

    # Initialize the model, loss function, and optimizer
    if args.set_weight_reg is not None:
        assert args.loss_func == "mse", "only MSE loss can use set_weight_reg != None" 

    model = ModularAdditionNN(group_order, args.hidden_size, 
                              activation=args.activation, 
                              use_bn=args.use_bn, 
                              inverse_mat_layer_reg=args.set_weight_reg)

    if args.load_initial_layerab is not None:
        state_dict = torch.load(args.load_initial_layerab)
        with torch.no_grad():
            model.layera.weight[:] = state_dict["weight"][:, :group_order]
            model.layerb.weight[:] = state_dict["weight"][:, group_order:]

    model = model.cuda()

    if args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Unknown optimizer! {args.optim}")

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=3e-4)

    results = []

    # Get a one hot y_train
    Y_train = F.one_hot(y_train.squeeze())
    Y_train = Y_train - 1.0 / group_order

    stats_tracker = StatsTracker()

    # Training loop
    for epoch in range(args.num_epochs):
        stats_tracker.set_epoch(epoch)
        # Test the model
        train_accuracies, train_loss = test_model(model, X_train, y_train, args.loss_func)
        test_accuracies, test_loss = test_model(model, X_test, y_test, args.loss_func)

        train_acc = train_accuracies[0]
        test_acc = test_accuracies[0]

        log.info(f"Train Accuracy/Loss: {train_acc}/{train_loss}")
        log.info(f"Test Accuracy/Loss: {test_acc}/{test_loss}\n")

        stats_tracker.update(**{
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_loss": train_loss,
            "test_loss": test_loss,
        })

        if args.save_interval is not None and (epoch % args.save_interval == 0 or epoch < args.init_save_range):
            results.append(dict(epoch=epoch, train_acc=train_acc, test_acc=test_acc, train_loss=train_loss, test_loss=test_loss))

            filename = f"model{epoch:05}_train{train_acc:.2f}_loss{train_loss:.4f}_test{test_acc:.2f}_loss{test_loss:.4f}.pt" 

            data = dict(model=model.state_dict(), results=results) 

            torch.save(data, filename)

        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train, Y=Y_train, stats_tracker=stats_tracker)

        # loss = criterion(outputs, y_train)
        loss = compute_loss(outputs, y_train, args.loss_func)
        
        # Backward and optimize
        loss.backward()

        # if epoch % 100 == 0:
        #     with torch.no_grad():
        #         print(model.layera.weight.grad.norm())
        #         print(model.layerb.weight.grad.norm())
        # import pdb
        # pdb.set_trace()

        optimizer.step()

        if args.normalize:
            model.normalize()

        if args.scale_down_top:
            model.scale_down_top()

        if epoch % args.eval_interval == 0:
            log.info(f'Epoch [{epoch}/{args.num_epochs}], Loss: {loss.item():.4f}')

    # save the stats_tracker
    stats_tracker.save("stats_tracker.pt")

    if args.post_process:
        # Process the data and save to a final file.
        log.info("Post-Processing data ...")
        entry = process_one(os.getcwd())

        log.info("Saving ... ")
        torch.save(entry, "./data.pth")

    print(os.getcwd())

if __name__ == '__main__':
    main()
