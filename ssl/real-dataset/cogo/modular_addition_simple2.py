from ast import Tuple
from collections import defaultdict
import json
import os
from typing import Iterable, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from modular_addition_load import process_one
from sympy.combinatorics import Permutation, PermutationGroup
from sympy.combinatorics.named_groups import SymmetricGroup

import hydra
import itertools

import math
import logging
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)

import common_utils
from muon_opt import MuonEnhanced

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

def generate_modular_addition_dataset(M):
    if isinstance(M, int):
        orders = [M]
    else: 
        orders = [ int(v) for v in M.split("x") ]

    cum_orders = [orders[0]]
    for o in orders[1:]:
        cum_orders.append(cum_orders[-1] * o)

    def flattern(xs):
        return sum( x * order for x, order in zip(xs[1:], cum_orders[:-1]) ) + xs[0]

    data = []
    for x in itertools.product(*(range(v) for v in orders)):
        for y in itertools.product(*(range(v) for v in orders)):
            # 
            z = modular_addition(x, y, orders)
            # flattern them
            data.append((flattern(x), flattern(y), flattern(z)))
            # print(x, y, z, data[-1])
    return data, math.prod(orders)

def generate_perm_dataset(M):
    # M is the size of the symmetric group
    g = SymmetricGroup(M)
    elements = { perm : i for i, perm in enumerate(g.generate_schreier_sims()) }
    
    # do a permutation
    data = []
    for g1, i in elements.items():
        for g2, j in elements.items():
            k = elements[g1 * g2]
            data.append((i, j, k))

    return data, int(g.order())

def to_zero_based_table(table_1b: List[List[int]], index_base: int = 1) -> List[List[int]]:
    off = index_base
    return [[x - off for x in row] for row in table_1b]

def triples_from_table(tbl0: List[List[int]]) -> List[Tuple[int,int,int]]:
    n = len(tbl0)
    return [(i, j, tbl0[i][j]) for i in range(n) for j in range(n)]

def load_non_abelian_collection(M, dk_max=2):
    # M is a index. 
    # Load the non-abelian collection from the file
    # Get all non-abelian group with max_k d_k == dk_max
    # Get the current folder of this script
    json_file = "/private/home/yuandong/luckmatters/ssl/real-dataset/cogo/smallgroups_nonabelian_upto_128.jsonl"
    data = [ json.loads(line) for line in open(json_file, "r") ]

    # find rec so that rec["irrep_degrees"] == dk_max
    data = [ rec for rec in data if max(rec["irrep_degrees"]) == dk_max ]

    print(f"Found {len(data)} non-abelian groups with max_k d_k == {dk_max}")

    # Load the group, get the cayley table
    rec = data[M]
    tbl0 = to_zero_based_table(rec["table"], rec.get("index_base", 1))
    triples = triples_from_table(tbl0)

    rec["name"] = rec["name"].replace("\'\'", "")
    print(f"SmallGroup({rec['order']},{rec['smallgroup_id']})  name={rec['name']}")
    print(f"  num_irreps={rec['num_irreps']}  first irrep degrees={rec['irrep_degrees'][:10]}{'...' if len(rec['irrep_degrees'])>10 else ''}")
    print(f"  triples sample: {triples[:min(6, len(triples))]}  (total {len(triples)})")

    return triples, int(rec["order"])

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
    def __init__(self, M, hidden_size, activation="sqr", use_bn=False, inverse_mat_layer_reg=None, other_layers=0):
        super(ModularAdditionNN, self).__init__()
        self.embedding = nn.Embedding(M, M).requires_grad_(False)
        with torch.no_grad():
            self.embedding.weight[:] = torch.eye(M, M)
            
        self.W = nn.Linear(2*M, hidden_size, bias=False)
        self.other_layers = nn.ModuleList([ nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(other_layers) ])
        self.V = nn.Linear(hidden_size, M, bias=False)

        self.num_other_layers = other_layers

        if use_bn:
            self.bn = nn.BatchNorm1d(hidden_size)
            self.use_bn = True
        else:
            self.use_bn = False

        self.relu = nn.ReLU()
        self.activation = activation
        self.M = M
        self.inverse_mat_layer_reg = inverse_mat_layer_reg

        if self.activation == "sqr": 
            self.act_fun = lambda x: x.pow(2)
        elif self.activation == "relu":
            self.act_fun = lambda x: self.relu(x)
        elif self.activation == "silu":
            self.act_fun = lambda x: x * torch.sigmoid(x)
        else:
            raise RuntimeError(f"Unknown activation = {self.activation}")
    
    def forward(self, x, Y=None, stats_tracker=None):
        embed_concat = torch.concat([self.embedding(x[:,0]), self.embedding(x[:,1])], dim=1) 
        # x = torch.relu(self.layer1(x))
        x = self.W(embed_concat) 
        if self.use_bn:
            x = self.bn(x)

        x = self.act_fun(x)

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
            if self.num_other_layers == 0:
                residual = Y - self.V(x)
                backprop_grad_norm = torch.norm(residual @ self.V.weight) 
                log.warning(f"Backpropagated gradient norm: {backprop_grad_norm}")
            else:
                backprop_grad_norm = None

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
                    # self.V.weight[:] = (Vt.t() @ ((U.t() @ Y) / (s[:,None] + self.inverse_mat_layer_reg))).t()
                    reg_diag = s / (s.pow(2) + self.inverse_mat_layer_reg)
                    log.warning(f"Using SVD, singular value [min, max] are {s.min(), s.max()}, inverse_mat_layer_reg is {self.inverse_mat_layer_reg}")
                    if update_weightc:
                        self.V.weight[:] = (Vt.t() @ ((U.t() @ Y) * reg_diag[:,None])).t()
                else:
                    kernel = x.t() @ x
                    # Check if the kernel scale is the same as the self.inverse_mat_layer_reg
                    kernel_scale = kernel.diag().mean().item()
                    log.warning(f"Kernel scale is {kernel_scale}, inverse_mat_layer_reg is {self.inverse_mat_layer_reg}")
                    if update_weightc:
                        self.V.weight[:] = (torch.linalg.inv(kernel + self.inverse_mat_layer_reg * torch.eye(x.shape[1]).to(x.device)) @ x.t() @ Y).t()

        for layer in self.other_layers:
            x = x + layer(x)
            x = self.act_fun(x)

        return [self.V(x)]

    def normalize(self):
        with torch.no_grad():
            self.W.weight[:] -= self.W.weight.mean(dim=1, keepdim=True) 
            self.V.weight[:] -= self.V.weight.mean(dim=0, keepdim=True) 

    def scale_down_top(self):
        # Scale down the top layer
        with torch.no_grad():
            cnorm = self.V.weight.norm() 
            if cnorm > 1:
                self.V.weight[:] /= cnorm


@hydra.main(config_path="config", config_name="dyn_madd.yaml")
def main(args):
    # Set random seed for reproducibility
    log.info(common_utils.print_info(args))
    common_utils.set_all_seeds(args.seed)
    # torch.manual_seed(args.seed)

    # Generate dataset
    if args.group_type == "modular_addition":
        dataset, group_order = generate_modular_addition_dataset(args.M)
        scaling_law_correction = 1
    elif args.group_type == "sym":
        dataset, group_order = generate_perm_dataset(args.M)
        scaling_law_correction = 1
    elif args.group_type == "collection":
        # In this case, M becomes a index. 
        dataset, group_order = load_non_abelian_collection(args.M, dk_max=args.group_collection_max_dk)
        scaling_law_correction = args.group_collection_max_dk / 2 
    else:
        raise RuntimeError(f"Unknown group type = {args.group_type}")

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

    # compute the test_size if use_critical_ratio is true
    if args.use_critical_ratio:
        # critical ratio delta
        test_size = 1 - scaling_law_correction * math.log(group_order) / group_order * (args.critical_ratio_multiplier - args.critical_ratio_delta) 
        test_size = max(min(test_size, 1), 0)
        log.warning(f"Use critical ratio has set. test_size = {test_size}")
    else:
        test_size = args.test_size
        log.warning(f"Use specified test_size = {test_size}")

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=args.seed)

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
                              inverse_mat_layer_reg=args.set_weight_reg, 
                              other_layers=args.other_layers)

    model = model.cuda()

    if args.optim == "sgd":
        optimizers = [optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)]
    elif args.optim == "adam":
        optimizers = [optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)]
    elif args.optim == "muon":
        optimizers = [
            optim.Adam(model.V.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay),
            MuonEnhanced(
                model.W.parameters(), 
                beta1 = 0.9,
                beta2 = 0.99,
                lr=args.learning_rate, 
                weight_decay=args.weight_decay, 
                use_bf16=False, 
                nesterov=False, 
                update_rms_compensate=False,
                update_spectral_compensate=False 
            )
        ]
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
        
        [ opt.zero_grad() for opt in optimizers ]
        
        # Forward pass
        outputs = model(X_train, Y=Y_train, stats_tracker=stats_tracker)

        # loss = criterion(outputs, y_train)
        loss = compute_loss(outputs, y_train, args.loss_func)
        
        # Backward and optimize
        loss.backward()

        # if epoch % 100 == 0:
        #     with torch.no_grad():
        #         print(model.W.weight.grad.norm())
        # import pdb
        # pdb.set_trace()
        [ opt.step() for opt in optimizers ]

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
