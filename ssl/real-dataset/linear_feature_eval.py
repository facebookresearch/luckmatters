import torch
import sys
import yaml
from torchvision import transforms, datasets
import torchvision
import numpy as np
import os
from sklearn import preprocessing
from torch.utils.data.dataloader import DataLoader
from models.resnet_base_network import ResNet18
from data.transforms import get_simclr_data_transforms_test
import hydra
import glob

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test):
    train = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)

    test = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False)
    return train_loader, test_loader


def get_features_from_encoder(encoder, loader, device):
    x_train = []
    y_train = []

    # get the features from the pre-trained model
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            feature_vector = encoder(x)
            x_train.extend(feature_vector)
            y_train.extend(y.numpy())

    x_train = torch.stack(x_train)
    y_train = torch.tensor(y_train)
    return x_train, y_train

import logging
log = logging.getLogger(__file__)

class Evaluator:
    def __init__(self, dataset, dataset_path, batch_size):
        data_transforms = get_simclr_data_transforms_test(dataset)
        if dataset == "stl10":
            train_dataset = datasets.STL10(dataset_path, split='train', download=False,
                                        transform=data_transforms)
            test_dataset = datasets.STL10(dataset_path, split='test', download=False,
                                        transform=data_transforms)
        elif dataset == "cifar10":
            train_dataset = datasets.CIFAR10(dataset_path, train=True, download=False,
                                        transform=data_transforms)
            test_dataset = datasets.CIFAR10(dataset_path, train=False, download=False,
                                        transform=data_transforms)
        else:
            raise RuntimeError(f"Unknown dataset! {args['dataset']}")

        log.info(f"Input shape: {train_dataset[0][0].shape}")

        self.stl_train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, drop_last=False, shuffle=True)
        self.stl_test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, drop_last=False, shuffle=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size

    def eval_model(self, encoder, save_path=None, num_epoch=50):
        remove_projection_head = True
        if remove_projection_head:
            output_feature_dim = encoder.feature_dim
            encoder = torch.nn.Sequential(*list(encoder.children())[:-1])
        else:
            output_feature_dim = encoder.projetion.net[-1].out_features

        device = self.device
        encoder = encoder.to(device)

        logreg = LogisticRegression(output_feature_dim, 10)
        logreg = logreg.to(device)

        encoder.eval()
        x_train, y_train = get_features_from_encoder(encoder, self.stl_train_loader, device)
        x_test, y_test = get_features_from_encoder(encoder, self.stl_test_loader, device)
        if save_path:
            np.savez(save_path, x_train.cpu().numpy(), x_test.cpu().numpy(), y_train.cpu().numpy(),
                    y_test.cpu().numpy())

        if len(x_train.shape) > 2:
            x_train = torch.mean(x_train, dim=[2, 3])
            x_test = torch.mean(x_test, dim=[2, 3])

        # log.info("Training data shape:", x_train.shape, y_train.shape)
        # log.info("Testing data shape:", x_test.shape, y_test.shape)

        x_train = x_train.cpu().numpy()
        x_test = x_test.cpu().numpy()
        scaler = preprocessing.StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train).astype(np.float32)
        x_test = scaler.transform(x_test).astype(np.float32)

        train_loader, test_loader = create_data_loaders_from_arrays(torch.from_numpy(x_train), y_train,
                                                                    torch.from_numpy(x_test), y_test)

        optimizer = torch.optim.Adam(logreg.parameters(), lr=3e-4, weight_decay=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        eval_every_n_epochs = 1

        best_acc = 0.
        for epoch in range(num_epoch):
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                logits = logreg(x)
                predictions = torch.argmax(logits, dim=1)

                loss = criterion(logits, y)

                loss.backward()
                optimizer.step()

            if epoch % eval_every_n_epochs == 0:
                correct = 0
                total = 0
                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)

                    logits = logreg(x)
                    predictions = torch.argmax(logits, dim=1)

                    total += y.size(0)
                    correct += (predictions == y).sum().item()

                acc = 100 * correct / total
                # log.info(f"Epoch {epoch} Testing accuracy: {acc}")
                if acc > best_acc:
                    best_acc = acc
        return best_acc


def linear_eval(dataset, dataset_path, batch_size, exp_name_list, load_epoch_list, default_network_params=None, default_trainer_params=None):
    evaluator = Evaluator(dataset, dataset_path, batch_size)

    result_dict = {}
    result_list = []

    for exp_name in exp_name_list:
        arg_file = os.path.join(f"{exp_name}", "args.pt") 
        if os.path.exists(arg_file): 
            args = torch.load(arg_file)
            network_params = args["network"]
            trainer_params = args["trainer"]
        else:
            network_params = default_network_params
            trainer_params = default_trainer_params

        log.info(network_params)

        if len(load_epoch_list) == 0:
            # Evaluate all models saved in the folder. 
            models = [ model for model in glob.glob(os.path.join(exp_name, "checkpoints", "model_*.pth")) ]
        else:
            models = [ f'{exp_name}/checkpoints/model_{str(epoch).zfill(3)}.pth' for epoch in load_epoch_list ]

        for load_path in models:
            save_path = None

            load_params = torch.load(
                os.path.join(load_path),
                map_location=torch.device(evaluator.device)
            )
            encoder = ResNet18(dataset=dataset, options=trainer_params["projector_params"], **network_params)
            encoder.load_state_dict(load_params['online_network_state_dict'])
            log.info("Load from {}.".format(load_path))

            best_acc = evaluator.eval_model(encoder, save_path=save_path)

            log.info(f"{load_path}: Best Acc {best_acc}")
            result_dict[load_path] = best_acc
            result_list.append(best_acc)

    for key in result_dict:
        log.info(f"{key}: {result_dict[key]}")
    log.info(f"mean acc: {np.mean(result_list)}, std: {np.std(result_list)}")
    return result_dict



@hydra.main("config/byol_config.yaml", strict=True)
def main(args):
    # root_dir = '/private/home/lantaoyu1/projects/PyTorch-BYOL/runs_09_03'
    # root_dir = '/checkpoint/lantaoyu1/PyTorch-BYOL/runs'

    # exp_name_list = [f"OriginalBN-RandomTargetInit_{i}" for i in range(5)]
    # exp_name_list = ['byol-FixBN']
    # exp_name_list = [f"ZeroMean-Std_seed-0_reinit-{i}" for i in [1, 2, 3, 4, 5, 10, 15, 20]]

    default_network_params = dict(args.network)
    default_trainer_params = dict(args.trainer)
    exp_name_list = args.test.exp_name_list.split(",")
    load_epoch_list = args.test.load_epoch_list
    linear_eval(args.dataset_path, args.test.batch_size, exp_name_list, load_epoch_list,
                default_network_params=default_network_params, 
                default_trainer_params=default_trainer_params)

if __name__ == "__main__":
    main()


'''
BYOL-AE
000: 45.86
010: 58.15
020: 68.26
030: 71.3
040: 72.76
050: 73.35
060: 75.23
070: 75.18

BYOL
000: 43.9
010: 61.61
020: 66.55
030: 68.53
040: 70.08
050: 71.23
060: 72.95
070: 72.15
'''

"""
# Epoch 90
byol-ZeroMean-Std: 77.9875
byol-ZeroMean-StdDetach: 70.5125
byol-Std: 69.4625
byol-ZeroMean: 65.7125
byol-ZeroMeanDetach: 24.0875
byol-StdDetach: 50.325
byol-ZeroMeanDetach-Std: 22.75
byol-ZeroMeanDetach-StdDetach: 44.6375

# Epoch 40
byol-ZeroMean-Std: 73.225
byol-ZeroMean-StdDetach: 72.575
byol-Std: 62.9875
byol-ZeroMean: 59.9125
byol-ZeroMeanDetach: 31.375
byol-StdDetach: 55.0
byol-ZeroMeanDetach-Std: 40.325
byol-ZeroMeanDetach-StdDetach: 35.35
"""
