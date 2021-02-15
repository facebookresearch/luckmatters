import os
import sys

import torch
import yaml
from torchvision import datasets
from data.multi_view_data_injector import MultiViewDataInjector
from data.transforms import get_simclr_data_transforms_train, get_simclr_data_transforms_test
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18
from byol_trainer import BYOLTrainer
from simclr_trainer import SimCLRTrainer
import argparse
import os
import hydra
from linear_feature_eval import linear_eval, Evaluator

sys.path.append("../")
import common_utils

import logging
log = logging.getLogger(__file__)

def hydra2dict(args):
    if args.__class__.__name__ != 'DictConfig':
        return args
        
    args = dict(args)
    for k in args.keys():
        args[k] = hydra2dict(args[k])

    return args

@hydra.main("config/byol_config.yaml")
def main(args):
    log.info("Command line: \n\n" + common_utils.pretty_print_cmd(sys.argv))
    log.info(f"Working dir: {os.getcwd()}")
    log.info("\n" + common_utils.get_git_hash())
    log.info("\n" + common_utils.get_git_diffs())

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    torch.manual_seed(args.seed)
    log.info(args.pretty())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f"Training with: {device}")

    data_transform = get_simclr_data_transforms_train(args['dataset'])
    data_transform_identity = get_simclr_data_transforms_test(args['dataset'])

    if args["dataset"] == "stl10":
        train_dataset = datasets.STL10(args.dataset_path, split='train+unlabeled', download=True,
                                    transform=MultiViewDataInjector([data_transform, data_transform, data_transform_identity]))
    elif args["dataset"] == "cifar10":
        train_dataset = datasets.CIFAR10(args.dataset_path, train=True, download=True,
                                    transform=MultiViewDataInjector([data_transform, data_transform, data_transform_identity]))
    else:
        raise RuntimeError(f"Unknown dataset! {args['dataset']}")

    args = hydra2dict(args)
    train_params = args["trainer"]
    if train_params["projector_same_as_predictor"]:
        train_params["projector_params"] = train_params["predictor_params"]

    # online network
    online_network = ResNet18(dataset=args["dataset"], options=train_params["projector_params"], **args['network']).to(device)
    if torch.cuda.device_count() > 1:
        online_network = torch.nn.parallel.DataParallel(online_network)

    pretrained_path = args['network']['pretrained_path']
    if pretrained_path:
        try:
            load_params = torch.load(pretrained_path, map_location=torch.device(device))
            online_network.load_state_dict(load_params['online_network_state_dict'])
            online_network.load_state_dict(load_params)
            log.info("Load from {}.".format(pretrained_path))
        except FileNotFoundError:
            log.info("Pre-trained weights not found. Training from scratch.")

    # predictor network
    if train_params["has_predictor"] and args["method"] == "byol":
        predictor = MLPHead(in_channels=args['network']['projection_head']['projection_size'],
                            **args['network']['predictor_head'], options=train_params["predictor_params"]).to(device)
        if torch.cuda.device_count() > 1:
            predictor = torch.nn.parallel.DataParallel(predictor)
    else:
        predictor = None

    # target encoder
    target_network = ResNet18(dataset=args["dataset"], options=train_params["projector_params"], **args['network']).to(device)
    if torch.cuda.device_count() > 1:
        target_network = torch.nn.parallel.DataParallel(target_network)

    params = online_network.parameters()

    # Save network and parameters.
    torch.save(args, "args.pt")

    if args["eval_after_each_epoch"]: 
        evaluator = Evaluator(args["dataset"], args["dataset_path"], args["test"]["batch_size"]) 
    else:
        evaluator = None

    if args["use_optimizer"] == "adam":
        optimizer = torch.optim.Adam(params, lr=args['optimizer']['params']["lr"], weight_decay=args["optimizer"]["params"]['weight_decay'])
    elif args["use_optimizer"] == "sgd":
        optimizer = torch.optim.SGD(params, **args['optimizer']['params'])
    else:
        raise RuntimeError(f"Unknown optimizer! {args['use_optimizer']}")

    if args["predictor_optimizer_same"]:
        args["predictor_optimizer"] = args["optimizer"]

    if predictor and train_params["train_predictor"]:
       predictor_optimizer = torch.optim.SGD(predictor.parameters(), **args['predictor_optimizer']['params'])

    ## SimCLR scheduler
    if args["method"] == "simclr":
        trainer = SimCLRTrainer(log_dir="./", model=online_network, optimizer=optimizer, evaluator=evaluator, device=device, params=args["trainer"])
    elif args["method"] == "byol":
        trainer = BYOLTrainer(log_dir="./",
                              online_network=online_network,
                              target_network=target_network,
                              optimizer=optimizer,
                              predictor_optimizer=predictor_optimizer,
                              predictor=predictor,
                              device=device,
                              evaluator=evaluator,
                              **args['trainer'])
    else:
        raise RuntimeError(f'Unknown method {args["method"]}')

    trainer.train(train_dataset)

    if not args["eval_after_each_epoch"]:
        result_eval = linear_eval(args["dataset"], args["dataset_path"], args["test"]["batch_size"], ["./"], [])
        torch.save(result_eval, "eval.pth")

if __name__ == '__main__':
    main()
