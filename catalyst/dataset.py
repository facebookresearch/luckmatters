import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data.dataset import Dataset

class RandomDataset(Dataset):
    def __init__(self, N, d, std, noise_type="gaussian"):
        super(RandomDataset, self).__init__()
        self.d = d
        self.std = std
        self.N = N
        self.noise_type = noise_type
        self.regenerate()

    def regenerate(self):
        self.x = torch.FloatTensor(self.N, *self.d)
        if self.noise_type == "gaussian":
            self.x.normal_(0, std=self.std) 
        elif self.noise_type == "uniform":
            self.x.uniform_(-self.std / 2, self.std / 2)
        else:
            raise NotImplementedError(f"Unknown noise type: {self.noise_type}")

    def __getitem__(self, idx):
        return self.x[idx], -1

    def __len__(self):
        return self.N

def init_dataset(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,), (0.5,))]) 

    transform_cifar10_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_cifar10_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == "gaussian" or args.dataset == "uniform":
        if args.use_cnn:
            d = (1, 16, 16)
        else:
            d = (args.data_d,)
        d_output = 100
        train_dataset = RandomDataset(args.random_dataset_size, d, args.data_std, noise_type=args.dataset)
        eval_dataset = RandomDataset(10240, d, args.data_std, noise_type=args.dataset)

    elif args.dataset == "mnist":
        train_dataset = datasets.MNIST(
                root=args.data_dir, train=True, download=True, 
                transform=transform)

        eval_dataset = datasets.MNIST(
                root=args.data_dir, train=False, download=True, 
                transform=transform)

        d = (1, 28, 28)
        d_output = 10

    elif args.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
                root=args.data_dir, train=True, download=True, 
                transform=transform_cifar10_train)

        eval_dataset = datasets.CIFAR10(
                root=args.data_dir, train=False, download=True, 
                transform=transform_cifar10_test)

        if not args.use_cnn:
            d = (3 * 32 * 32, )
        else: 
            d = (3, 32, 32)
        d_output = 10

    else:
        raise NotImplementedError(f"The dataset {args.dataset} is not implemented!")

    return d, d_output, train_dataset, eval_dataset
