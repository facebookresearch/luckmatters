from torchvision.transforms import transforms
from data.gaussian_blur import GaussianBlur


def get_simclr_data_transforms_train(dataset_name, args):
    s = args["jitter"]

    if dataset_name == "stl10":
        input_shape = (96,96,3)
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        return transforms.Compose(
                [   
                    transforms.RandomResizedCrop(size=input_shape[0]),
                    transforms.RandomHorizontalFlip(p=args["prob_hflip"]),
                    transforms.RandomApply([color_jitter], p=0.8),
                    transforms.RandomGrayscale(p=args["prob_grayscale"]),
                    GaussianBlur(kernel_size=int(args["blur_sz"] * input_shape[0])),
                    transforms.ToTensor()
                ])

    elif dataset_name == "cifar10":
        return transforms.Compose(
                [
                    transforms.RandomResizedCrop(32),
                    transforms.RandomHorizontalFlip(p=args["prob_hflip"]),
                    transforms.RandomApply([transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)], p=0.8),
                    transforms.RandomGrayscale(p=args["prob_grayscale"]),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                ])
    else:
        raise RuntimeError(f"unknown dataset: {dataset_name}")

def get_simclr_data_transforms_test(dataset_name):
    if dataset_name == "cifar10":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    elif dataset_name == "stl10":
        return transforms.Compose([transforms.ToTensor()])
    else:
        raise RuntimeError(f"unknown dataset: {dataset_name}")
