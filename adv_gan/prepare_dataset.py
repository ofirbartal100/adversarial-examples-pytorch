from torchvision import datasets, transforms
import torch

def load_dataset(dataset_name):

    if dataset_name == 'mnist':

        num_classes = 10
        in_channels = 1

        train = datasets.MNIST('/workspace/dabs/data/natural_images/', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    #transforms.Normalize((0.1307,), (0.3081,))
                                    #transforms.Normalize((0.5,), (0.5,))
                                    ]))


        test = datasets.MNIST('/workspace/dabs/data/natural_images/', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    #transforms.Normalize((0.1307,), (0.3081,))
                                    #transforms.Normalize((0.5,), (0.5,))
                                    ]))
        mean = torch.tensor([0], dtype=torch.float32)
        std = torch.tensor([1], dtype=torch.float32)
        normalize_fn = transforms.Normalize(mean.tolist(), std.tolist())


    elif dataset_name == 'fmnist':

        num_classes = 10
        in_channels = 1

        train = datasets.FashionMNIST('../data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ]))

        test = datasets.FashionMNIST('../data', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ]))


    elif dataset_name == 'cifar10':

        num_classes = 10
        in_channels = 3

        train = datasets.CIFAR10('/workspace/dabs/data/natural_images/CIFAR10/', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.ToTensor(),
                                    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
                                    ]))

        test = datasets.CIFAR10('/workspace/dabs/data/natural_images/CIFAR10/', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
                                    ]))
        
        mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32)
        std = torch.tensor([0.2470, 0.2435, 0.2615], dtype=torch.float32)
        normalize_fn = transforms.Normalize(mean.tolist(), std.tolist())

    return (train, test, in_channels, num_classes , normalize_fn )
