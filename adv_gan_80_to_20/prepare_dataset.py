from torchvision import datasets, transforms
import torch
import numpy as np

def load_dataset(dataset_name):

    if dataset_name == 'mnist':

        classes_to_mask = [0,5]
        num_classes = 10 - len(classes_to_mask)
        in_channels = 1

        train = datasets.MNIST('/workspace/dabs/data/natural_images/', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    #transforms.Normalize((0.1307,), (0.3081,))
                                    #transforms.Normalize((0.5,), (0.5,))
                                    ]))

        if classes_to_mask is not None:
            # 80% classes
            maskout = [ train.targets == ctmo for ctmo in classes_to_mask]
            maskout = ~ (torch.stack(maskout,axis=0).sum(0).bool())
            train.targets = train.targets[maskout]
            train.data = train.data[maskout]
            shift = torch.stack([ train.targets > ctmo for ctmo in classes_to_mask],axis=0).sum(0)
            train.targets = train.targets - shift


        test = datasets.MNIST('/workspace/dabs/data/natural_images/', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    #transforms.Normalize((0.1307,), (0.3081,))
                                    #transforms.Normalize((0.5,), (0.5,))
                                    ]))
        
        if classes_to_mask is not None:
            # 80% classes
            maskout = [ test.targets == ctmo for ctmo in classes_to_mask]
            maskout = ~ (torch.stack(maskout,axis=0).sum(0).bool())
            test.targets = test.targets[maskout]
            test.data = test.data[maskout]
            shift = torch.stack([ test.targets > ctmo for ctmo in classes_to_mask],axis=0).sum(0)
            test.targets = test.targets - shift


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

        classes_to_mask = [1,6]
        num_classes = 10 - len(classes_to_mask)
        in_channels = 3

        train = datasets.CIFAR10('/workspace/dabs/data/natural_images/CIFAR10/', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
                                    ]))

        if classes_to_mask is not None:
            # 80% classes
            targets = np.array(train.targets)
            maskout = [ targets == ctmo for ctmo in classes_to_mask]
            maskout = ~ (np.stack(maskout,axis=0).sum(0).astype(bool))
            targets = targets[maskout]
            train.data = train.data[maskout]
            shift = np.stack([ targets > ctmo for ctmo in classes_to_mask],axis=0).sum(0)
            train.targets = (targets - shift).tolist()

        test = datasets.CIFAR10('/workspace/dabs/data/natural_images/CIFAR10/', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
                                    ]))
        
        if classes_to_mask is not None:
            # 80% classes
            targets = np.array(test.targets)
            maskout = [ targets == ctmo for ctmo in classes_to_mask]
            maskout = ~ (np.stack(maskout,axis=0).sum(0).astype(bool))
            targets = targets[maskout]
            test.data = test.data[maskout]
            shift = np.stack([ targets > ctmo for ctmo in classes_to_mask],axis=0).sum(0)
            test.targets = (targets - shift).tolist()

    return (train, test, in_channels, num_classes)
