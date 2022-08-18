import os
import torch
import random
import numpy as np
from PIL import Image
from scipy import stats
from math import inf
import torch.nn.functional as F
from numpy.testing import assert_array_almost_equal
from torchvision import datasets, transforms

class Dataset:

    def __init__(self,
                 dataset_name,
                 data_dir='./data',
                 num_samples=0,
                 noise_type=None,
                 noise_rate=None,
                 random_seed=1,
                 device=torch.device('cuda')
                 ):
        self.dataset_name = dataset_name
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.device = device
        self.random_seed = random_seed
        self.train_sampler = None
        self.test_sampler = None

        if self.dataset_name == "cifar10":
            cifar_mean = [0.4914, 0.4822, 0.4465]
            cifar_std = [0.2023, 0.1994, 0.2010]
            transform_pipe = [transforms.RandomHorizontalFlip(),
                              transforms.RandomCrop(32, 4),
                              transforms.ToTensor(),
                              transforms.Normalize(cifar_mean, cifar_std)]

            self.train_set = Cifar10(root=data_dir,
                                     train=True,
                                     transform=transforms.Compose(transform_pipe),
                                     download=True)

            self.test_set = Cifar10(root=data_dir,
                                    train=False,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(),
                                         transforms.Normalize(cifar_mean, cifar_std)]),
                                    download=True)
            self.num_classes = 10
            self.input_size = 32 * 32 * 3
            self.is_noisy = []
            if noise_type is not None:
                self.clean_labels = torch.tensor(self.train_set.cifar10.targets)
                train_noisy_labels_tensor, is_noisy_labels = self.make_labels_noisy()
                self.is_noisy = is_noisy_labels[:]
                self.train_set.cifar10.targets = train_noisy_labels_tensor.detach()

        elif self.dataset_name == "cifar100":
            cifar_mean = [0.507, 0.487, 0.441]
            cifar_std = [0.267, 0.256, 0.276]
            transform_pipe = [transforms.RandomHorizontalFlip(),
                              transforms.RandomCrop(32, 4),
                              transforms.ToTensor(),
                              transforms.Normalize(cifar_mean, cifar_std), ]
            self.train_set = Cifar100(root=data_dir,
                                      train=True,
                                      transform=transforms.Compose(transform_pipe),
                                      download=True)
            self.test_set = Cifar100(root=data_dir,
                                     train=False,
                                     transform=transforms.Compose([transforms.ToTensor(),
                                                                   transforms.Normalize(cifar_mean,
                                                                                        cifar_std)]),
                                     download=True)
            self.num_classes = 100
            self.input_size = 32 * 32 * 3
            if noise_type is not None:
                self.clean_labels = torch.tensor(self.train_set.cifar100.targets)
                train_noisy_labels_tensor, is_noisy_labels = self.make_labels_noisy()
                self.is_noisy = is_noisy_labels[:]
                self.train_set.cifar100.targets = train_noisy_labels_tensor.detach()

        elif self.dataset_name == 'clothing1m':
            self.num_classes = 14
            self.input_size = 224 * 224 * 3
            self.num_samples = num_samples
            # self.num_samples = 250000
            c1m_mean = [0.6959, 0.6537, 0.6371]
            c1m_std = [0.3113, 0.3192, 0.3214]

            self.transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            self.transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
            self.train_set = Clothing1M(data_dir, num_samples=self.num_samples, mode='train',
                                        transform=self.transform_train)
            self.train_sampler = None


            self.test_set = Clothing1M(data_dir, num_samples=self.num_samples, mode='test',
                                       transform=self.transform_test)

    # ------------------------------------------------------------------------------------------------------------------
    # Taken from https://github.com/xiaoboxia/CDR/blob/6665a8ba265f0f60291ed7775042575db05bed61/utils.py
    def make_labels_noisy(self):
        clean_labels_np = self.clean_labels.detach().numpy()
        clean_labels_np = clean_labels_np[:, np.newaxis]
        m = clean_labels_np.shape[0]
        noisy_labels = clean_labels_np.copy()

        is_noisy = m * [None]
        if self.noise_rate is None:
            raise ValueError("Noise rate needs to be specified ....")

        if self.noise_type == "symmetric":
            noise_matrix = self.compute_noise_transition_symmetric()

        elif self.noise_type == "instance":
            noise_matrix = self.compute_noise_transition_instance()
        
        elif self.noise_type == "pairflip":
            noise_matrix = self.compute_noise_transition_pairflip()


        print('Size of noise transition matrix: {}'.format(noise_matrix.shape))

        if self.noise_type == "symmetric" or self.noise_type == "pairflip":
            assert noise_matrix.shape[0] == noise_matrix.shape[1]
            assert np.max(clean_labels_np) < noise_matrix.shape[0]
            assert_array_almost_equal(noise_matrix.sum(axis=1), np.ones(noise_matrix.shape[1]))
            assert (noise_matrix >= 0.0).all()

            flipper = np.random.RandomState(self.random_seed)
            for idx in np.arange(m):
                i = clean_labels_np[idx]
                flipped = flipper.multinomial(1, noise_matrix[i, :][0], 1)[0]
                noisy_labels[idx] = np.where(flipped == 1)[0]
                is_noisy[idx] = (noisy_labels[idx] != i)[0]
        elif self.noise_type == "instance":
            l = [i for i in range(self.num_classes)]
            for idx in np.arange(m):
                noisy_labels[idx] = np.random.choice(l, p=noise_matrix[idx])
                is_noisy[idx] = (noisy_labels[idx] != clean_labels_np[idx])[0]

        # noise_or_not = (noisy_labels != clean_labels_np)
        actual_noise_rate = (noisy_labels != clean_labels_np).mean()
        assert actual_noise_rate > 0.0
        print('Actual_noise_rate : {}'.format(actual_noise_rate))
        return torch.tensor(np.squeeze(noisy_labels)), is_noisy

    # ------------------------------------------------------------------------------------------------------------------
    # Taken from https://github.com/xiaoboxia/CDR/blob/6665a8ba265f0f60291ed7775042575db05bed61/utils.py
    def compute_noise_transition_symmetric(self):
        noise_matrix = np.ones((self.num_classes, self.num_classes))
        noise_matrix = (self.noise_rate / (self.num_classes - 1)) * noise_matrix

        if self.noise_rate > 0.0:
            # 0 -> 1
            noise_matrix[0, 0] = 1. - self.noise_rate
            for i in range(1, self.num_classes - 1):
                noise_matrix[i, i] = 1. - self.noise_rate
            noise_matrix[self.num_classes - 1, self.num_classes - 1] = 1. - self.noise_rate
            # print(noise_matrix)
        return noise_matrix

    # ------------------------------------------------------------------------------------------------------------------
    # Taken from https://github.com/xiaoboxia/CDR/blob/6665a8ba265f0f60291ed7775042575db05bed61/tools.py
    def compute_noise_transition_instance(self):
        clean_labels = self.clean_labels
        norm_std = 0.1
        np.random.seed(int(self.random_seed))
        torch.manual_seed(int(self.random_seed))
        torch.cuda.manual_seed(int(self.random_seed))

        noise_matrix = []
        flip_distribution = stats.truncnorm((0 - self.noise_rate) / norm_std,
                                            (1 - self.noise_rate) / norm_std,
                                            loc=self.noise_rate,
                                            scale=norm_std)
        flip_rate = flip_distribution.rvs(clean_labels.shape[0])

        W = np.random.randn(self.num_classes, self.input_size, self.num_classes)
        W = torch.FloatTensor(W).to(self.device)
        for i, (image, label, _) in enumerate(self.train_set):
            # 1*m *  m*10 = 1*10 = A.size()
            image = image.detach().to(self.device)
            A = image.view(1, -1).mm(W[label]).squeeze(0)
            A[label] = -inf
            A = flip_rate[i] * F.softmax(A, dim=0)
            A[label] += 1 - flip_rate[i]
            noise_matrix.append(A)
        noise_matrix = torch.stack(noise_matrix, 0).cpu().numpy()
        return noise_matrix

    # ------------------------------------------------------------------------------------------------------------------
    #https://github.com/tmllab/PES/blob/54662382dca22f314911488d79711cffa7fbf1a0/common/NoisyUtil.py
    def compute_noise_transition_pairflip(self):

        noise_matrix = np.eye(self.num_classes)

        if self.noise_rate > 0.0:
            # 0 -> 1
            noise_matrix[0, 0], noise_matrix[0,1] = 1. - self.noise_rate, self.noise_rate
            for i in range(1, self.num_classes - 1):
                noise_matrix[i, i], noise_matrix[i, i+1] = 1. - self.noise_rate, self.noise_rate
            noise_matrix[self.num_classes - 1, self.num_classes - 1], noise_matrix[self.num_classes-1, 0] = 1. - self.noise_rate, self.noise_rate
            # print(noise_matrix)
        return noise_matrix


    # ------------------------------------------------------------------------------------------------------------------
    def split_into_batches(self, batch_size, train_sampler=None):
        if train_sampler is not None:
            self.train_dataloader = torch.utils.data.DataLoader(dataset=self.train_set,
                                                                batch_size=batch_size,
                                                                sampler=train_sampler,
                                                                drop_last=False)

        else:
            self.train_dataloader = torch.utils.data.DataLoader(dataset=self.train_set,
                                                                batch_size=batch_size,
                                                                shuffle=True,
                                                                drop_last=False)

        self.test_dataloader = torch.utils.data.DataLoader(dataset=self.test_set,
                                                           batch_size=batch_size,
                                                           shuffle=False)
        self.sample_rate = batch_size / len(self.train_set)
        return self.train_dataloader, self.test_dataloader

# -------------------------------------------------------------------------------------------------------------
# Taken from https://github.com/chenpf1025/IDN/blob/master/dataset.py
class Clothing1M(torch.utils.data.Dataset):
    def __init__(self, root, mode='train',
                 soft=False, target_prob=None,
                 transform=None, target_transform=None, num_classes=14, num_samples=0):
        self.root = root
        self.mode = mode
        self.soft = soft
        self.target_prob = target_prob
        self.transform = transform
        self.target_transform = target_transform

        train_labels_path = os.path.join(root, 'annotations/noisy_label_kv.txt')
        self.train_labels = self.file_reader(train_labels_path, labels=True)
        test_labels_path = os.path.join(root, 'annotations/clean_label_kv.txt')
        self.test_labels = self.file_reader(test_labels_path, labels=True)

        if self.mode == 'train':
            file_path = os.path.join(root, "annotations/noisy_train_key_list.txt")
            imgs = self.file_reader(file_path, labels=False)
            random.shuffle(imgs)
            class_num = torch.zeros(num_classes)
            self.imgs = []
            for impath in imgs:
                label = self.train_labels[impath]
                if class_num[label] < (num_samples / 14) and len(self.imgs) < num_samples:
                    self.imgs.append(impath)
                    class_num[label] += 1
            random.shuffle(self.imgs)
            print('Number of training samples : ' + str(class_num))

        if self.mode == 'val':
            file_path = os.path.join(root, "annotations/clean_val_key_list.txt")
            self.imgs = self.file_reader(file_path, labels=False)

        if self.mode == 'test':
            file_path = os.path.join(root, 'annotations/clean_test_key_list.txt')
            self.imgs = self.file_reader(file_path, labels=False)

        if not os.path.exists(file_path):
            raise RuntimeError('Dataset not found or not extracted.' +
                               'You can contact the author of Clothing1M for the download link. <Xiao, Tong, '
                               'et al. (2015). Learning from massive noisy labeled data for image classification>')

    def __getitem__(self, index):
        impath = self.imgs[index]
        if self.mode == 'train':
            target = self.train_labels[impath]
        else:
            target = self.test_labels[impath]
        img = Image.open(impath).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.soft:
            target_soft = self.target_prob[index]
            return img, target_soft, target
        else:
            return img, target, index

    def __len__(self):
        return len(self.imgs)

    def file_reader(self, path, labels=False):
        if labels:
            file_data = {}
        else:
            file_data = []
        with open(path, 'r') as rf:
            for line in rf.readlines():
                row = line.strip().split()
                img_path = self.root + '/' + row[0][7:]
                if labels:
                    file_data[img_path] = int(row[1])
                else:
                    file_data.append(img_path)

        return file_data

# ----------------------------------------------------------------------------------------------------------------
class Cifar10(Dataset):
    def __init__(self, root, train, transform, download):
        self.cifar10 = datasets.CIFAR10(root=root,
                                        download=download,
                                        train=train,
                                        transform=transform)
        self.is_noisy = []

    def __getitem__(self, index):
        data, target = self.cifar10[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10)


class Cifar100(Dataset):
    def __init__(self, root, train, transform, download):
        self.cifar100 = datasets.CIFAR100(root=root,
                                          download=download,
                                          train=train,
                                          transform=transform)

    def __getitem__(self, index):
        data, target = self.cifar100[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar100)
