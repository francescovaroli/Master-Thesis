import glob
import torch
import gpytorch
from math import pi
# from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import CosineKernel, RBFKernel, LinearKernel, PolynomialKernel, LCMKernel, PeriodicKernel, MaternKernel

class SineData(Dataset):
    """
    Dataset of functions f(x) = a * sin(x - b) where a and b are randomly
    sampled. The function is evaluated from -pi to pi.

    Parameters
    ----------
    amplitude_range : tuple of float
        Defines the range from which the amplitude (i.e. a) of the sine function
        is sampled.

    shift_range : tuple of float
        Defines the range from which the shift (i.e. b) of the sine function is
        sampled.

    num_samples : int
        Number of samples of the function contained in dataset.

    num_points : int
        Number of points at which to evaluate f(x) for x in [-pi, pi].
    """
    def __init__(self, amplitude_range=(-1., 1.), shift_range=(-.5, .5),
                 num_samples=1000, num_points=100):
        self.amplitude_range = amplitude_range
        self.shift_range = shift_range
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        # Generate data
        self.data = []
        a_min, a_max = amplitude_range
        b_min, b_max = shift_range
        for i in range(num_samples):
            # Sample random amplitude
            a = (a_max - a_min) * torch.rand(1) + a_min
            # Sample random shift
            b = (b_max - b_min) * torch.rand(1) + b_min
            # Shape (num_points, x_dim)
            x = torch.linspace(-pi, pi, num_points).unsqueeze(1).cuda()

            # Shape (num_points, y_dim)
            y = a * torch.sin(x - b)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


class ExactGPModel(gpytorch.models.ExactGP):
    """
    GP generator shared by all other classes

    """
    def __init__(self, train_x, train_y, likelihood, mean_module=None, kernel_module=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = mean_module
        self.covar_module = kernel_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultiGPData(Dataset):
    """
    Dataset of functions sampled from multiple Gaussian Processes

    Parameters
    ----------
    mean_list : list of strings
        all mean modules to be used for generating GPs

    kernel_list : list of sting
        all kernels to be used for generating GPs

    num_samples : int
        Number of samples of the function contained in dataset.

    amplitude_range : range of x values

    num_points : int
        Number of points at which to evaluate f(x) for x in amplitude_range.
    """

    def __init__(self, mean_list, kernel_list, num_points=100, num_samples=1000, amplitude_range=(-5., 5.)):
        self.mean_list = mean_list
        self.kernel_list = kernel_list
        self.num_config = len(mean_list) * len(kernel_list)
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1
        self.amplitude_range = amplitude_range
        self.data = []

        # initialize likelihood and model
        x = torch.linspace(self.amplitude_range[0], self.amplitude_range[1], num_points).unsqueeze(1)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        mean_dict = {'constant': ConstantMean(), 'linear': LinearMean(1)}
        kernel_dict = {'RBF': RBFKernel(), 'cosine': CosineKernel(), 'linear': LinearKernel(),
                       'periodic': PeriodicKernel(period_length=0.5),
                       'LCM': LCMKernel(base_kernels=[CosineKernel()], num_tasks=1),
                       'polynomial': PolynomialKernel(power=2),
                       'matern': MaternKernel()}

        # create a different GP from each possible configuration
        for mean in self.mean_list:
            for kernel in self.kernel_list:
                # evaluate GP on prior distribution
                with gpytorch.settings.prior_mode(True):
                    model = ExactGPModel(x, None, likelihood, mean_module=mean_dict[mean],
                                         kernel_module=kernel_dict[kernel])

                    gp = model(x)
                    # sample from current configuration
                    for i in range(num_samples//self.num_config+1):
                        y = gp.sample()
                        self.data.append((x, y.unsqueeze(1))) #+torch.randn(y.shape)*0))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


class GPData2D(Dataset):
    """
    Dataset of functions sampled from a 2 dimensional Gaussian Process

    Parameters
    ----------
    mean : str

    kernel : str

    grid_bounds : range on x1 and x2 dimension

    grid_size : num of grid cells (len(x1)*len(x2))

    num_samples : int
        Number of samples of the function contained in dataset.

    """

    def __init__(self, mean_name='constant', kernel_name='RBF', grid_bounds=[(-1, 1), (-1, 1)], grid_size=100,
                 num_samples=1000):

        self.mean = mean_name
        self.kernel = kernel_name
        self.num_samples = num_samples
        self.grid_bounds = grid_bounds
        self.grid_size = grid_size
        self.x_dim = 2  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        self.data = []

        # create grid
        grid = torch.zeros(grid_size, len(grid_bounds))
        for i in range(len(grid_bounds)):
            grid_diff = float(grid_bounds[i][1] - grid_bounds[i][0]) / (grid_size - 2)
            grid[:, i] = torch.linspace(grid_bounds[i][0] - grid_diff, grid_bounds[i][1] + grid_diff, grid_size)

        x = gpytorch.utils.grid.create_data_from_grid(grid)

        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        mean_dict = {'constant': ConstantMean()}
        kernel_dict = {'RBF': RBFKernel(), 'cosine': CosineKernel(), 'linear': LinearKernel(),
                       'periodic': PeriodicKernel(),
                       'LCM': LCMKernel(base_kernels=[CosineKernel()], num_tasks=1),
                       'polynomial': PolynomialKernel(power=3),
                       'matern': MaternKernel()}

        # evaluate GP on prior distribution
        with gpytorch.settings.prior_mode(True):
            model = ExactGPModel(x, None, likelihood, mean_module=mean_dict[self.mean],
                                 kernel_module=gpytorch.kernels.GridKernel(kernel_dict[self.kernel], grid=grid))
            gp = model(x)
            for i in range(num_samples):
                y = gp.sample()
                self.data.append((x, y.unsqueeze(1))) #+torch.randn(y.size())*0.2))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


def mnist(batch_size=16, size=28, path_to_data='../../mnist_data'):
    """MNIST dataloader.

    Parameters
    ----------
    batch_size : int

    size : int
        Size (height and width) of each image. Default is 28 for no resizing.

    path_to_data : string
        Path to MNIST data files.
    """
    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    train_data = datasets.MNIST(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST(path_to_data, train=False,
                               transform=all_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def celeba(batch_size=16, size=32, crop=89, path_to_data='../celeba_data',
           shuffle=True):
    """CelebA dataloader.

    Parameters
    ----------
    batch_size : int

    size : int
        Size (height and width) of each image.

    crop : int
        Size of center crop. This crop happens *before* the resizing.

    path_to_data : string
        Path to CelebA data files.
    """
    transform = transforms.Compose([
        transforms.CenterCrop(crop),
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    celeba_data = CelebADataset(path_to_data,
                                transform=transform)
    celeba_loader = DataLoader(celeba_data, batch_size=batch_size,
                               shuffle=shuffle)
    return celeba_loader


class CelebADataset(Dataset):
    """CelebA dataset."""
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        path_to_data : string
            Path to CelebA data files.

        subsample : int
            Only load every |subsample| number of images.

        transform : torchvision.transforms
            Torchvision transforms to be applied to each image.
        """
        self.img_paths = glob.glob(path_to_data + '/*.jpg')[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        sample = Image.open(sample_path)

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0
