import torch
from torch.distributions import Normal
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    """Maps an (x_i, y_i) pair to a representation r_i.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    h_dim : int
        Dimension of hidden layer.

    r_dim : int
        Dimension of output representation r.
    """
    def __init__(self, x_dim, y_dim, h_dim, r_dim):
        super(Encoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim

        layers = [nn.Linear(x_dim + y_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, r_dim)]

        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        x : torch.Tensor
            Shape (batch_size, x_dim)

        y : torch.Tensor
            Shape (batch_size, y_dim)
        """
        # shouldn't along dim -1?
        input_pairs = torch.cat((x, y), dim=1)
        return self.input_to_hidden(input_pairs)


class MuSigmaEncoder(nn.Module):
    """
    Maps a representation r to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.

    Parameters
    ----------
    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.
    """
    def __init__(self, r_dim, z_dim):
        super(MuSigmaEncoder, self).__init__()

        self.r_dim = r_dim
        self.z_dim = z_dim

        self.r_to_hidden = nn.Linear(r_dim, r_dim)
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)

    def forward(self, r):
        """
        r : torch.Tensor
            Shape (batch_size, r_dim)
        """
        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return mu, sigma


class Decoder(nn.Module):
    """
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layer.

    y_dim : int
        Dimension of y values.
    """
    def __init__(self, x_dim, rep_dim, h_dim, y_dim, fixed_sigma):
        super(Decoder, self).__init__()

        self.x_dim = x_dim
        self.rep_dim = rep_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.fixed_sigma = fixed_sigma

        layers = [nn.Linear(x_dim + rep_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True)]

        self.xz_to_hidden = nn.Sequential(*layers)
        self.hidden_to_mu = nn.Linear(h_dim, y_dim)
        self.hidden_to_sigma = nn.Linear(h_dim, y_dim)

    def forward(self, x, rep):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        z : torch.Tensor
            Shape (batch_size, z_dim)

        Returns
        -------
        Returns mu and sigma for output distribution. Both have shape
        (batch_size, num_points, y_dim).
        """
        batch_size, num_points, _ = x.size()
        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.x_dim)
        rep_flat = rep.view(batch_size * num_points, self.rep_dim)
        # Input is concatenation of the representation with every row of x
        input_pairs = torch.cat((x_flat, rep_flat), dim=-1)
        hidden = self.xz_to_hidden(input_pairs)
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)
        # Reshape output into expected shape
        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        if self.fixed_sigma is None:
            sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        else:
            sigma = torch.Tensor(mu.shape)
            sigma.fill_(self.fixed_sigma)

        return mu, sigma




class NeuralProcess(nn.Module):
    """
    Implements Neural Process for functions of arbitrary dimensions.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layer in encoder and decoder.
    """
    def __init__(self, x_dim, y_dim, r_dim, z_dim, h_dim, fixed_sigma=None):
        super(NeuralProcess, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.fixed_sigma = fixed_sigma

        # Initialize networks
        self.xy_to_r = Encoder(x_dim, y_dim, h_dim, r_dim)
        self.r_to_mu_sigma = MuSigmaEncoder(r_dim, z_dim)
        self.xz_to_y = Decoder(x_dim, z_dim, h_dim, y_dim, fixed_sigma)

    def aggregate(self, r_i):
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.

        Parameters
        ----------
        r_i : torch.Tensor
            Shape (batch_size, num_points, r_dim)
        """
        return torch.mean(r_i, dim=1)

    def xy_to_mu_sigma(self, x, y):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        """
        batch_size, num_points, _ = x.size()
        # Flatten tensors, as encoder expects one dimensional inputs
        x_flat = x.view(batch_size * num_points, self.x_dim)
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        # Encode each point into a representation r_i
        r_i_flat = self.xy_to_r(x_flat, y_flat)
        # Reshape tensors into batches
        r_i = r_i_flat.view(batch_size, num_points, self.r_dim)
        # Aggregate representations r_i into a single representation r
        r = self.aggregate(r_i)
        # Return parameters of distribution
        return self.r_to_mu_sigma(r)

    def forward(self, x_context, y_context, x_target, y_target=None):
        """
        Given context pairs (x_context, y_context) and target points x_target,
        returns a distribution over target points y_target.

        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context, x_dim). Note that x_context is a
            subset of x_target.

        y_context : torch.Tensor
            Shape (batch_size, num_context, y_dim)

        x_target : torch.Tensor
            Shape (batch_size, num_target, x_dim)

        y_target : torch.Tensor or None
            Shape (batch_size, num_target, y_dim). Only used during training.

        Note
        ----
        We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        # Infer quantities from tensor dimensions
        batch_size, num_context, x_dim = x_context.size()
        _, num_target, _ = x_target.size()
        _, _, y_dim = y_context.size()

        if self.training:
            # Encode target and context (context needs to be encoded to
            # calculate kl term)
            mu_target, sigma_target = self.xy_to_mu_sigma(x_target, y_target)
            mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context)
            # Sample from encoded distribution using reparameterization trick
            q_target = Normal(mu_target, sigma_target)
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_target.rsample()
            # Repeat z, so it can be concatenated with every x. This changes shape
            # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
            z_sample = z_sample.unsqueeze(1).repeat(1, num_target, 1)
            # Get parameters of output distribution
            y_pred_mu, y_pred_sigma = self.xz_to_y(x_target, z_sample)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            return p_y_pred, q_target, q_context
        else:
            # At testing time, encode only context
            mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context)
            # Sample from distribution based on context
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_context.rsample()
            # Repeat z, so it can be concatenated with every x. This changes shape
            # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
            z_sample = z_sample.unsqueeze(1).repeat(1, num_target, 1)
            # Predict target points based on context
            y_pred_mu, y_pred_sigma = self.xz_to_y(x_target, z_sample)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)
            return p_y_pred

    def sample_z(self, x_context, y_context, num_target=1):

        mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context)
        # Sample from distribution based on context
        q_context = Normal(mu_context, sigma_context)
        z_sample = q_context.rsample()
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        z_sample = z_sample.unsqueeze(1).repeat(1, num_target, 1)
        return z_sample, q_context
