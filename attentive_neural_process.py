import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


class LatentEncoder(nn.Module):
    """
     Maps a (x, y) to mu and sigma which will define the normal
     distribution from which we sample the latent variable z.

     Parameters
     ----------
     x_dim : int
         Dimension of x

     y_dim : int
         Dimension of y

     z_dim : int
         Dimension of latent variable z.
     """
    def __init__(self, x_dim, y_dim, s_dim, z_dim, attention):
        super(LatentEncoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.s_dim = s_dim
        self.z_dim = z_dim

        self.self_attention = attention
        layers = [nn.Linear(x_dim + y_dim, s_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(s_dim, s_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(s_dim, s_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(s_dim, s_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(s_dim, s_dim)]

        self.xy_to_hidden = nn.Sequential(*layers)
        self.hidden_to_sigma = nn.Linear(s_dim, z_dim)
        self.hidden_to_mu = nn.Linear(s_dim, z_dim)


    def forward(self, x, y):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        """

        batch_size, num_points, _ = x.size()
        x_flat = x.view(batch_size * num_points, self.x_dim)
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)

        # input through mlp
        input_pairs = torch.cat((x_flat, y_flat), dim=1)
        hidden_flat = self.xy_to_hidden(input_pairs)
        hidden = hidden_flat.view(batch_size, num_points, self.s_dim)

        # self attention
        #s_i = self.self_attention(x, x, hidden)
        s_i  = torch.relu(hidden)
        s = s_i.mean(dim=1)

        mu = self.hidden_to_mu(s)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(s))
        return mu, sigma


class DeterministicEncoder(nn.Module):

    """Maps an (x_i, y_i) pair to a representation r_i and then applies attention.

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
    def __init__(self, x_dim, y_dim, r_dim, attention):
        super(DeterministicEncoder, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        layers = [nn.Linear(x_dim + y_dim, r_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(r_dim, r_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(r_dim, r_dim)]

        self.xy_to_hidden = nn.Sequential(*layers)
        self.attention = attention

    def forward(self, context_x, context_y, target_x):
        """Encodes the inputs into one representation.

            Args:
              context_x: Tensor of shape [B,num_points,d_x]. For this 1D regression
                  task this corresponds to the x-values.
              context_y: Tensor of shape [B,num_points,d_y]. For this 1D regression
                  task this corresponds to the y-values.
              target_x: Tensor of shape [B,target_observations,d_x].
                  For this 1D regression task this corresponds to the x-values.

            Returns:
              The encoded representation. Tensor of shape [B,target_observations,d]
            """

        # Concatenate x and y along the filter axes
        encoder_input = torch.cat([context_x, context_y], dim=-1)

        # Pass final axis through MLP
        hidden = self.xy_to_hidden(encoder_input)

        # Apply attention
        hidden = self.attention(context_x, target_x, hidden)

        return hidden



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
    def __init__(self, x_dim, rep_dim, h_dim, y_dim):
        super(Decoder, self).__init__()

        self.x_dim = x_dim
        self.rep_dim = rep_dim
        self.h_dim = h_dim
        self.y_dim = y_dim


        layers = [nn.Linear(x_dim + rep_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
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
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        return mu, sigma



def uniform_attention(q, v):
    """Uniform attention. Equivalent to np.

    Args:
      q: queries. tensor of shape [B,m,d_k].
      v: values. tensor of shape [B,n,d_v].

    Returns:
      tensor of shape [B,m,d_v].
    """
    total_points = q.size()[1]
    rep = v.mean(dim=1, keepdims=True)  # [B,1,d_v]
    rep = rep.repeat([1, total_points, 1])
    return rep


def laplace_attention(q, k, v, scale, normalise):
    """Computes laplace exponential attention.

    Args:
      q: queries. tensor of shape [B,m,d_k].
      k: keys. tensor of shape [B,n,d_k].
      v: values. tensor of shape [B,n,d_v].
      scale: float that scales the L1 distance.
      normalise: Boolean that determines whether weights sum to 1.

    Returns:
      tensor of shape [B,m,d_v].
    """
    k = k.unsqueeze(1)  # [B,1,n,d_k]
    q = q.unsqueeze(2)  # [B,m,1,d_k]
    unnorm_weights = - torch.abs((k - q) / scale)  # [B,m,n,d_k]
    unnorm_weights = unnorm_weights.sum(dim=-1)  # [B,m,n]
    if normalise:
        weight_fn = nn.Softmax()
    else:
        weight_fn = lambda x: 1 + torch.tanh(x)
    weights = weight_fn(unnorm_weights)  # [B,m,n]
    rep = torch.bmm(weights, v)  # [B,m,d_v]
    return rep


def dot_product_attention(q, k, v, normalise):
    """Computes dot product attention.

    Args:
      q: queries. tensor of  shape [B,m,d_k].
      k: keys. tensor of shape [B,n,d_k].
      v: values. tensor of shape [B,n,d_v].
      normalise: Boolean that determines whether weights sum to 1.

    Returns:
      tensor of shape [B,m,d_v].
    """
    d_k = q.size()[-1]
    scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    unnorm_weights = torch.einsum('bjk,bik->bij', k, q) / scale  # [B,m,n]
    if normalise:
        weights = torch.softmax(unnorm_weights, dim=-1)
    else:
        weights = torch.sigmoid(unnorm_weights) # [B,m,n]
    weights2 = torch.nn.Softmax(unnorm_weights)  # previous implementation
    rep = torch.bmm(weights, v)  # [B,m,d_v]
    return rep


# def multihead_attention(q, k, v, num_heads=8):



class Attention(nn.Module):
    """The Attention module."""

    def __init__(self, rep, output_sizes, x_dim, att_type, scale=1., normalise=True,
                 num_heads=8):
        super(Attention, self).__init__()
        """Create attention module.

        Takes in context inputs, target inputs and
        representations of each context input/output pair
        to output an aggregated representation of the context data.
        Args:
          rep: transformation to apply to contexts before computing attention.
              One of: ['identity','mlp'].
          output_sizes: list of number of hidden units per layer of mlp.
              Used only if rep == 'mlp'.
          att_type: type of attention. One of the following:
              ['uniform','laplace','dot_product','multihead']
          scale: scale of attention.
          normalise: Boolean determining whether to:
              1. apply softmax to weights so that they sum to 1 across context pts or
              2. apply custom transformation to have weights in [0,1].
          num_heads: number of heads for multihead.
        """
        self._rep = rep
        self._output_sizes = output_sizes
        self._type = att_type
        self._scale = scale
        self._normalise = normalise
        self.mlp = nn.Linear(x_dim, output_sizes)
        if self._type == 'multihead':
            self._num_heads = num_heads

    def forward(self, x1, x2, r):
        """Apply attention to create aggregated representation of r.

        Args:
          x1: tensor of shape [B,n1,d_x].
          x2: tensor of shape [B,n2,d_x].
          r: tensor of shape [B,n1,d].

        Returns:
          tensor of shape [B,n2,d]

        Raises:
          NameError: The argument for rep/type was invalid.
        """
        if self._rep == 'identity':
            k, q = (x1, x2)
        elif self._rep == 'mlp':
            k = self.mlp(x1)
            q = self.mlp(x2)
        else:
            raise NameError("'rep' not among ['identity','mlp']")

        if self._type == 'uniform':
            rep = uniform_attention(q, r)
        elif self._type == 'laplace':
            rep = laplace_attention(q, k, r, self._scale, self._normalise)
        elif self._type == 'dot_product':
            rep = dot_product_attention(q, k, r, self._normalise)
        #elif self._type == 'multihead':
        #    rep = multihead_attention(q, k, r, self._num_heads)
        else:
            raise NameError(("'att_type' not among ['uniform','laplace','dot_product'"
                             ",'multihead']"))

        return rep

class AttentiveNeuralProcess(nn.Module):
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
    def __init__(self, x_dim, y_dim, r_dim, z_dim, h_dim, a_dim, att_type, rep='identity', scale=1.):
        super(AttentiveNeuralProcess, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.a_dim = a_dim

        # Initialize networks
        self.xy_to_z = LatentEncoder(x_dim, y_dim, r_dim, z_dim, Attention(rep, a_dim, x_dim, att_type, scale))
        self.xy_to_a = DeterministicEncoder(x_dim, y_dim, a_dim, Attention(rep, a_dim, x_dim, att_type, scale))
        self.xrep_to_y = Decoder(x_dim, z_dim+a_dim, h_dim, y_dim)

    def forward(self, x_context, y_context, x_target, y_target=None):
        """
        Given context pairs (x_context, y_context) and target points x_target,
        returns a distribution over target points y_target.

        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context, x_dim). Note that x_context is a
            subset of x_for i in range(60):
        z_sample = torch.randn((1, z_dim))
        mu, _ = neuralprocess.xrep_to_y(x_target, z_sample)
        plt.figure(3)
        plt.xlabel('x')
        plt.ylabel('means of y distribution')
        plt.title('Samples from trained prior')
        plt.plot(x_target.cpu().numpy()[0], mu.cpu().detach().numpy()[0],
                 c='b', alpha=0.5)target.

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
            mu_target, sigma_target = self.xy_to_z(x_target, y_target)
            mu_context, sigma_context = self.xy_to_z(x_context, y_context)
            # Sample from encoded distribution using reparameterization trick
            q_target = Normal(mu_target, sigma_target)
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_target.rsample()

            # Repeat z, so it can be concatenated with every x. This changes shape
            # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
            z_sample = z_sample.unsqueeze(1).repeat(1, num_target, 1)
            # Compute deterministic representation
            a_repr = self.xy_to_a(x_context, y_context, x_target)
            # Concatenate latent and deterministic representation
            representation = torch.cat([z_sample, a_repr], dim=-1)
            # Get parameters of output distribution
            y_pred_mu, y_pred_sigma = self.xrep_to_y(x_target, representation)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            return p_y_pred, q_target, q_context
        else:
            # At testing time, encode only context
            mu_context, sigma_context = self.xy_to_z(x_context, y_context)
            # Sample from distribution based on context
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_context.rsample()
            # Repeat z, so it can be concatenated with every x. This changes shape
            # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
            z_sample = z_sample.unsqueeze(1).repeat(1, num_target, 1)
            # Compute deterministic representation
            a_repr = self.xy_to_a(x_context, y_context, x_target)
            # Concatenate latent and deterministic representation
            representation = torch.cat([z_sample, a_repr], dim=-1)
            # Predict target points based on context
            y_pred_mu, y_pred_sigma = self.xrep_to_y(x_target, representation)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            return p_y_pred

