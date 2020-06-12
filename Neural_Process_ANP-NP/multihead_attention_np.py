import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.nn import Linear
import math


class MultiheadAttention(nn.Module):
    """
    Multihead attention mechanism (dot attention)
    """

    def __init__(self, num_hidden_k):
        """
        :param num_hidden_k: dimension of hidden
        """
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query):
        # Get attention score
        attn = torch.bmm(query, key.transpose(1, 2))  # (heads x N_t x H_head)(heads x H_head x N_c) --> heads x N_t x N_c
        attn = attn / math.sqrt(self.num_hidden_k)

        attn = torch.softmax(attn, dim=-1)

        # Dropout
        attn = self.attn_dropout(attn)

        # Get Context Vector
        result = torch.bmm(attn, value)  # (heads x N_t x N_c)(heads x N_c x H_heads) --> (heads x N_t x H_heads)

        return result, attn


class Attention(nn.Module):
    """
    Attention Network
    """

    def __init__(self, num_hidden, h=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads
        """
        super(Attention, self).__init__()
        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h

        self.multihead = MultiheadAttention(self.num_hidden_per_attn)

        self.residual_dropout = nn.Dropout(p=0.1)

        self.final_linear = Linear(num_hidden * 2, num_hidden)

        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, key, value, query):
        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        residual = query

        # Make multihead  (...) x H_tot --> (...) x heads x H_head --> heads x N x H_head
        key = key.view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        value = value.view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        query = query.view(batch_size, seq_q, self.h, self.num_hidden_per_attn)
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.num_hidden_per_attn)

        # Get context vector
        result, attns = self.multihead(key, value, query)

        # Concatenate all multihead context vector
        result = result.view(self.h, batch_size, seq_q, self.num_hidden_per_attn)  # (heads x B x N_t x H_heads)
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)  # (B x N_t x H_tot)

        # Concatenate context vector with input (most important)
        result = torch.cat([residual, result], dim=-1)  # (B x N_t x (H_tot+r_dim))

        # Final linear
        result = self.final_linear(result)

        # Residual dropout & connection
        result = self.residual_dropout(result)
        result = result + residual

        # Layer normalization
        result = self.layer_norm(result)
        return result, attns


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
    def __init__(self, x_dim, y_dim, s_dim, z_dim, attention=None):
        super(LatentEncoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.s_dim = s_dim
        self.z_dim = z_dim

        self.self_attentions = nn.ModuleList([attention for _ in range(1)]) if attention is not None else None
        layers = [Linear(x_dim + y_dim, s_dim),
                  nn.ReLU(inplace=True),
                  Linear(s_dim, s_dim)]

        self.xy_to_hidden = nn.Sequential(*layers)
        self.hidden_to_sigma = Linear(s_dim, z_dim)
        self.hidden_to_mu = Linear(s_dim, z_dim)

        self.input_projection = Linear(x_dim + y_dim, s_dim)


    def forward(self, x, y):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        """

        batch_size, num_points, _ = x.size()
        if self.self_attentions is None:
            x_flat = x.view(batch_size * num_points, self.x_dim)
            y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
            # input through mlp
            input_pairs = torch.cat((x_flat, y_flat), dim=1)
            hidden_flat = self.xy_to_hidden(input_pairs)
            encoder_input = hidden_flat.view(batch_size, num_points, self.s_dim)
        else:
            input_pairs = torch.cat((x, y), dim=-1)
            encoder_input = self.input_projection(input_pairs)
            for attention in self.self_attentions:
                encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)

        s_i = torch.relu(encoder_input)
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
        layers = [Linear(x_dim + y_dim, r_dim),
                  nn.ReLU(inplace=True),
                  Linear(r_dim, r_dim)]

        self.xy_to_hidden = nn.Sequential(*layers)
        self.cross_attentions = nn.ModuleList([attention for _ in range(1)])
        self.context_projection = Linear(x_dim, r_dim)
        self.target_projection = Linear(x_dim, r_dim)

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
        encoder_input = torch.cat([context_x, context_y], dim=-1)  # B x N x (Xd+Yd)

        # Pass final axis through MLP
        encoder_input = self.xy_to_hidden(encoder_input)

        # Apply projection
        query = self.target_projection(target_x)
        keys = self.context_projection(context_x)

        # cross attention layer
        for attention in self.cross_attentions:
            query, _ = attention(keys, encoder_input, query)

        return query

    def get_input_key(self, context_x, context_y):
        # Concatenate x and y along the filter axes
        encoder_input = torch.cat([context_x, context_y], dim=-1)

        # Pass final axis through MLP
        encoder_input = self.xy_to_hidden(encoder_input)
        keys = self.context_projection(context_x)

        return encoder_input, keys

    def get_repr(self, encoder_input, keys, target_x):
        query = self.target_projection(target_x)

        # cross attention layer
        for attention in self.cross_attentions:
            query, _ = attention(keys, encoder_input, query)

        return query

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
    def __init__(self, x_dim, rep_dim, h_dim, y_dim, fixed_sigma, min_sigma):
        super(Decoder, self).__init__()

        self.x_dim = x_dim
        self.rep_dim = rep_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.fixed_sigma = fixed_sigma
        self.min_sigma = min_sigma

        layers = [Linear(x_dim + rep_dim, h_dim),
                  nn.ReLU(inplace=True),
                  Linear(h_dim, h_dim),
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
            sigma = self.min_sigma + (1 - self.min_sigma) * F.softplus(pre_sigma)
        else:
            sigma = torch.Tensor(mu.shape)
            sigma.fill_(self.fixed_sigma)
        return mu, sigma



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
    def __init__(self, x_dim, y_dim, r_dim, z_dim, h_dim, a_dim, use_self_att=True, fixed_sigma=None, min_sigma=0.1):
        super(AttentiveNeuralProcess, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.a_dim = a_dim

        # Initialize networks
        self_att = Attention(r_dim) if use_self_att else None
        self.xy_to_z = LatentEncoder(x_dim, y_dim, r_dim, z_dim, )
        self.xy_to_a = DeterministicEncoder(x_dim, y_dim, a_dim, Attention(a_dim))
        self.xz_to_y = Decoder(x_dim, z_dim+a_dim, h_dim, y_dim, fixed_sigma, min_sigma)
        self.id = 'ANP'

    def forward(self, x_context, y_context, x_target, y_target=None):
        """
        Given context pairs (x_context, y_context) and target points x_target,
        returns a distribution over target points y_target.

        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context, x_dim). Note that x_context is a
            subset of x_for i in range(60):

        y_context : torch.Tensor
            Shape (batch_size, num_context, y_dim)

        x_target : torch.Tensor
            Shape (batch_size, num_target, x_dim)

        y_target : torch.Tensor or None
            Shape (batch_size, num_target, y_dim). Only used during training.

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
            a_repr = self.xy_to_a(x_context, y_context, x_target)  # B x N_t x a_dim
            # Concatenate latent and deterministic representation
            representation = torch.cat([z_sample, a_repr], dim=-1)
            # Get parameters of output distribution
            y_pred_mu, y_pred_sigma = self.xz_to_y(x_target, representation)  # B x N_t x (h_dim + a_dim)
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
            y_pred_mu, y_pred_sigma = self.xz_to_y(x_target, representation)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            return p_y_pred

    def sample_z(self, x_context, y_context, num_target=1):

        mu_context, sigma_context = self.xy_to_z(x_context, y_context)
        # Sample from distribution based on context
        q_context = Normal(mu_context, sigma_context)
        z_sample = q_context.rsample()
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        z_sample = z_sample.unsqueeze(1).repeat(1, num_target, 1)
        return z_sample, q_context

