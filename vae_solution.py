import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
  def __init__(self, nc, nef, nz, isize, device):
    super(Encoder, self).__init__()

    # Device
    self.device = device

    # Encoder: (nc, isize, isize) -> (nef*8, isize//16, isize//16)
    self.encoder = nn.Sequential(
        nn.Conv2d(nc, nef, 4, 2, padding=1),
        nn.LeakyReLU(0.2, True),
        nn.BatchNorm2d(nef),

        nn.Conv2d(nef, nef * 2, 4, 2, padding=1),
        nn.LeakyReLU(0.2, True),
        nn.BatchNorm2d(nef * 2),

        nn.Conv2d(nef * 2, nef * 4, 4, 2, padding=1),
        nn.LeakyReLU(0.2, True),
        nn.BatchNorm2d(nef * 4),

        nn.Conv2d(nef * 4, nef * 8, 4, 2, padding=1),
        nn.LeakyReLU(0.2, True),
        nn.BatchNorm2d(nef * 8),
    )

  def forward(self, inputs):
    batch_size = inputs.size(0)
    hidden = self.encoder(inputs)
    hidden = hidden.view(batch_size, -1)
    return hidden

class Decoder(nn.Module):
  def __init__(self, nc, ndf, nz, isize):
    super(Decoder, self).__init__()

    # Map the latent vector to the feature map space
    self.ndf = ndf
    self.out_size = isize // 16
    self.decoder_dense = nn.Sequential(
        nn.Linear(nz, ndf * 8 * self.out_size * self.out_size),
        nn.ReLU(True)
    )

    self.decoder_conv = nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(ndf * 8, ndf * 4, 3, 1, padding=1),
        nn.LeakyReLU(0.2, True),

        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(ndf * 4, ndf * 2, 3, 1, padding=1),
        nn.LeakyReLU(0.2, True),

        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(ndf * 2, ndf, 3, 1, padding=1),
        nn.LeakyReLU(0.2, True),

        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(ndf, nc, 3, 1, padding=1)
    )

  def forward(self, input):
    batch_size = input.size(0)
    hidden = self.decoder_dense(input).view(
        batch_size, self.ndf * 8, self.out_size, self.out_size)
    output = self.decoder_conv(hidden)
    return output


class DiagonalGaussianDistribution(object):
  # Gaussian Distribution with diagonal covariance matrix
  def __init__(self, mean, logvar=None):
    super(DiagonalGaussianDistribution, self).__init__()
    # Parameters:
    # mean: A tensor representing the mean of the distribution
    # logvar: Optional tensor representing the log of the standard variance
    #         for each of the dimensions of the distribution

    self.mean = mean
    if logvar is None:
        logvar = torch.zeros_like(self.mean)
    self.logvar = torch.clamp(logvar, -30., 20.)

    self.std = torch.exp(0.5 * self.logvar)
    self.var = torch.exp(self.logvar)

  def sample(self, noise=None):
    # Provide a reparameterized sample from the distribution
    # Return: Tensor of the same size as the mean
    # WRITE CODE HERE
    epsilon = torch.randn_like(self.mean)
    if noise is not None:
      epsilon = noise
    sample = self.mean + self.std * epsilon
    return sample

  def kl(self):
    # Compute the KL-Divergence between the distribution with the standard normal N(0, I)
    # Return: Tensor of size (batch size,) containing the KL-Divergence for each element in the batch
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    kl_div = 0.5 * torch.sum(-self.logvar + self.var + (self.mean)**2- 1, axis=-1)       # WRITE CODE HERE
    return kl_div

  def nll(self, sample, dims=[1, 2, 3]):
    # Computes the negative log likelihood of the sample under the given distribution
    # Return: Tensor of size (batch size,) containing the log-likelihood for each element in the batch
    # https://www.statlect.com/glossary/log-likelihood

    negative_ll = 0.5 * torch.sum(np.log(2 * np.pi) + self.logvar + ((sample - self.mean) / self.std)**2, dim=dims)    # WRITE CODE HERE
    return negative_ll

  def mode(self):
    # Returns the mode of the distribution
    mode = self.mean     # WRITE CODE HERE
    return mode


class VAE(nn.Module):
  def __init__(self, in_channels=3, decoder_features=32, encoder_features=32, z_dim=100, input_size=32, device=torch.device("cuda:0")):
    super(VAE, self).__init__()

    self.z_dim = z_dim
    self.in_channels = in_channels
    self.device = device

    # Encode the Input
    self.encoder = Encoder(nc=in_channels,
                            nef=encoder_features,
                            nz=z_dim,
                            isize=input_size,
                            device=device
                            )

    # Map the encoded feature map to the latent vector of mean, (log)variance
    out_size = input_size // 16
    self.mean = nn.Linear(encoder_features * 8 * out_size * out_size, z_dim)
    self.logvar = nn.Linear(encoder_features * 8 * out_size * out_size, z_dim)

    # Decode the Latent Representation
    self.decoder = Decoder(nc=in_channels,
                           ndf=decoder_features,
                           nz=z_dim,
                           isize=input_size
                           )

  def encode(self, x):
    # Input:
    #   x: Tensor of shape (batch_size, 3, 32, 32)
    # Returns:
    #   posterior: The posterior distribution q_\phi(z | x)
    # WRITE CODE HERE
    mapping_x = self.encoder(x)
    mean = self.mean(mapping_x.reshape(mapping_x.size(0), -1))
    logvar = self.logvar(mapping_x.reshape(mapping_x.size(0), -1))
    posterior = DiagonalGaussianDistribution(mean=mean,logvar=logvar)
    return posterior

  def decode(self, z):
    # Input:
    #   z: Tensor of shape (batch_size, z_dim)
    # Returns
    #   conditional distribution: The likelihood distribution p_\theta(x | z)

    # WRITE CODE HERE
    output = self.decoder(z)
    return DiagonalGaussianDistribution(mean=output)

  def sample(self, batch_size, noise=None):
    # Input:
    #   batch_size: The number of samples to generate
    # Returns:
    #   samples: Generated samples using the decoder
    #            Size: (batch_size, 3, 32, 32)

    # WRITE CODE HERE
    if noise is None:
      noise = torch.randn(batch_size, self.z_dim).to(self.device)
    
    return self.decode(noise).mode()

  def log_likelihood(self, x, K=100):
    # Approximate the log-likelihood of the data using Importance Sampling
    # Inputs:
    #   x: Data sample tensor of shape (batch_size, 3, 32, 32)
    #   K: Number of samples to use to approximate p_\theta(x)
    # Returns:
    #   ll: Log likelihood of the sample x in the VAE model using K samples
    #       Size: (batch_size,)
    posterior = self.encode(x)           # q_\phi(z | x)
    prior = DiagonalGaussianDistribution(torch.zeros_like(posterior.mean))

    log_likelihood = torch.zeros(x.shape[0], K).to(self.device)
    for i in range(K):
      z = prior.sample()
      z_i = posterior.sample(z)                        # WRITE CODE HERE (sample from q_phi)
      recon = self.decode(z_i)                    # WRITE CODE HERE (decode to conditional distribution) p_\theta(x | z)
      log_likelihood[:, i] = recon.nll(x, dims=list(range(1,x.dim()))) - posterior.nll(z_i, dims=list(range(1,z_i.dim())))     # WRITE CODE HERE (log of the summation terms in approximate log-likelihood, that is, log p_\theta(x, z_i) - log q_\phi(z_i | x))
      del z,z_i, recon
    ll  = torch.logsumexp(log_likelihood, dim=1) - torch.log(torch.tensor(K,dtype=torch.float64)).to(self.device)    # WRITE CODE HERE (compute the final log-likelihood using the log-sum-exp trick)
    return ll

  def forward(self, x, noise=None):
    # Input:
    #   x: Tensor of shape (batch_size, 3, 32, 32)
    # Returns:
    #   reconstruction: The mode of the distribution p_\theta(x | z) as a candidate reconstruction
    #                   Size: (batch_size, 3, 32, 32)
    #   Conditional Negative Log-Likelihood: The negative log-likelihood of the input x under the distribution p_\theta(x | z)
    #                                         Size: (batch_size,)
    #   KL: The KL Divergence between the variational approximate posterior with N(0, I)
    #       Size: (batch_size,)
    posterior = self.encode(x)    # WRITE CODE HERE
    if noise is None:
      noise = torch.randn_like(posterior.mode())
    latent_z = posterior.sample(noise)     # WRITE CODE HERE (sample a z)
    recon = self.decode(latent_z)        # WRITE CODE HERE (decode)

    return recon.mode(), recon.nll(x), posterior.kl()


def interpolate(model, z_1, z_2, n_samples):
  # Interpolate between z_1 and z_2 with n_samples number of points, with the first point being z_1 and last being z_2.
  # Inputs:
  #   z_1: The first point in the latent space
  #   z_2: The second point in the latent space
  #   n_samples: Number of points interpolated
  # Returns:
  #   sample: The mode of the distribution obtained by decoding each point in the latent space
  #           Should be of size (n_samples, 3, 32, 32)
  lengths = torch.linspace(0., 1., n_samples).unsqueeze(1).to(z_1.device)
  z = lengths * z_1 + (1.0 - lengths) * z_2    # WRITE CODE HERE (interpolate z_1 to z_2 with n_samples points)
  return model.decode(z).mode()