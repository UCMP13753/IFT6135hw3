import torch

discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))    # WRITE CODE HERE
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))        # WRITE CODE HERE

criterion = torch.nn.BCELoss()    # WRITE CODE HERE

def discriminator_train(discriminator, generator, real_samples, fake_samples):
  # Takes as input real and fake samples and returns the loss for the discriminator
  # Inputs:
  #   real_samples: Input images of size (batch_size, 3, 32, 32)
  #   fake_samples: Input images of size (batch_size, 3, 32, 32)
  # Returns:
  #   loss: Discriminator loss
  # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

  ones = torch.full((real_samples.size(0),), 1, dtype=torch.float, device=real_samples.device)  # WRITE CODE HERE (targets for real data)
  zeros = torch.full((fake_samples.size(0),), 0, dtype=torch.float, device=fake_samples.device)   # WRITE CODE HERE (targets for fake data)

  real_output = discriminator(real_samples)    # WRITE CODE HERE (output of discriminator on real data)
  fake_output = discriminator(fake_samples)    # WRITE CODE HERE (output of discriminator on fake data)

  loss = criterion(real_output, ones) + criterion(fake_output, zeros)           # WRITE CODE HERE (define the loss based on criterion and above variables)

  return loss

def generator_train(discriminator, generator, fake_samples):
  # Takes as input fake samples and returns the loss for the generator
  # Inputs:
  #   fake_samples: Input images of size (batch_size, 3, 32, 32)
  # Returns:
  #   loss: Generator loss

  ones = torch.full((fake_samples.size(0),), 1, dtype=torch.float, device=fake_samples.device)     # WRITE CODE HERE (targets for fake data but for generator loop)

  output = discriminator(fake_samples) # WRITE CODE HERE (output of the discriminator on the fake data)

  loss = -criterion(output, ones)   # WRITE CODE HERE (loss for the generator based on criterion and above variables)

  return loss

def sample(generator, num_samples, noise=None):
  # Takes as input the number of samples and returns that many generated samples
  # Inputs:
  #   num_samples: Scalar denoting the number of samples
  # Returns:
  #   samples: Samples generated; tensor of shape (num_samples, 3, 32, 32)
  if noise is None:
    noise = torch.randn(num_samples, 32, 1, 1, device=generator.device)
  with torch.no_grad():
    samples = generator(noise)   # WRITE CODE HERE (sample from p_z and then generate samples from it)
    print(samples.shape)
  return samples


def interpolate(generator, z_1, z_2, n_samples):
  # Interpolate between z_1 and z_2 with n_samples number of points, with the first point being z_1 and last being z_2.
  # Inputs:
  #   z_1: The first point in the latent space
  #   z_2: The second point in the latent space
  #   n_samples: Number of points interpolated
  # Returns:
  #   sample: A sample from the generator obtained from each point in the latent space
  #           Should be of size (n_samples, 3, 32, 32)

  # WRITE CODE HERE (interpolate z_1 to z_2 with n_samples points and then)
  # WRITE CODE HERE (    generate samples from the respective latents     )
  lengths = torch.linspace(0., 1., n_samples).unsqueeze(1).to(z_1.device)
  z = z_1 + lengths * (z_2 - z_1)    # WRITE CODE HERE (interpolate z_1 to z_2 with n_samples points)
  return sample(generator, n_samples, z)