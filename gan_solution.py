import torch
from torch import nn

# Training Hyperparameters
batch_size = 64   # Batch Size
z_dim = 32        # Latent Dimensionality
gen_lr = 1e-4     # Learning Rate for the Generator
disc_lr = 1e-4    # Learning Rate for the Discriminator
device = "cuda" if torch.cuda.is_available() else "cpu"
criterion = torch.nn.BCEWithLogitsLoss()    # WRITE CODE HERE
image_size = 32
input_channels = 1
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, z_dim, channels, generator_features=32):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, generator_features * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(generator_features * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(generator_features * 4, generator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_features * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( generator_features * 2, generator_features * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_features * 1),
            nn.ReLU(True),
            nn.ConvTranspose2d( generator_features * 1, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, input):
        return self.model(input)

class Discriminator(nn.Module):
    def __init__(self, channels, discriminator_features=32):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, discriminator_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(discriminator_features, discriminator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(discriminator_features * 2, discriminator_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(discriminator_features * 4, 1, 4, 1, 0, bias=False),
        )

        self.apply(weights_init)

    def forward(self, input):
        return self.model(input)

generator = Generator(z_dim, input_channels).to(device)
discriminator = Discriminator(input_channels).to(device)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))    # WRITE CODE HERE
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))        # WRITE CODE HERE


def discriminator_train(discriminator, generator, real_samples, fake_samples):
  # Takes as input real and fake samples and returns the loss for the discriminator
  # Inputs:
  #   real_samples: Input images of size (batch_size, 3, 32, 32)
  #   fake_samples: Input images of size (batch_size, 3, 32, 32)
  # Returns:
  #   loss: Discriminator loss
  # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

  ones = torch.ones((real_samples.shape[0],1,1,1), device=real_samples.device)  # WRITE CODE HERE (targets for real data)
  zeros = torch.zeros((fake_samples.shape[0],1,1,1), device=fake_samples.device)   # WRITE CODE HERE (targets for fake data)

  real_output = discriminator(real_samples)    # WRITE CODE HERE (output of discriminator on real data)
  fake_output = discriminator(fake_samples)    # WRITE CODE HERE (output of discriminator on fake data)

  loss = criterion(real_output, ones) + criterion(fake_output, zeros)          # WRITE CODE HERE (define the loss based on criterion and above variables)
  return loss

def generator_train(discriminator, generator, fake_samples):
  # Takes as input fake samples and returns the loss for the generator
  # Inputs:
  #   fake_samples: Input images of size (batch_size, 3, 32, 32)
  # Returns:
  #   loss: Generator loss

  ones = torch.ones((fake_samples.shape[0],1,1,1), device=fake_samples.device)     # WRITE CODE HERE (targets for fake data but for generator loop)

  output = discriminator(fake_samples) # WRITE CODE HERE (output of the discriminator on the fake data)
  
  loss = criterion(output, ones)   # WRITE CODE HERE (loss for the generator based on criterion and above variables)

  return loss

def sample(generator, num_samples, noise=None):
  # Takes as input the number of samples and returns that many generated samples
  # Inputs:
  #   num_samples: Scalar denoting the number of samples
  # Returns:
  #   samples: Samples generated; tensor of shape (num_samples, 3, 32, 32)
  if noise is None:
    noise = torch.randn(num_samples, z_dim, 1, 1).to(device=device)
  with torch.no_grad():
    samples = generator(noise)   # WRITE CODE HERE (sample from p_z and then generate samples from it)
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
  lengths = torch.linspace(0., 1., n_samples).view(n_samples, 1, 1, 1).to(z_1.device)
  z = (lengths * z_1 + (1.0 - lengths) * z_2)    # WRITE CODE HERE (interpolate z_1 to z_2 with n_samples points)
  return sample(generator, n_samples, z)