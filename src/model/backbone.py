import torch


class SimpleCNN(torch.nn.Module):

  def __init__(self, N_repr: int, device: torch.device):

    super(SimpleCNN, self).__init__()
    self.device = device
    self.N_repr = N_repr
    self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=2, device=self.device)
    self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2, device=self.device)
    self.fc1 = torch.nn.Linear(in_features=64*8*8, out_features=self.N_repr, device=self.device)
    self.relu = torch.nn.ReLU()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    '''
    Parameters:
      `x`. Shape (M, 2, C, H, W) or (M, C, H, W). The x.device must be the same as self.device.

    Returns:
      `x_out`. Shape (M, 2, N_repr) or (M, N_repr).
    '''

    n_in_axes = len(x.shape)
    assert n_in_axes in {4, 5}, 'E: Incompatible shape of augmentation pair minibatch.'
    assert x.device == self.device, 'E: Incompatible devices between data and trainable parameters.'
    C, H, W = x.shape[-3:]
    x = torch.reshape(input=x, shape=(-1, C, H, W))

    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = torch.reshape(input=x, shape=(-1, 64*8*8))
    x = self.relu(self.fc1(x))

    if n_in_axes == 5:
      x = torch.reshape(input=x, shape=(-1, 2, self.N_repr))

    return x

class ResNet(torch.nn.Module):

  def __init__(self, N_repr: int, device: torch.device):

    super(ResNet, self).__init__()
    self.device = device
    self.N_repr = N_repr
    self.stem = torch.nn.Sequential\
    (
      torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, device=device),
      torch.nn.BatchNorm2d(num_features=16, device=device),
      torch.nn.ReLU(inplace=True)
    )
    self.res_block1 = ResidualBlock(in_channels=16, out_channels=64, device=device)
    self.pool = torch.nn.AdaptiveAvgPool2d(output_size=(8, 8))
    self.fc1 = torch.nn.Linear(in_features=64*8*8, out_features=self.N_repr, device=self.device)
    self.relu = torch.nn.ReLU()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    '''
    Parameters:
      `x`. Shape (M, 2, C, H, W) or (M, C, H, W). The x.device must be the same as self.device.

    Returns:
      `x_out`. Shape (M, 2, N_repr) or (M, N_repr).
    '''

    n_in_axes = len(x.shape)
    assert n_in_axes in {4, 5}, 'E: Incompatible shape of augmentation pair minibatch.'
    assert x.device == self.device, 'E: Incompatible devices between data and trainable parameters.'

    C, H, W = x.shape[-3:]
    x = torch.reshape(input=x, shape=(-1, C, H, W))

    x = self.stem(x)
    x = self.res_block1(x)
    x = self.pool(x)
    x = torch.reshape(input=x, shape=(x.shape[0], -1))
    x = self.fc1(x)

    if n_in_axes == 5:
      x = torch.reshape(input=x, shape=(-1, 2, self.N_repr))

    return x

class ResidualBlock(torch.nn.Module):

  def __init__(self, in_channels: int, out_channels: int, device: torch.device):

    super().__init__()
    self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, device=device)
    self.bn1 = torch.nn.BatchNorm2d(num_features=out_channels, device=device)
    self.relu = torch.nn.ReLU(inplace=True)
    self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, device=device)
    self.bn2 = torch.nn.BatchNorm2d(num_features=out_channels, device=device)

    self.downsample = torch.nn.Sequential()

    if in_channels != out_channels:
      self.downsample = torch.nn.Sequential\
      (
        torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, device=device),
        torch.nn.BatchNorm2d(out_channels, device=device)
      )

  def forward(self, x: torch.Tensor) -> torch.Tensor:

    identity = self.downsample(x)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)

    x += identity

    x = self.relu(x)

    return x