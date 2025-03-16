import torch


class SimpleCNN(torch.nn.Module):

  def __init__(self, N_repr: int, device: torch.device):
    super(SimpleCNN, self).__init__()
    self.N_repr = N_repr
    self.device = device
    self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, device=self.device)
    self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, device=self.device)
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
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

    x = self.pool(self.relu(self.conv1(x)))
    x = self.pool(self.relu(self.conv2(x)))
    x = torch.reshape(input=x, shape=(-1, 64*8*8))
    x = self.relu(self.fc1(x))

    if n_in_axes == 5:
      x = torch.reshape(input=x, shape=(-1, 2, self.N_repr))

    return x

