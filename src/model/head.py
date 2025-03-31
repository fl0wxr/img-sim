import torch


class MLP(torch.nn.Module):

  def __init__(self, N_repr: int, device: torch.device):
    super(MLP, self).__init__()
    self.N_repr = N_repr
    self.device = device
    self.N_proj = 32
    self.fc1 = torch.nn.Linear(in_features=self.N_repr, out_features=64, device=self.device)
    self.fc2 = torch.nn.Linear(in_features=64, out_features=self.N_proj, device=self.device)
    self.relu = torch.nn.ReLU()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    '''
    Parameters:
      `x`. Shape (M, 2, N_repr) or (M, N_repr).

    Return:
      `x_out`. Shape (M, 2, self.N_proj) or (M, self.N_proj).
    '''

    n_in_axes = len(x.shape)

    assert n_in_axes in {2, 3}, 'E: Incompatible shape of augmentation pair minibatch.'
    assert x.device == self.device, 'E: Incompatible devices between data and trainable parameters.'
    x = torch.reshape(input=x, shape=(-1, self.N_repr))

    x = self.relu(self.fc1(x))
    x = self.fc2(x)

    if n_in_axes == 3:
      x = torch.reshape(input=x, shape=(-1, 2, self.N_proj))

    return x