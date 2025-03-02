from numpy.typing import NDArray


def contrastive_loss(embedded_pairs: tuple[NDArray]) -> float:
  '''
  Description:
    Computation of the contrastive loss function for a set of input embedding pairs.

  Parameters:
    `embedded_pairs`. Shape (M, 2, N). Where M is the number of pairs and N is the dimensionality of the embedding space. The second axis is the pair axis, where the position with index 0 holds one embedded instance and index 1 holds the other embedded instance.

  Returns:
    `loss_value`.
  '''

  pass








