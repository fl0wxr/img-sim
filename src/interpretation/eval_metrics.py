import torch


def similarity(vec: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
  '''
  Description:
    Cosine similarity between vector pairs in a Euclidean vector space equipped with the cosine inner product.

  Parameters:
    `vec`. Shape (n,). The basis vector. Where n is the size of the basis vector.
    `mat`. Shape (m, n). Where n is the size of each vector, and m is the number of vectors.

  Returns:
    `sim_mat`. Shape (m,). Similarity sequence.
  '''

  vec = torch.reshape(input=vec, shape=(1, *vec.shape))

  norm2_vec = torch.norm(input=vec, p=2, dim=0)
  norm2_mat = torch.norm(input=mat, p=2, dim=1, keepdim=True) # Shape (m, 1)
  vec_normalized = vec / (norm2_vec + 1e-10)
  mat_normalized = mat / (norm2_mat + 1e-10)
  sim_mat = torch.matmul(input=mat_normalized, other=vec_normalized.T)[:, 0]

  return sim_mat