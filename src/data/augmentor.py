from copy import deepcopy
import numpy as np
import cv2

from numpy.typing import NDArray


def apply_crop(img_arr: NDArray, nrm_ul: tuple[float, float], nrm_lr: tuple[float, float]) -> NDArray:
  '''
  Description:
    Crops an image while preserving the initial image resolution.

  Parameters:
    img_arr. Shape (H, W, ...). Input image. If there is a channel axis, then the color system is BGR.
    nrm_ul. (H_nrm_ul, W_nrm_ul). Up left (ul) normalized (nrm) coordinates of preserved subimage.
      H_nrm_u. Up (u) vertical subimage coordinate.
      W_nrm_l. Left (l) horizontal subimage coordinate.
    nrm_lr. (H_nrm_lr, W_nrm_lr). Low right (lr) normalized (nrm) coordinates of preserved subimage.
      H_nrm_l. Low (l) vertical subimage coordinate.
      W_nrm_r. Right (r) horizontal subimage coordinate.

  Returns:
    img_arr_out. Shape (H, W, ...). Cropped image.
  '''

  H_nrm_u, W_nrm_l = nrm_ul
  H_nrm_l, W_nrm_r = nrm_lr

  H, W = img_arr.shape[:2]

  H_u = int(H_nrm_u*H)
  W_l = int(W_nrm_l*W)
  H_l = int(H_nrm_l*H)
  W_r = int(W_nrm_r*W)

  img_arr_out = img_arr[H_u:H_l, W_l:W_r]
  img_arr_out = cv2.resize(src=img_arr_out, dsize=(W, H))

  return img_arr_out

def apply_jitter(img_nrm_arr: NDArray) -> NDArray:
  '''
  Description:
    Performs jitter distortion to a normalized input image where the pixel intensity value belongs to [0, 1].

  Parameters:
    `img_nrm_arr`. Shape (H, W, ...). Normalized input image array.

  Returns:
    `img_nrm_arr_out`. Shape (H, W, ...).
  '''

  def apply_intensity_distortion(img_nrm_arr: NDArray, beta: float) -> NDArray:
    '''
    Description:
      Performs intensity distortion by adding an intensity offset value beta to all pixel intensity values.

    Parameters:
      `img_nrm_arr`. Shape (H, W, ...).
      `beta`. Belongs to the interval [-0.2, 0.2]. Intensity offset.

    Returns:
      `img_nrm_arr_out`. Shape (H, W, ...).
    '''

    assert -.2 <= beta <= .2, "E: Disallowed beta argument detected."

    return np.clip(a=img_nrm_arr+beta, a_min=0, a_max=1)

  def apply_contrast_distortion(img_nrm_arr: NDArray, alpha: float) -> NDArray:
    '''
    Description:
      Performs contrast distortion based on an alpha distortion parameter. In other words it either shrinks or expands image contrast the depending on the alpha parameter.

    Parameters:
      `img_nrm_arr`. Shape (H, W, ...).
      `alpha`. Belongs to the interval [0.8, 1.2]. Contrast distortion scaling factor.
    '''

    assert 0.8 <= alpha <= 1.2, "E: Disallowed alpha argument detected."

    mean = np.mean(img_nrm_arr)
    img_nrm_arr_out = alpha * (img_nrm_arr - mean) + mean

    return np.clip(a=img_nrm_arr_out, a_min=0, a_max=1)

  def apply_saturation_distortion(img_nrm_arr: NDArray) -> NDArray:
    pass

  def apply_hue_distortion(img_nrm_arr: NDArray) -> NDArray:
    pass

  img_nrm_arr_out = deepcopy(img_nrm_arr)

  img_nrm_arr_out = apply_intensity_distortion(img_nrm_arr=img_nrm_arr_out, beta=0.1)
  img_nrm_arr_out = apply_contrast_distortion(img_nrm_arr_out, alpha=0.8)
  img_nrm_arr_out = apply_saturation_distortion(img_nrm_arr_out)
  img_nrm_arr_out = apply_hue_distortion(img_nrm_arr_out)

  return img_nrm_arr_out


