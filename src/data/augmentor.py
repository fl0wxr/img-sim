from copy import deepcopy
import numpy as np
import cv2

from numpy.typing import NDArray


def apply_crop(img_arr: NDArray, nrm_ul: tuple[float, float], nrm_lr: tuple[float, float]) -> NDArray:
  '''
  Description:
    Crops an image while preserving the initial image resolution.

  Parameters:
    `img_arr`. Shape (H, W, ...). Input image. If there is a channel axis, then the color system is BGR.
    `nrm_ul`. (H_nrm_ul, W_nrm_ul). Up left (ul) normalized (nrm) coordinates of preserved subimage.
      `H_nrm_u`. Up (u) vertical subimage coordinate.
      `W_nrm_l`. Left (l) horizontal subimage coordinate.
    `nrm_lr`. (H_nrm_lr, W_nrm_lr). Low right (lr) normalized (nrm) coordinates of preserved subimage.
      `H_nrm_l`. Low (l) vertical subimage coordinate.
      `W_nrm_r`. Right (r) horizontal subimage coordinate.

  Returns:
    `img_arr_out`. Shape (H, W, ...). Cropped image.
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

def bgr2gs(img_nrm_arr_bgr: NDArray) -> NDArray:
  '''
  Description:
    Converts a BGR image to its grayscale equivalent.

  Parameters:
    `img_nrm_arr_bgr`. Shape (H, W, 3). Normalized input image array. Pixels adhere to the BGR color model.

  Returns:
    `img_nrm_arr_gs_out`. Shape (H, W, 3).
  '''

  img_nrm_arr_gs_out = np.stack([0.114*img_nrm_arr_bgr[..., 0], 0.587*img_nrm_arr_bgr[..., 1], 0.299*img_nrm_arr_bgr[..., 2]], axis=-1)

  return img_nrm_arr_gs_out

def apply_jitter(img_nrm_arr_bgr: NDArray) -> NDArray:
  '''
  Description:
    Performs jitter distortion to a normalized input image where the pixel intensity value belongs to [0, 1].

  Parameters:
    `img_nrm_arr_bgr`. Shape (H, W, ...). Pixels adhere to the BGR color model.

  Returns:
    `img_nrm_arr_bgr_out`. Shape (H, W, ...).
  '''

  def apply_intensity_distortion(img_nrm_arr: NDArray, beta: float) -> NDArray:
    '''
    Description:
      Performs intensity distortion by adding an intensity offset value for either darkening (beta<0) or brightening (beta>0) of pixel intensity values.

    Parameters:
      `img_nrm_arr_bgr`. Shape (H, W, ...). Pixels adhere to the BGR color model.
      `beta`. Belongs to the interval [-0.2, 0.2]. Intensity offset.

    Returns:
      `img_nrm_arr_bgr_out`. Shape (H, W, ...).
    '''

    assert -.2 <= beta <= .2, "E: Disallowed beta argument detected."

    img_nrm_arr_out = img_nrm_arr + beta

    return np.clip(a=img_nrm_arr_out, a_min=0, a_max=1)

  def apply_contrast_distortion(img_nrm_arr: NDArray, alpha: float) -> NDArray:
    '''
    Description:
      Performs contrast distortion based on a  pixel intensity distortion parameter. In other words it either shrinks (alpha<1) or expands (alpha>1) the image's intensity histogram depending on the alpha parameter. The shrinking/expansion is applied with respect to the mean intensity value.

    Parameters:
      `img_nrm_arr`. Shape (H, W, ...).
      `alpha`. Belongs to the interval [0.8, 1.2]. Contrast distortion scaling factor.

    Returns:
      `img_nrm_arr_out`. Shape (H, W, ...).
    '''

    assert .8 <= alpha <= 1.2, "E: Disallowed alpha argument detected."

    mean = np.mean(img_nrm_arr)
    img_nrm_arr_out = alpha * (img_nrm_arr - mean) + mean

    return np.clip(a=img_nrm_arr_out, a_min=0, a_max=1)

  def apply_saturation_distortion(img_nrm_arr_bgr: NDArray, alpha: float) -> NDArray:
    '''
    Description:
      Modifies image saturation based on a pixel color distortion parameter. Either increases (alpha<1) or decreases (alpha>1) the pixelwise diversification of RGB color intensities either corresponding to more vivid or duller colors respectively.

    Parameters:
      `img_nrm_arr_bgr`. Shape (H, W, 3). Pixels adhere to the BGR color model.
      `alpha`. Belongs to the interval [0.8, 1.2]. Color saturation distortion factor.

    Returns:
      `img_nrm_arr_bgr_out`. Shape (H, W, 3). Pixels adhere to the BGR color model.
    '''

    assert .8 <= alpha <= 1.2, "E: Disallowed alpha argument detected."

    img_nrm_arr_gs = bgr2gs(img_nrm_arr_bgr)
    img_nrm_arr_bgr_out = alpha * img_nrm_arr_bgr + (1-alpha) * img_nrm_arr_gs
    img_nrm_arr_bgr_out = np.clip(a=img_nrm_arr_bgr_out, a_min=0, a_max=1)

    return img_nrm_arr_bgr_out

  def apply_hue_distortion(img_nrm_arr_bgr: NDArray, delta: int) -> NDArray:
    '''
    Description:
      Modifies image hue based on a hue distortion offset parameter delta. By modifying the hue value, the perceived color is modified. OpenCV's hue value belongs to the interval [0, 180), with direction Red -> Green -> Blue (delta > 0). For example, for delta>0, green colored pixels are pushed towards more blue colors.

    Parameters:
      `img_nrm_arr_bgr`. Shape (H, W, 3). Pixels adhere to the BGR color model.
      `delta`. Belongs to the interval [-0.1*180, 0.1*180]. Hue distortion offset.

    Returns:
      `img_nrm_arr_bgr_out`. Shape (H, W, 3). Pixels adhere to the BGR color model.
    '''

    assert -.1*180 <= delta <= .1*180, "E: Disallowed delta argument detected."

    # Pixel: (blue, green, red) -> (hue, saturation, value)
    img_nrm_arr_hsv = cv2.cvtColor(src=img_nrm_arr_bgr, code=cv2.COLOR_BGR2HSV)
    img_nrm_arr_hsv[..., 0] = (img_nrm_arr_hsv[..., 0] + delta) % 180
    img_nrm_arr_bgr_out = cv2.cvtColor(src=img_nrm_arr_hsv, code=cv2.COLOR_HSV2BGR)

    return img_nrm_arr_bgr_out

  img_nrm_arr_bgr_out = deepcopy(img_nrm_arr_bgr)

  img_nrm_arr_bgr_out = apply_intensity_distortion(img_nrm_arr=img_nrm_arr_bgr_out, beta=0.1)
  img_nrm_arr_bgr_out = apply_contrast_distortion(img_nrm_arr=img_nrm_arr_bgr_out, alpha=0.8)
  img_nrm_arr_bgr_out = apply_saturation_distortion(img_nrm_arr_bgr=img_nrm_arr_bgr_out, alpha=0.8)
  img_nrm_arr_bgr_out = apply_hue_distortion(img_nrm_arr_bgr=img_nrm_arr_bgr_out, delta=0.1*180)

  return img_nrm_arr_bgr_out


