import numpy as np
from scipy.fft import fft2
from scipy.fft import ifft2


def normxcorr2_masked(
    fixed_image, moving_image=None, fixed_mask=None, moving_mask=None
):
    """Masked normalized cross correlation.

    Calculate a masked normalized cross-correlation by fast Fourier transforms
    instead of by spatial correlation (ref. 1).
 
    `fixed_mask` and `moving_mask` should consist of only True and False
    values or 1's and 0's, where 1's indicate indexes of valid information
    in the corresponding image, while 0's indicate indexes that should be
    masked (ignored).

    `fixed_image` and `moving_image` need not be the same size, but
    `fixed_mask` must be the same size as `fixed_image`, and `moving_mask`
    must be the same size as `moving_image`.

    The resulting `correlation_matrix` contains correlation coefficients 
    with values ranging from -1.0 to 1.0.

    Args:
      fixed_image:  A matrix of values to cross-correlate with `moving_image`.
      moving_image: A matrix of values to cross-correlate with `fixed_image`.
                    If not given, returns an autocorrelogram of `fixed_image`.
      fixed_mask:   A boolean matrix the shape of `fixed_image` with ones
                    in valid bins. If argument is not given, all `nan`
                    values in `fixed_image` will be masked.
      moving_mask:  A boolean matrix the shape of `moving_image` with ones
                    in valid bins. If argument is not given, all `nan`
                    values in `moving_image` will be masked.

    Returns:
      correlation_matrix: Masked normalized cross correlation of
                          `fixed_image` with `moving_image`.
    
    References:
      [1]: Padfield 2012 IEEE Trans Image Process
           "Masked object registration in the Fourier domain"
            https://doi.org/10.1109/TIP.2011.2181402


    2011-??-?? padfield. matlab function.
    2021-06-06 waaga. normxcorr2_masked.py (python 3).
    """

    def non_negative_real(x):
        """Translate to non-negative real numbers

        Args:
        x: A matrix of values.

        Returns:
        b: The non-negative matrix `x` with values shifted by 
            adding the minimum value of `x` to all values.

        """
        b = x.astype(np.float64)
        min_b = np.min(b)
        if min_b < 0:
            b = b - min_b
        return b

    def find_closest_valid_dimension(n):
        """Find closest valid dimension.
        
        To speed up the FFT calculations, find closest valid 
        dimension above the desired dimension. This will be a 
        combination of 2's, 3's and/or 5's.

        Args:
        n: Combined size of `fixed_image` and `moving_image`

        Returns:
        new_n: Optimal dimension.

        """
        new_n = n - 1
        result = 0
        while result != 1:
            new_n += 1
            result = factorize_number(new_n)
        return new_n

    def factorize_number(n):
        """Remainder after division of `n` by [2, 3, 5]

        Args:
        n: A whole number.

        Returns:
        n: The remainder after devision of `n` by 2, 3, or 5.

        """
        for x in [2, 3, 5]:
            while np.int(n) - np.int(n / x) * np.int(x) == 0:
                n /= x
        return n

    ### Preprocess input data:

    # Handle missing function arguments
    if moving_image is None:
        moving_image = fixed_image
    if fixed_mask is None:
        fixed_mask = np.isnan(fixed_image) == False
    if moving_mask is None:
        moving_mask = np.isnan(moving_image) == False

    # For numerical robust results in the normalized cross-correlation, we
    # need to make sure the input values are non-negative real numbers.
    fixed_image = non_negative_real(fixed_image)
    moving_image = non_negative_real(moving_image)
    fixed_mask = fixed_mask.astype("float64")
    moving_mask = moving_mask.astype("float64")

    # Validate masks.
    fixed_mask[fixed_mask <= 0] = 0
    fixed_mask[fixed_mask > 0] = 1
    moving_mask[moving_mask <= 0] = 0
    moving_mask[moving_mask > 0] = 1

    # Mask input.
    fixed_image[fixed_mask == 0] = 0
    moving_image[moving_mask == 0] = 0

    # Flip the moving image and mask in both dimensions so that its
    # correlation can be easily handled.
    rotated_moving_image = np.rot90(moving_image, k=2)
    rotated_moving_mask = np.rot90(moving_mask, k=2)

    ### Calculate the masked normalized cross-correlation:
    fixed_image_size = np.asarray(fixed_image.shape)
    moving_image_size = np.asarray(moving_image.shape)
    combined_size = fixed_image_size + moving_image_size - 1

    # To make the FFT caluclation faster, we find the next largest size that
    # is a multiple of a combination of 2, 3, and/or 5.
    optimal_size = np.ndarray(2, dtype="int32")
    optimal_size[0] = find_closest_valid_dimension(combined_size[0])
    optimal_size[1] = find_closest_valid_dimension(combined_size[1])

    # Compute numerator:
    fixed_fft = fft2(fixed_image, s=optimal_size)
    rotated_moving_fft = fft2(rotated_moving_image, s=optimal_size)
    fixed_mask_fft = fft2(fixed_mask, s=optimal_size)
    rotated_moving_mask_fft = fft2(rotated_moving_mask, s=optimal_size)

    n_overlap_masked_pixel = np.real(ifft2(rotated_moving_mask_fft * fixed_mask_fft))
    n_overlap_masked_pixel = np.round(n_overlap_masked_pixel)
    n_overlap_masked_pixel = np.maximum(n_overlap_masked_pixel, np.finfo(float).eps)

    mask_correlated_fixed_fft = np.real(ifft2(rotated_moving_mask_fft * fixed_fft))

    mask_correlated_rotated_moving_fft = np.real(
        ifft2(fixed_mask_fft * rotated_moving_fft)
    )

    numerator = (
        np.real(ifft2(rotated_moving_fft * fixed_fft))
        - mask_correlated_fixed_fft
        * mask_correlated_rotated_moving_fft
        / n_overlap_masked_pixel
    )

    # Compute denominator:
    fixed_squared_fft = fft2(fixed_image * fixed_image, s=optimal_size)

    fixed_denom = (
        np.real(ifft2(rotated_moving_mask_fft * fixed_squared_fft))
        - mask_correlated_fixed_fft ** 2 / n_overlap_masked_pixel
    )

    fixed_denom = np.maximum(fixed_denom, 0)

    rotated_moving_squared_fft = fft2(
        rotated_moving_image * rotated_moving_image, s=optimal_size
    )

    moving_denom = (
        np.real(ifft2(fixed_mask_fft * rotated_moving_squared_fft))
        - mask_correlated_rotated_moving_fft ** 2 / n_overlap_masked_pixel
    )

    moving_denom = np.maximum(moving_denom, 0)
    denom = np.sqrt(fixed_denom * moving_denom)

    # Divide numerator by non-negative denominator values.
    correlation_matrix = np.zeros(numerator.shape)
    tolerance = 1000 * np.spacing(np.max(np.abs(denom)))
    i_nonzero = np.nonzero(denom > tolerance)
    correlation_matrix[i_nonzero] = numerator[i_nonzero] / denom[i_nonzero]

    # Handle overflow. Correlation coefficients should be in the range [-1, 1].
    correlation_matrix[correlation_matrix < -1] = -1
    correlation_matrix[correlation_matrix > 1] = 1

    # Crop out the correct return size.
    correlation_matrix = correlation_matrix[0 : combined_size[0], 0 : combined_size[1]]
    n_overlap_masked_pixel = n_overlap_masked_pixel[
        0 : combined_size[0], 0 : combined_size[1]
    ]
    return correlation_matrix

