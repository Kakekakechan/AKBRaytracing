
import numpy as np

__all__ = ["compute_psf_fft", "psf_to_db", "ensure_even_size"]

def ensure_even_size(arr):
    '''
    If array side lengths are odd, pad by 1 pixel to make even (helps symmetric fftshift).
    Returns padded array and a tuple of slices to crop back later if needed.
    '''
    ny, nx = arr.shape
    pad_y = 1 if (ny % 2) else 0
    pad_x = 1 if (nx % 2) else 0
    if pad_x or pad_y:
        arr2 = np.pad(arr, ((0,pad_y),(0,pad_x)), mode="constant", constant_values=0)
        crop = (slice(0, ny), slice(0, nx))
        return arr2, crop
    return arr, None

def _hann2d(shape):
    '''Return separable 2D Hann window (unit peak).'''
    ny, nx = shape
    wx = 0.5 - 0.5*np.cos(2*np.pi*np.arange(nx)/nx)
    wy = 0.5 - 0.5*np.cos(2*np.pi*np.arange(ny)/ny)
    w2 = np.outer(wy, wx)
    w2 /= w2.max()
    return w2

def compute_psf_fft(
    opd_m,
    amp,
    wavelength_m,
    pupil_dx_m,
    focal_length_m,
    pad_factor=2,
    window=None,
    return_efield=False,
):
    '''
    Fraunhofer PSF from pupil plane OPD and amplitude via FFT.

    Parameters
    ----------
    opd_m : (Ny, Nx) array
        Optical path difference at the exit pupil in meters. NaNs are allowed (masked to zero amplitude).
    amp : (Ny, Nx) array
        Real amplitude transmission of the pupil (incl. apodization & aperture). NaNs -> 0.
        Typical choice: binary aperture (1 inside pupil, 0 outside) or smoothly apodized values in [0,1].
    wavelength_m : float
        Wavelength in meters.
    pupil_dx_m : float
        Sample pitch in the pupil plane in meters (assumed square pixels).
    focal_length_m : float
        Effective focal length for the Fraunhofer propagation (meters).
    pad_factor : int, optional
        Zero-padding factor (>=1). Increases sampling density in the image plane.
    window : {"hann", None}, optional
        Optional apodization window applied to the pupil to suppress ringing from sharp edges.
        Use carefully; it changes the PSF. Default None.
    return_efield : bool, optional
        If True, also return the complex focal plane field (after fft & scaling).

    Returns
    -------
    psf : (Ny*pad_factor, Nx*pad_factor) array
        Normalized intensity PSF (peak = 1 for an unaberrated, unabscured pupil with amp normalized).
    x_im, y_im : 1D arrays
        Image-plane coordinates in meters at the focal plane.
        Pixel size is: dx_im = wavelength_m * focal_length_m / (Nx * pupil_dx_m)
    (optional) efield_im : complex ndarray
        Complex field in the image plane (arbitrary overall phase).
    '''
    if opd_m.shape != amp.shape:
        raise ValueError("opd_m and amp must have the same shape")
    if pad_factor < 1 or int(pad_factor) != pad_factor:
        raise ValueError("pad_factor must be a positive integer")
    opd = np.array(opd_m, dtype=float)
    A = np.array(amp, dtype=float)
    # Robust to NaNs
    A = np.where(np.isfinite(A), A, 0.0)
    opd = np.where(np.isfinite(opd), opd, 0.0)
    # Pupil field
    phase = (2.0*np.pi/wavelength_m) * opd
    U_p = A * np.exp(1j * phase)
    # Optional window
    if window is not None:
        wname = str(window).lower()
        if wname == "hann":
            U_p = U_p * _hann2d(U_p.shape)
        else:
            raise ValueError(f"Unsupported window '{window}'. Options: 'hann' or None.")
    # Make even size (helps exact symmetry after fftshift)
    U_p, crop = ensure_even_size(U_p)
    ny, nx = U_p.shape
    # Zero padding
    py = ny * pad_factor
    px = nx * pad_factor
    pad_y0 = (py - ny)//2
    pad_x0 = (px - nx)//2
    U_pad = np.pad(U_p, ((pad_y0, py-ny-pad_y0), (pad_x0, px-nx-pad_x0)), mode="constant")
    # FFT with proper scaling:
    # Continuous FT approximated by discrete sum => multiply by pupil sample area (dx*dy)
    # Fraunhofer constant factor (1/(i λ f)) does not affect intensity except global scale;
    # we normalize PSF peak afterwards.
    dx = pupil_dx_m
    dA = dx*dx
    U_im = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U_pad))) * dA
    # Coordinates in image plane
    fx = np.fft.fftshift(np.fft.fftfreq(px, d=dx))  # cycles/m
    fy = np.fft.fftshift(np.fft.fftfreq(py, d=dx))
    x_im = wavelength_m * focal_length_m * fx  # meters
    y_im = wavelength_m * focal_length_m * fy  # meters
    # Intensity and normalization (peak = 1)
    I = np.abs(U_im)**2
    Imax = I.max()
    if Imax > 0:
        I = I / Imax
    if return_efield:
        return I, x_im, y_im, U_im / np.sqrt(Imax if Imax > 0 else 1.0)
    return I, x_im, y_im

def psf_to_db(psf, floor_db=-60.0):
    '''Convert linear PSF to dB with floor clipping.'''
    with np.errstate(divide="ignore"):
        psf_db = 10.0*np.log10(np.maximum(psf, 10.0**(floor_db/10.0)))
    return psf_db
