
# Example usage (replace dummy data with your exit pupil arrays):
#
import numpy as np
from psf_fft import compute_psf_fft, psf_to_db

wavelength_m = 13.5e-9      # 532.1 nm
focal_length_m = 0.0100       # 100 mm
pupil_dx_m = 50e-6           # 10 µm sampling in pupil plane

# Dummy pupil (circular, radius ~ 1 mm)
N = 1024
x = (np.arange(N) - N//2) * pupil_dx_m
X, Y = np.meshgrid(x, x, indexing='xy')
r = np.sqrt(X**2 + Y**2)
NA = 0.5
R = focal_length_m*NA
amp = (r <= R).astype(float)   # clear circular aperture

# Example OPD: defocus-like profile (~ 0.5 waves PV)
opd = np.zeros_like(amp, dtype=float)
# mask = (r <= R)
# rho2 = (r[mask]/R)**2
# opd[mask] = 0.25 * wavelength_m * (2*rho2 - 1.0)

psf, x_im, y_im = compute_psf_fft(opd, amp, wavelength_m, pupil_dx_m, focal_length_m, pad_factor=16, window=None)

# Plotting (optional):
import matplotlib.pyplot as plt
plt.figure()
plt.pcolormesh(X, Y, opd, cmap='jet', shading='auto')
plt.colorbar(label='\u03BB')

plt.figure()
plt.pcolormesh(X, Y, amp, cmap='jet', shading='auto')
plt.colorbar(label='amp')

plt.figure()
plt.plot(x,amp[amp.shape[0]//2,:])

plt.figure()
plt.imshow(psf, extent=[x_im[0], x_im[-1], y_im[0], y_im[-1]], origin='lower')
plt.xlabel('x on sensor [m]')
plt.ylabel('y on sensor [m]')
plt.title('PSF (linear)')
plt.colorbar()

plt.figure()
plt.plot(x_im,psf[psf.shape[0]//2,:])
plt.figure()
plt.plot(x_im,np.log(psf[psf.shape[0]//2,:]))
plt.show()
