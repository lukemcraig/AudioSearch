import scipy.ndimage.filters
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
# =scipy.ndimage.filters.maximum_filter1d()
signal = abs(np.random.randn(50))

# ax = plt.subplot(1, 2, 1)
plt.plot(signal)
# plt.subplot(1, 2, 2)
filtered_signal = scipy.ndimage.filters.maximum_filter1d(signal, size=6)
plt.plot(filtered_signal)
peaks = np.argwhere(filtered_signal == signal)
plt.scatter(peaks, filtered_signal[peaks])
plt.show()
