import scipy.ndimage.filters
import matplotlib.pyplot as plt
import numpy as np

# every time-frequency bin gets set to the maximum of the rectangular window that is centered in
def main():
    np.random.seed(0)
    # =scipy.ndimage.filters.maximum_filter1d()
    signal = abs(np.random.randn(50))
    # ax = plt.subplot(1, 2, 1)
    plt.plot(signal)
    # plt.subplot(1, 2, 2)
    filter_size = 6
    filtered_signal = scipy.ndimage.filters.maximum_filter1d(signal, size=filter_size)
    plt.plot(filtered_signal)
    peaks = np.argwhere(filtered_signal == signal)
    plt.scatter(peaks, filtered_signal[peaks])
    plot_window(filter_size, left_pos=3)
    plt.show()
    return


def plot_window(filter_size, left_pos):
    plt.axvspan(left_pos, left_pos + filter_size, alpha=0.5, color='green')
    plt.axvline(left_pos + filter_size / 2, color='green')
    return


main()
