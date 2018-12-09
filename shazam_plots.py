from matplotlib import pyplot as plt, ticker as plticker, colors as colors, cm


def plot_grid_of_filter_size(max_filter_size):
    plt.gca().xaxis.set_major_locator(plticker.MultipleLocator(base=max_filter_size[0]))
    plt.gca().yaxis.set_major_locator(plticker.MultipleLocator(base=max_filter_size[1]))
    plt.grid(True, color='Black')
    return


def plot_spectrogram_peaks(peak_locations, f, t):
    # plt.scatter(t[peak_locations[:, 1]], f[peak_locations[:, 0]], marker="*", c="red")
    plt.scatter(peak_locations[:, 1], peak_locations[:, 0], marker="*", c="Tomato", edgecolor="Snow", linewidths=.5)
    return


def plot_spectrogram(Sxx, f, t, alpha=1.0):
    color_norm = colors.LogNorm(vmin=1 / (2 ** 20), vmax=1)
    # color_norm = colors.LogNorm(vmin=Sxx.min(), vmax=Sxx.max())
    # plt.pcolormesh(t, f, Sxx, norm=color_norm, cmap='Greys')
    # plt.pcolormesh(t, f, Sxx, norm=color_norm, alpha=alpha)
    # plt.pcolormesh(Sxx, alpha=alpha, cmap='gray')
    # plt.pcolormesh(Sxx, norm=color_norm, alpha=alpha, cmap='gray')
    plt.pcolormesh(Sxx, norm=color_norm, alpha=alpha)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    plt.ylabel('Frequency Bin Index')
    plt.xlabel('Time Segment Index')
    # plt.yscale('log')
    # mscale.register_scale(PowerScale)
    # plt.yscale('powerscale', power=.5)

    # plt.yticks(rotation=0)
    # plt.yticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
    return


def plot_recognition_rate(recognition_rate, snrs_to_test, n_songs, clips_length, marker="o", linestyle='-', noise_type="White"):
    plt.style.use('ggplot')
    plt.plot(snrs_to_test, recognition_rate, marker=marker, linestyle=linestyle, label=str(clips_length) + " sec")
    plt.xticks(snrs_to_test)
    plt.ylim(0, 100)
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.xlabel('Signal to Noise Ratio (dB)')
    plt.ylabel('Recognition Rate')
    plt.title(str(n_songs) + " Songs. Additive " + noise_type + " Noise")

    # plt.gca().xaxis.set_major_formatter(plticker.FormatStrFormatter('%d dBFS'))
    plt.gca().yaxis.set_major_formatter(plticker.FormatStrFormatter('%d %%'))
    plt.legend()
    # plt.show()
    return


def plot_spectrogram_and_peak_subplots(Sxx, f, max_filter, max_filter_size, peak_locations, t):
    ax = plt.subplot(1, 3, 1)
    plt.title("1. Spectrogram")
    plot_spectrogram(Sxx, f, t)
    plt.subplot(1, 3, 2, sharex=ax, sharey=ax)
    plt.title("2. Spectrogram + Peaks")
    plot_spectrogram(Sxx, f, t)
    plot_spectrogram_peaks(peak_locations, f, t)
    plt.subplot(1, 3, 3, sharex=ax, sharey=ax)
    plt.title("3. Peaks")
    plot_spectrogram_peaks(peak_locations, f, t)
    plt.xlim(0, 350)
    plt.ylim(0, 512)
    plt.show()
    return


def plot_spectrogram_and_peak_subplots_detailed(Sxx, f, max_filter, max_filter_size, peak_locations, t):
    ax = plt.subplot(2, 3, 1)
    plt.title("1. Spectrogram")
    plot_spectrogram(Sxx, f, t)
    plt.subplot(2, 3, 2, sharex=ax, sharey=ax)
    plt.title("2. Max Filtered")
    plot_grid_of_filter_size(max_filter_size)
    plot_spectrogram(max_filter, f, t)
    # rect = patches.Rectangle((0, 0), max_filter_size[1] * t_step, max_filter_size[0] * f_step, edgecolor="black")
    # plt.gca().add_patch(rect)
    # ylim = plt.ylim()
    # xlim = plt.xlim()
    # for i in range(23):
    #     plt.axvline(x=max_filter_size[0] * t_step * i)
    # for i in range(9):
    #     plt.axhline(y=max_filter_size[1] * f_step * i)
    # plt.ylim(ylim)
    # plt.xlim(xlim)
    plt.subplot(2, 3, 3, sharex=ax, sharey=ax)
    plt.title("3.(A) Max Filtered == Spectrogram")
    plot_spectrogram(max_filter, f, t)
    plot_grid_of_filter_size(max_filter_size)
    plot_spectrogram_peaks(peak_locations, f, t)
    plt.subplot(2, 3, 4, sharex=ax, sharey=ax)
    plt.title("3.(B) Max Filtered == Spectrogram")
    plot_spectrogram(Sxx, f, t)
    plot_grid_of_filter_size(max_filter_size)
    plot_spectrogram_peaks(peak_locations, f, t)
    plt.subplot(2, 3, 5, sharex=ax, sharey=ax)
    plt.title("3.(C) Max Filtered == Spectrogram")
    # plot_spectrogram(Sxx, f, t)
    plot_grid_of_filter_size(max_filter_size)
    plot_spectrogram_peaks(peak_locations, f, t)
    plt.xlim(0, 500)
    plt.ylim(0, 512)
    plt.show()
    return


def start_hist_subplots(n_possible_songs):
    plt.style.use('ggplot')
    ax = plt.subplot(n_possible_songs, 1, 1)
    return ax


def make_next_hist_subplot(ax, i, n_possible_songs, song_id, n_matching_fingerprints):
    if i > 0:
        plt.subplot(n_possible_songs, 1, i + 1, sharey=ax)
    plt.title("song_id:" + str(song_id) + ", n_fingerprints: " + str(n_matching_fingerprints))


def show_hist_plot(max_hist_song, song_doc):
    plt.suptitle("matching song id=" + str(max_hist_song) + ",correct song=" + str(song_doc['_id']))
    plt.tight_layout()

    plt.show()


def plot_hist_of_stks(unique, filtered_hist, alpha=1):
    # hist_mpl, bin_edges_mpl, patches = plt.hist(stks_in_songID.values, bins='auto', rwidth=.7)
    x = unique[filtered_hist != 0]
    y = filtered_hist[filtered_hist != 0]
    plt.bar(x, y, alpha=alpha)
    plt.ylabel("count")
    plt.xlabel("time-offset delta")
    return


def plot_show():
    plt.show()


def plot_scatter_of_fingerprint_offsets(color_index, db_fp_offset, db_fp_song_id, local_fp_offset, n_fingerprints):
    plt.style.use('ggplot')
    # viridis = cm.get_cmap('viridis', n_fingerprints).colors
    plt.scatter(db_fp_offset, local_fp_offset)  # c=viridis[color_index])
    # plt.text(db_fp_offset, local_fp_offset, db_fp_song_id)
    return


def finish_scatter_of_fingerprint_offsets():
    plt.title("Incorrect Song")
    plt.xlabel("Remote fingerprint offset")
    plt.ylabel("Local fingerprint offset")
    return
