import numpy as np
from audio_search_main import combine_parts_into_key, decode_hash, add_noise, get_rms_linear, convert_to_db


def test_hash_function():
    peak_f = 1024
    second_peak_f = 1022
    time_delta = 1023
    key = combine_parts_into_key(peak_f, second_peak_f, time_delta)
    d_pf, d_spf, d_td = decode_hash(key)
    assert d_pf == peak_f
    assert d_spf == second_peak_f
    assert d_td == time_delta
    return


def test_add_noise_snr():
    desired_snr_db = -11

    data = np.sin(np.arange(0, 300)) * .42

    data_with_noise = add_noise(data, desired_snr_db)

    actual_snr_linear = get_rms_linear(data) / get_rms_linear(data_with_noise - data)
    actual_snr_db = convert_to_db(actual_snr_linear)
    np.testing.assert_almost_equal(actual=actual_snr_db, desired=desired_snr_db)
    return
