from shazam import combine_parts_into_key, decode_hash


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
