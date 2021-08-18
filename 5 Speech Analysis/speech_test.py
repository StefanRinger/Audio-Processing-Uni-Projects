import numpy as np

def test_AC(func):
    test_frame = [-1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0]
    lpc_order = 8
    estimated_out = [1, 0.0, 0.9230769230769231, 0.0, 2.391249591500338e-16, 0.0, 5.828670879282072e-16, 0.0, 0.07692307692307729]

    test_lpc_coeffs_AC = func(test_frame, lpc_order)
    return np.allclose(estimated_out, test_lpc_coeffs_AC)


def test_LD(func):
    test_frame = [-1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0]
    lpc_order = 8
    estimated_out = [1, 0.0, 0.9230769230769231, 0.0, 2.391249591500338e-16, 0.0, 5.828670879282072e-16, 0.0, 0.07692307692307729]

    test_lpc_coeffs_ld = func(test_frame, lpc_order)
    return np.allclose(estimated_out, test_lpc_coeffs_ld)


def test_formants(func):
    test_lpc = [1, 0.0, 0.9230769230769231, 0.0, 2.391249591500338e-16, 0.0, 5.828670879282072e-16, 0.0, 0.07692307692307729]
    estimated_out = np.array([0.011473177123205891+0.8323403782470172j,
                              -0.011473177123206394+0.8323403782470116j,
                              -0.561863850275283+0.29081020669423574j])
    test_formants = func(test_lpc)
    return np.allclose(estimated_out,  test_formants)


def test_complex2hertz(func):
    complex_numbers = [1, 1j, -1, -1j]
    estimated_out = np.array([ 0., 25., 50., -25.])
    test_hertz = func(complex_numbers, 100)
    return np.allclose(estimated_out, test_hertz)


def test_CalcResPredGain(func):
    test_frame = np.array([-1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0])
    test_lpc = np.array([1, 0.0, 0.9230769230769231, 0.0, 2.391249591500338e-16, 0.0, 5.828670879282072e-16, 0.0, 0.07692307692307729])
    output_res = np.array([-1., 0., 0.07692308, 0., -0.07692308, 0., 0.07692308, 0., -0.15384615, 0., 0.15384615, 0., -0.15384615, 0., 0.15384615, 0.])
    test_predGain = 8.56818842342
    res1, predGain = func(test_lpc, test_frame)
    res = np.array(res1)
    return np.allclose(output_res, res) and np.allclose(test_predGain, predGain)

