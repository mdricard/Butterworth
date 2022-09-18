import numpy as np
import math

def low_pass(raw, sampling_rate, filter_cutoff):
    """
    From the 4th edition of Biomechanics and Motor Control of Human Movement
    by David A. Winter p 69 for filter coefficient corrections.
    This algorithm implements a 4th order zero-phase shift recursive
    Butterworth low pass filter.  Last edited 9-18-2022

    Input parameters
        raw[] is a numpy array containing noise to be removed
        sampling_rate in Hz of the raw[] signal
        filter_cutoff in Hz for the low pass filter

    Output parameters
        smooth[] filtered result
    """
    n = len(raw)
    temp = np.zeros(n + 4, dtype=float)
    prime = np.zeros(n + 4, dtype=float)
    smooth = np.zeros(n, dtype=float)
    sr = sampling_rate
    fc = filter_cutoff
    nPasses = 2.0

    cw = (2.0 ** (1.0 / nPasses) - 1.0) ** (1.0 / 4.0)
    wc = math.tan(math.pi * fc / sr) / cw
    K1 = math.sqrt(2.0) * wc
    K2 = (wc) ** 2.0
    a0 = K2 / (1.0 + K1 + K2)
    a1 = 2.0 * a0
    a2 = a0
    K3 = 2.0 * a0 / K2
    b1 = -2.0 * a0 + K3
    b2 = 1.0 - 2.0 * a0 - K3

    temp[0] = raw[0] + (raw[0] - raw[1])
    temp[1] = raw[0] + (raw[0] - raw[2])
    temp[n + 3] = raw[n - 1] + (raw[n - 1] - raw[n - 2])
    temp[n + 2] = raw[n - 1] + (raw[n - 1] - raw[n - 3])

    for i in range(0, n):
        temp[i + 2] = raw[i]
    for i in range(0, (n + 4)):
        prime[i] = temp[i]
    for i in range(2, (n + 4)):
        prime[i] = a0 * temp[i] + a1 * temp[i - 1] + a2 * \
                   temp[i - 2] + b1 * prime[i - 1] + b2 * prime[i - 2]
    for i in range(0, (n + 4)):
        temp[i] = prime[i]
    for i in range((n + 1), -1, -1):
        prime[i] = a0 * temp[i] + a1 * temp[i + 1] + a2 * \
                   temp[i + 2] + b1 * prime[i + 1] + b2 * prime[i + 2]
    for i in range(0, n):
        smooth[i] = prime[i + 2]
    return smooth


def single_pass(raw, sampling_rate, filter_cutoff):
    """
    From the 4th edition of Biomechanics and Motor Control of Human Movement
    by David A. Winter p 69 for filter coefficient corrections.
    This algorithm implements a 2nd order single pass recursive
    Butterworth low pass filter.  The algorithm will produce a phase shift.
    Last edited 9-18-2022

    Input parameters
        raw[] is a numpy array containing noise to be removed
        sampling_rate in Hz of the raw[] signal
        filter_cutoff in Hz for the low pass filter

    Output parameters
        smooth[] filtered result
    """
    n = len(raw)
    temp = np.zeros(n + 4, dtype=float)
    prime = np.zeros(n + 4, dtype=float)
    smooth = np.zeros(n, dtype=float)
    sr = sampling_rate
    fc = filter_cutoff
    nPasses = 1.0

    cw = (2.0 ** (1.0 / nPasses) - 1.0) ** (1.0 / 4.0)
    wc = math.tan(math.pi * fc / sr) / cw
    K1 = math.sqrt(2.0) * wc
    K2 = (wc) ** 2.0
    a0 = K2 / (1.0 + K1 + K2)
    a1 = 2.0 * a0
    a2 = a0
    K3 = 2.0 * a0 / K2
    b1 = -2.0 * a0 + K3
    b2 = 1.0 - 2.0 * a0 - K3

    temp[0] = raw[0] + (raw[0] - raw[1])
    temp[1] = raw[0] + (raw[0] - raw[2])
    temp[n + 3] = raw[n - 1] + (raw[n - 1] - raw[n - 2])
    temp[n + 2] = raw[n - 1] + (raw[n - 1] - raw[n - 3])

    for i in range(0, n):
        temp[i + 2] = raw[i]
    for i in range(0, (n + 4)):
        prime[i] = temp[i]
    for i in range(2, (n + 4)):
        prime[i] = a0 * temp[i] + a1 * temp[i - 1] + a2 * \
                   temp[i - 2] + b1 * prime[i - 1] + b2 * prime[i - 2]
    for i in range(0, n):
        smooth[i] = prime[i + 2]
    return smooth


def critically_damped(raw, sampling_rate, filter_cutoff):
    """ algorithm implements a 20th order recursive critically damped
        low pass zero-lag Butterworth filter.

        Robertson DG, Dowling JJ (2003) Design and responses of Butterworth and critically
         damped digital filters. J Electromyograph & Kinesiol; 13, 569 - 573.

    Input parameters
        raw[] is a numpy array containing noise to be removed
        sampling_rate in Hz of the raw[] signal
        filter_cutoff in Hz for the low pass filter

    Output parameters
        smooth[] filtered result
    """
    n = len(raw)
    temp = np.zeros(n + 4, dtype=float)
    prime = np.zeros(n + 4, dtype=float)
    smooth = np.zeros(n, dtype=float)
    sr = sampling_rate
    fc = filter_cutoff
    nPasses = 5.0  # five double (forward & backward) passes

    cw = math.sqrt((2.00 ** (1.00 / (2 * nPasses))) - 1.00)
    fc = filter_cutoff
    wc = math.tan(math.pi * fc / sr) / cw
    K1 = 2.00 * wc
    K2 = wc * wc
    a0 = K2 / (1.0 + K1 + K2)
    a1 = 2.00 * a0
    a2 = a0
    K3 = 2.00 * a0 / K2
    b1 = 2.00 * a0 * ((1.0 / K2) - 1.00)
    b2 = 1.00 - (a0 + a1 + a2 + b1)
    # --------------------------------------------------------------
    #                           Pass 1
    # --------------------------------------------------------------
    temp[0] = raw[0] + (raw[0] - raw[1])
    temp[1] = raw[0] + (raw[0] - raw[2])
    temp[n + 3] = raw[n - 1] + (raw[n - 1] - raw[n - 2])
    temp[n + 2] = raw[n - 1] + (raw[n - 1] - raw[n - 3])

    for i in range(0, n):
        temp[i + 2] = raw[i]
    for i in range(0, (n + 4)):
        prime[i] = temp[i]
    for i in range(3, (n + 4)):
        prime[i] = a0 * temp[i] + a1 * temp[i - 1] + a2 * temp[i - 2] + b1 * prime[i - 1] + b2 * prime[i - 2]
    for i in range(0, (n + 4)):
        temp[i] = prime[i]
    for i in range((n + 1), -1, -1):
        prime[i] = a0 * temp[i] + a1 * temp[i + 1] + a2 * \
                   temp[i + 2] + b1 * prime[i + 1] + b2 * prime[i + 2]
    # --------------------------------------------------------------
    #                           Pass 2
    # --------------------------------------------------------------
    temp[0] = prime[2] + (prime[2] - prime[3])
    temp[1] = prime[2] + (prime[2] - prime[4])
    temp[n + 3] = prime[n + 1] + (prime[n + 1] - prime[n])
    temp[n + 2] = prime[n + 1] + (prime[n + 1] - prime[n - 1])

    for i in range(0, n):
        temp[i + 2] = prime[i + 2]
    for i in range(0, (n + 4)):
        prime[i] = temp[i]
    for i in range(3, (n + 4)):
        prime[i] = a0 * temp[i] + a1 * temp[i - 1] + a2 * \
                   temp[i - 2] + b1 * prime[i - 1] + b2 * prime[i - 2]
    for i in range(0, (n + 4)):
        temp[i] = prime[i]
    for i in range((n + 1), -1, -1):
        prime[i] = a0 * temp[i] + a1 * temp[i + 1] + a2 * \
                   temp[i + 2] + b1 * prime[i + 1] + b2 * prime[i + 2]

    # --------------------------------------------------------------
    #                           Pass 3
    # --------------------------------------------------------------
    temp[0] = prime[2] + (prime[2] - prime[3])
    temp[1] = prime[2] + (prime[2] - prime[4])
    temp[n + 3] = prime[n + 1] + (prime[n + 1] - prime[n])
    temp[n + 2] = prime[n + 1] + (prime[n + 1] - prime[n - 1])

    for i in range(0, n):
        temp[i + 2] = prime[i + 2]
    for i in range(0, (n + 4)):
        prime[i] = temp[i]
    for i in range(3, (n + 4)):
        prime[i] = a0 * temp[i] + a1 * temp[i - 1] + a2 * \
                   temp[i - 2] + b1 * prime[i - 1] + b2 * prime[i - 2]
    for i in range(0, (n + 4)):
        temp[i] = prime[i]
    for i in range((n + 1), -1, -1):
        prime[i] = a0 * temp[i] + a1 * temp[i + 1] + a2 * \
                   temp[i + 2] + b1 * prime[i + 1] + b2 * prime[i + 2]

    # --------------------------------------------------------------
    #                           Pass 4
    # --------------------------------------------------------------
    temp[0] = prime[2] + (prime[2] - prime[3])
    temp[1] = prime[2] + (prime[2] - prime[4])
    temp[n + 3] = prime[n + 1] + (prime[n + 1] - prime[n])
    temp[n + 2] = prime[n + 1] + (prime[n + 1] - prime[n - 1])

    for i in range(0, n):
        temp[i + 2] = prime[i + 2]
    for i in range(0, (n + 4)):
        prime[i] = temp[i]
    for i in range(3, (n + 4)):
        prime[i] = a0 * temp[i] + a1 * temp[i - 1] + a2 * \
                   temp[i - 2] + b1 * prime[i - 1] + b2 * prime[i - 2]
    for i in range(0, (n + 4)):
        temp[i] = prime[i]
    for i in range((n + 1), -1, -1):
        prime[i] = a0 * temp[i] + a1 * temp[i + 1] + a2 * \
                   temp[i + 2] + b1 * prime[i + 1] + b2 * prime[i + 2]

    # --------------------------------------------------------------
    #                           Pass 5
    # --------------------------------------------------------------
    temp[0] = prime[2] + (prime[2] - prime[3])
    temp[1] = prime[2] + (prime[2] - prime[4])
    temp[n + 3] = prime[n + 1] + (prime[n + 1] - prime[n])
    temp[n + 2] = prime[n + 1] + (prime[n + 1] - prime[n - 1])

    for i in range(0, n):
        temp[i + 2] = prime[i + 2]
    for i in range(0, (n + 4)):
        prime[i] = temp[i]
    for i in range(3, (n + 4)):
        prime[i] = a0 * temp[i] + a1 * temp[i - 1] + a2 * \
                   temp[i - 2] + b1 * prime[i - 1] + b2 * prime[i - 2]
    for i in range(0, (n + 4)):
        temp[i] = prime[i]
    for i in range((n + 1), -1, -1):
        prime[i] = a0 * temp[i] + a1 * temp[i + 1] + a2 * \
                   temp[i + 2] + b1 * prime[i + 1] + b2 * prime[i + 2]

    for i in range(0, n):
        smooth[i] = prime[i + 2]
    return smooth  # return the smoothed raw
 
 

 