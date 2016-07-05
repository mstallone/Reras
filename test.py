import numpy as np
import scipy as sp
import os

import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy import signal
import sunau as au  # allows us o read in .au files

# let's calculate a simple FFT
fs = 8000
T = 1.0 / fs
n = np.array(range(fs / 8))
f = 500  # fundamental freq

# make our sinusoid
x = np.sin(2.0 * np.pi * f * n * T)
x = x * signal.hann(len(x))

# let's plot
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(n * T, x)
# plt.show()

fftLen = np.int64(pow(2, np.ceil(np.log(len(x)) / np.log(2))))
print fftLen, len(x)

# compute the fft
X = np.fft.fft(x, n=fftLen)
print X.dtype
X = X[0: fftLen / 2 + 1]

Xmag = np.absolute(X)
Xmag = 20 * np.log10(Xmag)
Xmag = Xmag - np.amax(Xmag)
Xmag[Xmag < -60] = -60

freqs = np.linspace(0, fs / 2, len(X))

plt.subplot(1, 2, 2)
plt.plot(freqs, Xmag)
plt.show()

# read in the wave files
classical = read("classical.00000.wav")
hiphop = read("hiphop.00000.wav")
speech = read("voices.wav")

# get the sample rates
classicalSR = classical[0]
hiphopSR = hiphop[0]
speechSR = speech[0]

print classicalSR, hiphopSR, speechSR

start = 0
end = 1
classical1 = classicalVals[start * SR]

fftLen = np.int64pow(2, np.ceil(np.log(len(classical1)) / np.log(2)))
freqs = np.linspace(0, SR / 2, len(range(fftLen / 2 + 1)))
lastFreq = 8000
lastInd = len(freqs(freqs < lastFreq))
