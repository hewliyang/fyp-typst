#import "@preview/showybox:2.0.1": showybox
#set heading(numbering: "1.")

= Preliminaries

This section includes the fundamental concepts and techniques essential for understanding TTS systems and how they can be evaluated, providing a rigorous foundation for subsequent discussions.

== Audio Signals

Audio signals are continuous waveforms that represent variations in sound pressure over time. In digital systems, these signals undergo a process of conversion and processing to enable computational analysis and synthesis.

=== Analog to Digital

The process of converting continuous analog signals to discrete digital representations involves three key steps.

1. *Sampling*

  #figure(
    image("assets/sampling.svg", width: 100%),
    caption: [
      Analog to digital sampling from a 5Hz sine wave
    ],
  )

  A continuous-time signal $x(t)$ is sampled at discrete time intervals, resulting in a sequence of samples $x[n]$ where $n$ is the sample index. The sampling process can me mathematically represented as

  $
    x[n] = x(n\T)
  $

  Additionally,

  - $T$ is the sampling period
  - $f_s = 1/T$ is the sampling rate, which is defined as the number of samples taken per second.
  - $x[n]$ is the amplitude of the signal $x(t)$ at time t=$n\T$

  A larger $f_s$ preserves more detail of the original signal which may be helpful for accurately reconstructing the original signal in certain applications such as high fidelity audio streaming, but imposes a computational and storage penalty.

  For example, at a sampling rate $f_s$ of 48,000 kHz, a mere 10-second audio clip would result in a vector of size $10 times 48,000$ samples, ie: shape `(1, 480_000)` or `(480_000,)`

  On the other hand, a smaller sampling rate may lose important details in the original signal, especially if the sampling rate drops below the *Nyquist* rate, which can negatively impact downstream performance.

  In general, most TTS datasets such as in @tab-tts-datasets such as Gigaspeech @GigaSpeech2021
  are typically resampled to 16kHz, which strikes a good balance between detail and computational efficiency.

  Additionally, a raw discretized waveform, even at this sampling rate is still unsuitable for sequence modelling due to the high temporal resolution. Instead, the waveform is typically converted into an intermediary representation, such as a spectrogram which is temporally more compact.

2. *Quantization*:

  The amptitude for each sample is approximated to a finite set of values. For a $b$-bit quantizer, the quantization levels are typically

  $
    Q = {-1 + (2k) / (2^b - 1)}:k=0,1,...,2^b-1
  $

  Similar to sampling frequency, a higher bit depth $b$ allows for a more accurate recording of the true amplitude of the sound wave, with similar trade offs in storage and computation.

  For example, a 16-bit quantization would provide 65,536 bins, or a range of [-32768, 32768]. Evidently, there will be some information loss due to quantization to some extent depending on the choice of $b$.

  As machine learning models typically operates in floating point operations, this range can be normalized to $[-1, 1]$ at the precision level of choice.

  In audio processing libraries such as `librosa` and `torchaudio`, the default data type assigned when reading in audio files is a `float32` which is equivalent to a bit depth of 24.

  #showybox(
    title: [Documentation for `librosa.load()` method],
    frame: (
      border-color: blue,
      title-color: blue.lighten(30%),
      body-color: blue.lighten(95%),
      footer-color: blue.lighten(80%),
    ),
    footer: "Method signature extracted from documentation version 0.10.2 (Sept 2024)",
  )[
    ```
    librosa.load(path, *, sr=22050, mono=True, offset=0.0, duration=None, dtype=<class 'numpy.float32'>, res_type='soxr_hq')
    ```
  ]

3. *Nyquist-Shannon Sampling Theorem*:

  To accurately represent a signal with a maximum frequency component of $f_{max}$, the Nyquist-Shannon Sampling Theorem states that the sampling rate $f_s = 1 / T$ must satisfy $f_s >= 2f_{max}$

  As the maximum audible frequency by humans are $<=$ 8 kHz, sampling speech at 16kHz is the minimum $f_s$ that satisfies the theorem, and hence is a typical choice for tasks such as TTS and ASR

To summarize, the result of the Analog to Digital conversion process is simply a 1D vector of shape $[1, t times f_s]$, where $t$ is the length of a signal in seconds with values that can be integers or floats.

=== Fourier Transform

The *Continuous Fourier Transform (CFT)* is used to decompose a continuous signal such as an audio signal into its constituent frequencies. It is defined as:

$
  X(f) = integral_(-infinity)^infinity x(t) exp{-i 2 pi} upright(d)t
$<eq-fourier-transform>

Where:

- $t$ is time
- $f$ is frequency
- $x(t)$ is the input signal in the time domain (amplitude vs time)
- $X(f)$ is the complex valued frequency domain representation of the signal (amplitude vs frequency)

In audio processing, the frequency domain representation of audio signals are often more informative compared to it's time domain counterpart for identifying types of sounds in a signal.

As we have established above, in digital signal processing we work with *discrete* signals. Similarly, the *Discrete Fourier Transform (DFT)* is defined as:

$
  X[k] = sum_(n=0)^(N-1)x[n] dot exp{-i 2 pi k / N n}
$
for $k=0,1,..., N-1$ where:

- $x[n]$ is the discrete input signal of sample $n$
- $X[k]$ is the $k$-th frequency component of the DFT
- $N$ is the total number of samples

Notably, a key property of fourier transformers is that they are invertible, and hence can be used for instance in removing unwanted frequencies in audio by flattening peaks in $X[k]$, then applying the inverse CFT or DFT to obtain a cleaned signal x[n].

Additionally, as the transforms produce complex values and since in the audio domain we generally are only concerned with power of signals, it is not uncommon to only consider the absolute values i.e. $abs(x[n])$.

=== Fast Fourier Transform (FFT)

#grid(
  columns: 2,
  figure(
    caption: "Sum of 2 sine waves at different frequencies",
    kind: image,
    [#image("assets/simple-wave.svg")],
  ),
  figure(
    caption: "Resulting FFT of Figure 2",
    kind: image,
    [#image("assets/simple-wave-fft.svg")],
  ),
)

The FFT is an efficient implementation of the DFT, and is commonly implemented in mathematical and signal processing libraries like `numpy`, `scipy`, `librosa` and `torchaudio` as a manifestation of the works of @cooley1965algorithm.

The time complexity of the proposed algorithm is $O(n log N)$, while a brute force approach would require $O(N^2)$.

=== Short Time Fourier Transform (STFT) <section-stft>

While the FFT is powerful, it's limitation is that it assumes the signal is stationary, i.e. the frequency content does not change over time. However, it is obvious that many real-world signals like speech are in fact non-stationary.

The STFT addresses this by applying the FFT to short segments of the signal over time. It can be expressed mathematically as:

$
  "STFT"{x[n]}(m,k) = sum_(n=-infinity)^infinity x[n]w[n-m] exp{-i 2 pi k / N n}
$

where
- $x[n]$ is the input signal
- $w[n]$ is the window function
- $m$ is the time index of the window
- $k$ is the frequency index

In practice,

1. The signal is divided into shorter segments which are often overlapping using a window function, in order to avoid losing information at the edges
2. FFT is applied to each segment
3. The results are concatenated to create a time-frequency representation

When using a library such as `librosa`, the primary parameters that one should pay attention to is:
- `n_fft`: number of samples per window, i.e. size of the window
- `hop_length`: number of samples to skip until the next window, i.e. the higher the value, the lower the overlap between windows and vice versa

The resulting vector or tensor, i.e. the linear spectrogram takes the shape

$
  ["n_fft" / 2 + 1, floor((T dot f_s - "n_fft") / "hop_length") + 1]
$

where the first dimension corresponds to the frequency bins while the second correspond to the time frames. The magnitude of each element in the spectrogram i.e. `spectrogram[t][f]` corresponds to the amplitude of the frequency $f$ at time $t$.

#figure(
  caption: "STFT of a sample from LibriSpeech",
  kind: image,
  [#image("assets/linear-spectrogram.svg")],
) <image-linear-spectrogram-speech>

#figure(
  caption: "STFT of an instrument (trumpet)",
  kind: image,
  [#image("assets/linear-spectrogram-trumpet.svg")],
) <image-linear-spectrogram-instrument>

@image-linear-spectrogram-speech shows the STFT of a 10 second speech sample from the LibriSpeech dataset while @image-linear-spectrogram-instrument shows the STFT of a clip of a trumpet, illustrating the feature rich representation that a spectrogram can provide.

== Representations for Deep Learning

In @section-stft, we have explored a method of representing raw discrete waveforms

== Flow Matching
#lorem(120)

== Autogressive/Causal Sequence Models
#lorem(120)

== Diffusion
#lorem(120)

== Overview of TTS Architectures
#lorem(120)

=== Tacotron 1 & 2
#lorem(120)

=== Speech T5
#lorem(120)

=== VITS
#lorem(120)

=== Tortoise TTS
#lorem(120)

== Evaluating TTS Systems
#lorem(120)

=== Subjective Evaluation
#lorem(120)

=== Objective Metrics
#lorem(120)

=== Predictor Networks
#lorem(120)

=== Self Supervised Networks
#lorem(120)

=== Survey of the latest TTS papers
#lorem(120)