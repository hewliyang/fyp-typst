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
  - $x[n]$ is the amplitude of the signal $x(t)$ at time t=$n\T$ in dB

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

  As the maximum audible frequency by humans are $<=$ 8 kHz, sampling speech at 16kHz is the minimum $f_s$ that satisfies the theorem, and hence is a typical choice for tasks such as TTS.

To summarize, the result of the Analog to Digital conversion process is simply a 1D vector of shape $[1, t times f_s]$, where $t$ is the length of a signal in seconds with values that can be integers or floats.

=== FFT

=== STFT


== Representations for Deep Learning
#lorem(120)

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