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

The *Fourier Transform* is used to decompose a continuous signal such as an audio signal into its constituent frequencies. It is defined as:

$
  X(f) = integral_(-infinity)^infinity x(t) exp{-i 2 pi} upright(d)t
$<eq-fourier-transform>

Where:

- $t$ is time
- $f$ is frequency
- $x(t)$ is the input signal in the time domain (amplitude vs time)
- $X(f)$ is the complex valued frequency domain representation of the signal (amplitude vs frequency)

In audio processing, the frequency domain representation of audio signals is often more informative compared to it's time domain counterpart for identifying types of sounds in a signal.

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
$ <eq-spectrogram-shape>

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

@image-linear-spectrogram-speech and @image-linear-spectrogram-instrument illustrates the interpretable nature of spectrograms, as the difference between human speech and music instruments can be easily differentiated compared to more primitive forms such as the discrete waveform.

== Representations of Discrete Audio Signals

In deep learning for audio processing, the choice of input representation can signifcantly impact model performance and efficiency. In @section-stft, we have explored a method of representing raw discrete waveforms as spectrograms using the STFT.

Indeed, this is a valid format for further processing by battle tested techniques from adjacent fields such as Convolutional Neural Networks (CNN) in computer vision. In addition, as the STFT is a non-lossy operation, enabling the resulting spectrogram can be easily inverted via the inverse STFT back to a waveform, without requiring the use of a vocoder.

However, in practice, both raw waveforms and linear spectrograms are rarely used for tasks such as TTS due to
1. High dimensionality on the frequency scale - requires more compute and memory for processing
2. Perceptual misalignment with the human auditory system - humans are more perceptive to differences lower frequencies compared to higher frequencies.

Mel-spectrograms address both of these issues, but it's transformaton operation is lossy and hence requires the use of a vocoder to invert the mel-spectrogram back into an audible waveform. Despite this, they are widely used across many of the earlier prominent works such as Tacotron @wang2017tacotronendtoendspeechsynthesis. VITS @kim2021conditionalvariationalautoencoderadversarial utilises both linear spectrograms as input to a posterior encoder, but uses mel-spectrograms for computing the reconstruction loss during training.

WaveNet @oord2016wavenetgenerativemodelraw and achieved success utilizing raw waveforms while modern language model inspired causal decoders such as @betker2023betterspeechsynthesisscaling rely on discrete audio tokens, which are learnt representations from a VQVAE.

=== Mel Spectrograms

Mel spectrograms are a popular representation that captures both time and frequency information, tailored to human auditory perception. The difference between a linear spectrogram and mel spectrograms are simply a projection of the frequency dimension onto the mel scale.

The mel scale is designed to better represent how humans perceive frequency differences. Chiefly:

- Humans are more sensitive to small changes in pitch at lower frequencies than at higher frequencies.
- THe mel scale is approximately linear below 1000Hz and logarithmic above 1000Hz.

The conversion from Hz to mel is given by:

$
  m = 2595 log_10 (1 + f / 700)
$

where $m$ is the mel scale value and $f$ is the frequency in Hz.

Transforming a discrete waveform into a mel-spectrogram entails largely the same recipe as the STFT, with a few extra steps which are applying a mel filterbank, then logarithmic scaling.

Mathematically,

$
  "S"_"mel" = log(M dot.c abs("STFT(x)")^2)
$ <eqn-mel-transform>

where $M$ is the transformation matrix and $x$ is the discrete input signal.

#figure(
  caption: "Visualization of triangular mel filter banks",
  kind: image,
  [#image("assets/mel-filter-banks.svg", height: 200pt)],
) <image-mel-filterbanks>

The operation in @eqn-mel-transform can be thought of as multiple 1D convolutions on the linear spectrogram, where each filter bank represented by the different colors corresponds to one convolution kernel. Equivalently, each kernel would be represented as a row of the transformation matrix $M$.

Let us denote the number of filter banks or the number of bands as $n_"mels"$. Also recall @eq-spectrogram-shape. The resulting shape of the mel spectrogram is now:

$
  [n_"mels", floor((T dot f_s - "n_fft") / "hop_length") + 1]
$

such that the frequency spectrum has been quantized to a chosen $n_"mels"$.

The choice of $n_"mels"$ is depends on the use case. For example both @wang2017tacotronendtoendspeechsynthesis and @kim2021conditionalvariationalautoencoderadversarial both choose a value of 80 bands for training Tacotron and VITS respectively.

Also recall @image-linear-spectrogram-instrument. Applying the mel-scale transform with $n_"mels"=128$, we get

#figure(
  caption: [Mel scale spectrogram of @image-linear-spectrogram-instrument],
  kind: image,
  [#image("assets/mel-spectrogram.svg")],
)

=== Mel Frequency Cepstral Coefficients (MFCC)

MFCCs are a compact representation of the spectral envelope of a signal and is also widely used in speech and music processing tasks. MFCCs build upon the mel spectrogram representation, applying additional transformations in order to capture the most salient features of the signal.

The process of computing MFCCs involves:
1. Applying the Discrete Cosine Transform (DCT) to the log mel spectrogram
2. Keep only the first N coefficients, which is typically 13-20 for speech applications

$
  "MFCC" = "DCT"("S"_"mel")[:N]
$

The DCT can be thought of as a way to seperate the overall shape of the spectrum which is captured by lower order coefficients from the fine details, which are captured by the higher order coefficients.

In the context of evaluation, MFCCs can be used to compute the Mel Cepstral Distortion metric (MCD), which is a pairwise metric measuring the spectral difference between two signals. Concretely,

$
  "MCD" = (10 / ln(10)) sqrt(2 sum_(n=1)^N (c_n - c'_n)^2)
$

where $c_n$ and $c'_n$ are the $n^("th")$ coefficients of the natural and synthesized speech respectively. The lower the MCD, the higher their spectral similarity.

=== Discrete Audio Tokens

Taking a language modelling approach for speech generation was first introduced in Tortoise @betker2023betterspeechsynthesisscaling, who indentified the opportunity to leverage the same recipe detailed in the original DALLE-1 paper by @ramesh2021zeroshottexttoimagegeneration.

More specifically, Betker firstly trained a VQVAE to learn latent, discrete representations of speech signals via a mel spectrogram reconstruction loss, producing an intermediate representation which he described as MEL tokens.

==== VQVAE

The VQVAE architecture consists of three main components which are analogous to the original paper by @oord2018neuraldiscreterepresentationlearning:

#figure(
  caption: "Original figure describing VQ-VAE from 2017 DeepMind paper altered with inputs as spectrograms instead of images",
  kind: image,
  [#image("assets/vqvae.png")],
)

1. An encoder that maps the input signal (mel spectrogram) to a latest space
2. A vector quantization layer that discretizes the continuous latent representations using *codebooks*.
3. A decoder that reconstructs the audio from the quantized representations.

A codebook is a finite set of learned vector embeddings. During the forward pass, each latent vector produced by the encoder is replaced by it's nearest neighbour in the codebook, i.e. for an input latent $z_(e(x)) in RR^n$ produced by the encoder network:

$
  q(z_e (x)) = op("arg min", limits: #true)_(k) ||z_e (x) - e_k||_2
$

where $e_k in RR^n$ are the codebook entries, $n$ is the embedding dimension of the codes, while $k$ is the codebook size. A larger codebook allows for finer representation of the audio signal but increases computational complexity. For completeness, it is worth to note that in @betker2023betterspeechsynthesisscaling,

- $k = 8192$
- $n = 256$


==== Autoregressive Prior

Secondly, an autoregressive decoder, specifically GPT-2 @radford2019language was trained using the next token prediction objective on these MEL tokens conditioned on text token labels.

$
  P(x_1, ..., x_T | y_1, ..., y_Q) = product_(t=1)^T P(x_t | x_1, ..., x_(t-1), y_1, ... y_Q)
$

During inference, regular text tokens $y_i$ are passed to the AR decoder as inputs, and MEL tokens $x_i$ are sampled as outputs before being passed to the decoder and subsequently vocoder to be transformed into an audible waveform.

This approach allows the model to generate each token conditioned on all previously generated tokens, capturing long-range dependencies in the audio signal. Indeed, the benchmarks indicate superior consistency and prosody and cadence compared to non-autoregressive models such as VITS.

Since then, there has been no shortage of similar systems such as AudioLM @borsos2023audiolmlanguagemodelingapproach, Base TTS @łajszczak2024basettslessonsbuilding, VALL-E @wang2023neuralcodeclanguagemodels, and XTTS by @casanova2024xttsmassivelymultilingualzeroshot which each has tricks of it's own but operates on the same foundations.

The mental model of such models are simple and proven to work at scale from the adjacent field of language modelling, but suffer from runtime & compute penalties due to the need for autoregressive decoding. Namely, it is slow and memory requirements scale with the input length at both training and inference time.

Additionally, recent discrete neural audio codecs such as Encodec @défossez2022highfidelityneuralaudio have shown state of the art results in the audio compression-decompression space even in the out of domain cases by employing the same latent space quantization techniques. In fact, the codes generated by these models can be used off the shelf as audio tokens directly for causal modelling which is demonstrated by @wang2023neuralcodeclanguagemodels in VALL-E. This is to say, the weights of the codec model can be frozen during training.

#figure(
  caption: "Architecture of VALL-E",
  kind: image,
  [#image("assets/vall-e.png", height: 200pt)],
)

At the same time, no bespoke decoder or vocoder networks is required to invert the tokens back to a waveform. Simply passing the sampled codes back into the frozen Encodec decoder reconstructs the waveform.

On the other hand, systems such as VALL-E suffer from content inaccuracies and high word error rates (WER). This phenomenon is in part, due to the fact that acoustic codecs are designed primarily for audio compression and the latents prioritize capturing acoustic fidelity over semantic richness as suggested by @ye2024codecdoesmatterexploring. The authors demonstrated that self-supervised features, such as those extracted by HuBERT @hsu2021hubertselfsupervisedspeechrepresentation can augment the codec latents with a richer semantic representation, leading to better performance.

== Evaluating TTS Systems

TTS systems are judged by their ability to produce accurate, natural-sounding and intelligible speech. These systems can be assessed using both subjective and objective methods.

Studies have shown however that objective metrics tend to not correlate well @vasilijevic2011perceptual with human perception, which necessitates the need for subjective, listening tests typically using volunteers (paid or unpaid).

Subjective tests require a human in the loop, meaning it is time-consuming, error prone and expensive. It is also not feasible to scale evaluation efforts for multi-system use cases. Additionally, it cannot be used during training to validate the real-time performance of systems while training @TTS-scores.

=== Absolute vs Relative Evaluations

*Absolute* evaluations, or reference-free evaluations access the performance of a TTS system on a fixed scale, independent of other systems or comparisons. They provide a direct evaluation of the model's quality, usually on predefined scales. They are used when evaluating systems in isolation, and when comparisons are not possible.

Examples of absolute evaluations include

- Mean Opinion Scores (MOS)
- Multiple Stimuli with Hidden Reference and Anchor (MUSHRA)
- Mel-Cepstral Distortion (MCD)
- Word Error Rate (WER)

These metrics are easy to interpret and are a direct measurement of performance, but can be influenced by listeners bias.

*Relative* evaluations compare the performance of one TTS system against the ground truths, i.e. a held-out test set or against other systems, focusing on preference and ranking rather than independent scores. It is used when the goal is to determine which system is better in a pairwise or multi-system comparison.

Common relative evaluations in TTS include:

- ABX tests: double blind trials to determine a perceptible difference between two signals
- Preference tests: participants are asked to compare samples $> 2$ samples and choose the highest performing sample (such as most natural) without any explicit scoring
- Spectral reconstruction error/loss: where $k=1$ corresponds to Mean Absolute Error (MAE) and $k=2$ corresponds to Mean Squared Error.

$
  L_"recon" = "MAE" = ||x_"mel" - hat(x)_"mel"||_k
$

A holistic evaluation set up typically uses a combination of subjective and objective metrics in order to provide a comprehensive understanding of the system's performance.

=== Subjective Evaluation

Subjective Evaluation involves human listeners assessing the quality of synthesized speech. These evaluations can focus on attributes such as naturalness, intelligibility or preference.

==== Mean Opinion Score (MOS)

MOS is a widely-used subjective metric where human listeners rate the quality of speech on a fixed Likert scale, typically from 1 to 5 where 1 is "bad" and 5 is "excellent". MOS is actually a subset of a classification of Absolute Category Rating (ACR) tests used historically in the telecommunications industry for assessing transmission quality @5946971.

Recommended experimental setups are inscribed in the ITU-T P.800 standard @ITU1996.

The MOS is computed as the average of all ratings.

$
  "MOS" = 1 / N sum_(i=1)^N r_i
$

where $r_i$ is the rating provided by the $i^"th"$ listener and $N$ is the total number of listeners.

#figure(
  caption: "Example of a MOS scoring rubric for naturalness",
  kind: table,
  [#table(
      columns: 2,
      [Rating], [Quality],
      [5], [Excellent],
      [4], [Good],
      [3], [Fair],
      [2], [Poor],
      [1], [Bad],
    )],
)

MOS can be used to evaluate multiple dimensions of speech and not just naturalness including intelligibility, speaker similarity, typically denoted by subscripting as $"MOS"_"sim"$ for example.

==== Multiple Stimuli Hidden Reference and Anchor (MUSHRA)

MUSHRA is a more rigorous subjective testing methodology compared to MOS, which also originates from evaluating lossy-compression in the telecommunications industry, described by ITU-R BS.1534-1 @itu_bs1534-3_2015. Listeners are presented with multiple stimuli, including a hidden reference and low quality anchor, and are instead asked to rate samples on a more fine-grained scale from 0 to 100.

In the context of TTS evaluations, the lower and upper anchor references are typically not used in order to save costs, while achieving a higher overall listening time per test such as in BASE TTS @łajszczak2024basettslessonsbuilding.

==== Analysis for statistical significance

Firstly, both MOS and MUSHRA requires careful execution of the experimental setup in order to ensure statistically significant results.

As recruiting a large number of volunteers in reality can be costly and time consuming, works such as @5946971 introduce toolkits for conducting such tests on crowdsourcing platforms such as Amazon's Mechanical Turk platform.

Secondly, the relevant statistical tests must be done after conducting said listenings tests. Typically, the Students t-test is applicable when comparing pairwise systems such as synthesized vs ground-truth or ANOVA for multi-system comparisons.

Correlation metrics such as Pearsons correlation coefficient (PCC) or Spearman's rank correlation coefficient (SPCC) are also typically computed to measure the perceived correlation compared to ground truths.

When reporting MOS and MUSHRA, it is also standard practice to report a confidence interval, i.e. $hat(u) plus.minus t sqrt("var"(hat(u))) $, where $hat(u)$ is the observed metric and $t$ is the correct percentile from the t-distribution.

==== ELO

The ELO rating system, originating from chess can be adapted for pairwise comparisons in TTS evaluation. In this context, listeners compare pairs of speech outputs, and systems receive ELO ratings based on the outcome of these comparisons.

In particular, assume a system $A$ with rating $E_A$ competes against system $B$ with rating $E_B$. Then, the expected score for system A is:

$
  E_A = 1 / (1 + 10^((E_B-E_A) / 400) )
$

If system $A$ wins, it's new rating is:

$
  R'_A = R_A + K * (S_A-E_A)
$

where

$
  S_A := cases(
  1 wide& "A wins",
  0.5 wide& "draw",
  0 wide& "otherwise",
)
$

An example of such as system is #link("https://huggingface.co/spaces/TTS-AGI/TTS-Arena")[TTS Arena] @tts-arena by HuggingFace, where members of the public are prompted to do pairwise comparisons on naturalness.


=== Objective Metrics

Objective metrics rely on automated methods to evaluate TTS systems. These metrics compare are mainly reference dependent, comparing synthesized speech against a ground truth.

==== Word Error Rate (WER)

WER is a common objective metric to evaluate the intelligibility of synthesized speech by measuring how well an ASR system transcribes it.

Mathematically, it is defined as:
$
  "WER" = (S + D + I) / N
$

Where:

- $S$ is the number of substitutions
- $D$ is the number of deletions,
- $I$ is the number of insertions
- $N$ is the total number of words in the reference transcript

WER is widely used due to it's simplicity, and the exceptional quality of recent ASR models like Whisper @radford2022robustspeechrecognitionlargescale.

==== Character Error Rate (CER)

CER is defined identically to WER, except at the character level instead of the world level. It is used when detecting small mistakes, particularly in languages with complex orthographies are critical.

==== Signal-to-Noise Ratio (SNR)

SNR is a basic metric that measures the ratio between the desired speech signal and background noise. Higher SNR indicates clearer and more intelligible speech. SNR can be used to identify generated noise in synthetic speech.

The SNR is computed as

$
  "SNR" = 10log_(10) (P_"signal" / P_"noise")
$

where $P$ represents power of the speech and noise signal respectively.

==== Perceptual Evaluation of Speech Quality (PESQ)

PESQ attempts to predict the quality of speech as perceived by human listeners, i.e. tries to mimic MOS. It compares synthesized speech with a reference signal and evaluates the perceptual quality and was standarzied in ITU-T P.862 @itu_pesq_p862_2001.

PESQ is a model-based metric, but does not actually have any learnable parameters and hence is still considered an objective metric. However, it is important to note that PESQ was originally designed to measure degradations in telecommunication transmissions, and hence may not generalize well for TTS prediction tasks.

The metric ranges from -0.5 to 4.5. A Python implementation of PESQ is made available by @miao_wang_2022_6549559.

==== Short-Term Objective Intelligibility (STOI)

STOI is yet another object measure used to predict the intelligibility of speech in noisy environments. It compares short-term temporal envelopes of the synthesized and reference speech signals, and provides a score between 0 and 1. A higher STOI indicates higher intelligibility.

$
  "STOI" = 1 / K sum_(k=1)^K "corr"(x_k,y_k)
$

where $x_k$ and $y_k$ are the time frames of the reference and synthesized signals respectively.

=== Survey of the latest TTS papers

For completeness, the following table enumerates the different types of metrics used in leading TTS architectures by their respective authors, showing the variance in testing rigour across the field.

#show figure: set block(breakable: true)

#figure(
  caption: [Evaluation Metrics Across TTS Systems],
  kind: table,
  [
    #set text(size: 10pt)
    #show table.cell.where(y: 0): strong
    #table(
      columns: 3,
      stroke: 0.5pt,
      [System], [Metric], [Description],
      table.cell(
        stroke: (bottom: 0pt),
        [FastSpeech 2],
      ),
      [Mean Opinion Score (MOS)],
      [Overall quality assessment of synthesized speech compared to ground truth],

      table.cell(
        stroke: (bottom: 0pt),
        [@ren2022fastspeech2fasthighquality],
      ), [Comparative MOS (CMOS)], [Relative naturalness comparison with other systems],
      table.cell(
        stroke: (bottom: 0pt),
        [],
      ),
      [Statistical Measures (σ, γ, K)],
      [Analysis of pitch accuracy using standard deviation, skewness, and kurtosis],

      [], [DTW Distance], [Similarity measurement between synthesized and ground truth pitch contours],
      table.cell(
        stroke: (bottom: 0pt),
        [StyleTTS 2],
      ), [MOS-N], [Assessment of speech naturalness/human-likeness],
      table.cell(
        stroke: (bottom: 0pt),
        [@li2023styletts2humanleveltexttospeech],
      ), [MOS-S], [Similarity evaluation for multi-speaker models],
      table.cell(
        stroke: (bottom: 0pt),
        [],
      ), [CMOS-N], [Comparative naturalness assessment between different configurations],
      table.cell(
        stroke: (bottom: 0pt),
        [],
      ), [MCD & MCD-SL], [Mel-cepstral distortion measurement, with and without speech length weighting],
      table.cell(
        stroke: (bottom: 0pt),
        [],
      ), [F0 RMSE], [Accuracy of pitch prediction],
      table.cell(
        stroke: (bottom: 0pt),
        [],
      ), [DUR MAD], [Accuracy of phoneme duration prediction],
      table.cell(
        stroke: (bottom: 0pt),
        [],
      ), [WER], [Speech recognition accuracy/intelligibility],
      table.cell(
        stroke: (bottom: 0pt),
        [],
      ), [CV (dur/f0)], [Diversity assessment through duration and pitch variation],
      [], [RTF], [Speed of synthesis relative to real-time],
      table.cell(
        stroke: (bottom: 0pt),
        [Voicebox],
      ), [WER], [Speech correctness and intelligibility measurement],
      table.cell(
        stroke: (bottom: 0pt),
        [@le2023voiceboxtextguidedmultilingualuniversal],
      ), [QMOS], [Subjective audio quality assessment],
      table.cell(stroke: (bottom: 0pt), []), [SMOS], [Speaker and style similarity evaluation],
      table.cell(stroke: (bottom: 0pt), []), [MS-MAE/MS-Corr], [Phoneme duration prediction accuracy and correlation],
      table.cell(stroke: (bottom: 0pt), []), [FDD/FSD], [Quality and diversity assessment of duration/speech samples],
      [], [SIM-r/SIM-o], [Audio similarity evaluation using WavLM-TDCNN],
      table.cell(
        stroke: (bottom: 0pt),
        [NaturalSpeech 3],
      ), [SIM-O/SIM-R], [Speaker similarity assessment with original/reconstructed prompts],
      table.cell(
        stroke: (bottom: 0pt),
        [@ju2024naturalspeech3zeroshotspeech],
      ), [UTMOS], [Objective substitute for MOS evaluation],
      table.cell(stroke: (bottom: 0pt), []), [WER], [Speech intelligibility measurement],
      table.cell(stroke: (bottom: 0pt), []), [MCD & MCD-Acc], [Prosodic similarity and emotion accuracy assessment],
      table.cell(stroke: (bottom: 0pt), []), [CMOS/SMOS], [Comparative naturalness and similarity evaluation],
      table.cell(stroke: (bottom: 0pt), []), [PESQ/STOI], [Perceptual quality and intelligibility measurement],
      [], [MSTFT], [Spectral distance measurement],
      table.cell(stroke: (bottom: 0pt), [CLaM TTS]), [CER/WER], [Character and word-level transcription accuracy],
      table.cell(
        stroke: (bottom: 0pt),
        [@kim2024clamttsimprovingneuralcodec],
      ), [SIM-o/SIM-r], [Speaker similarity using WavLM-TDCNN embeddings],
      table.cell(stroke: (bottom: 0pt), []), [QMOS/SMOS], [Quality and similarity assessment],
      table.cell(stroke: (bottom: 0pt), []), [CMOS], [Comparative quality evaluation],
      [], [PESQ/ViSQOL], [Objective speech quality metrics],
      table.cell(stroke: (bottom: 0pt), [XTTS]), [CER], [Pronunciation accuracy assessment],
      table.cell(
        stroke: (bottom: 0pt),
        [@casanova2024xttsmassivelymultilingualzeroshot],
      ), [UTMOS], [Predicted naturalness score],
      table.cell(stroke: (bottom: 0pt), []), [SECS], [Speaker similarity using ECAPA2 embeddings],
      [], [CMOS/SMOS], [Comparative naturalness and speaker similarity evaluation],
      table.cell(stroke: (bottom: 0pt), [BASE TTS]), [MUSHRA], [Quality comparison on 0-100 scale],
      table.cell(
        stroke: (bottom: 0pt),
        [@łajszczak2024basettslessonsbuilding],
      ), [Expert Evaluation], [Assessment of handling complex linguistic features],
      table.cell(stroke: (bottom: 0pt), []), [WER], [Speech intelligibility measurement],
      [], [SIM], [Speaker similarity evaluation],
    )
  ],
)