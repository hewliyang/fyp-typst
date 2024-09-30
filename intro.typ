#set heading(numbering: "1.")

= Introduction

== Terminology

#{
  set align(center)
  box(height: 32pt, width: 100%, rect([TODO]))
}
- Phoneme
- Spectrogram
- Prosody
- Timbre
- Tokens
- ASR
- TTS
- WER
- CER
- MSE
- MAE


== Text to Speech

Text-to-speech (TTS) is a fundamental task in speech technology that aims to convert written text into natural sounding speech. Formally, TTS can be defined as a function $f$ that maps a sequence of inputs $T = {t_1, t_2, ..., t_n}$ to a sequence of acoustic features $A={a_1, a_2, ..., a_m}$.

$
  f: T -> A
$<eq-tts-1>

where $t_i$ represents individual textual units which could be characters, phonemes, words, or tokens. $a_j$ represents acoustic features, which could be mel-spectograms, linear spectrograms or waveform samples.

TTS is well known as as the more challenging - inverse task of Automatic Speech Recognition (ASR). The reason for this is because TTS, or rather $f$ is a fundamentally one-to-many problem, i.e. for the same input $T$, there exists many linguistically "correct" $A$'s. More explicitly, the same sentence can be expressed verbally in a large, and arguably infinite number of ways due to differences in slang, accents, style, pace, and prosody that is unique to a human speaker.

#figure(
  caption: [Factors for the one-to-many nature of TTS],
  kind: table,
  [
    #table(
      stroke: none,
      columns: 2,
      table.hline(start: 0, stroke: 1pt),
      [Factor], [Description],
      table.hline(start: 0, stroke: 1pt),
      [Prosody], [Variations in intonation, stress, rhythm],
      [Timbre & Speaker Characteristics], [Voice qualities, speaking styles, accents],
      [Contextual Ambiguity], [Certain words can have multiple valid pronounciations, eg: "read" depending on present or past tense],
      table.hline(start: 0, stroke: 1pt),
    )<tab-one-to-many>
  ],
)

Adjusting @eq-tts-1, we reformulate $f$ as:
$
  f: T -> {A_1, A_2, ..., A_k}
$

where $A_i$ now represents the set of valid acoustic realizations.

== Implications on Training Data

Typically, a large amount of high quality, diverse voices is required in order to train a speech
model that can accurately capture the various possible realizations. These datasets are notably expensive and time consuming to aquire, with many requiring proprietary licensing or royalties paid to the original speakers.

On the other hand, efforts by organization such as *Librivox* and volunteer readers, there are also considerable high quality, openly available text $<=>$ speech datasets suitable for both ASR and TTS under public licenses.

#figure(
  caption: [Popular open source TTS datasets],
  kind: table,
  [
    #set text(size: 10.5pt)
    #show table.cell.where(y: 0): strong
    #table(
      columns: 5,
      [Name], [Type], [Languages], [Total Duration], [Multi Speaker?],
      [`lj_speech` @ljspeech17], [Audiobook], [`en`], [23h 55m], [No],
      [`LibriSpeech` @7178964], [Audiobook], [`en`], [$approx$ 1000h], [Yes],
      [MLS @Pratap2020MLSAL],
      [Audiobook],
      [`en`,`de`,`nl`,`es`,#linebreak()`fr`,`it`,`pt`,`pl`],
      [$approx$ 44.5hr `en` #linebreak() 6hr others],
      [Yes],

      [`VCTK` @Veaux2017CSTRVC], [Crowdsource, #linebreak()Volunteers], [`en`], [$approx$ 44h], [Yes],
      [`GigaSpeech` @GigaSpeech2021],
      [Audiobook, #linebreak()Podcasts,#linebreak()Youtube],
      [`en`],
      [$approx$ 10,000h],
      [Yes],

      [Emilia @emilia],
      [In the wild],
      [`en`,`zh`,`de`,`fr`,`jp`,`kr`],
      [$approx$ 46,000hr `en`, 50,000hr `zh`, 5000hr others],
      [Yes],
    )
  ],
)<tab-tts-datasets>

The advent of autogressive transformers and GPT has shined a new light on the possibilities of performance gains that comes with scaling, which the sizes of these datasets are not nearly enough for. Inspiration can be taken from large language models, which are trained on nearly all contents on the internet, and has been seen is newer datasets such as Emilia @emilia, where potentially noisy in the wild data are cleaned and annotated for training.

Secondly, the datasets presented in @tab-tts-datasets also are primary English and Chinese. Availability of high quality recordings for low resource languages remain an issue.


== Implications on Evaluation Techniques

=== Comparison to Automatic Speech Recognition (ASR)

In ASR, evaluating a model $g$ would be as simple as formulating an objective function that quantifies the disparity between the predicted text sequence $hat(T)$ and the ground truth $T$:

$
  E_"ASR" = d(T, hat(T))
$

where $d$ is typically a string distance metric like Word Error Rate (WER) or Character Error Rate (CER).

In contrast, TTS evaluations lack a straightforward objective function due to the one-to-many nature. For instance, let the set of all possible acoustic realizations be $bold(A) = {A_i}_"i=0"^k$ such that

$
  E_"TTS" = d(bold(A), hat(A))
$

where in this case, $d$ could be a the average L1 or L2 loss between the mel-spectrogram representation of each $(A_i, hat(A))$ pair, i.e. spectral loss.

However, coming up with the set $bold(A)$ in the first place is intractable because $k$ is unbounded.

=== Metric Dimensionality

ASR primarily focuses on transcription accuracy which can be captured by a single dimensional metric. However, TTS evaluation must consider multiple dimensions. In particular:

1. Intelligibility - how clear is the speech and how easy is it to understand
2. Naturalness - how close does it sound to human speech
3. Speaker similarity - degree of similarity of timbre compared to a reference voices in a multi-speaker system
4. Prosody - the appropriateness of intonation, stress and rythym.

=== Subjectivity

WER or CER is a metric that can be computed automatically and objectively. While some objectives metrics exist for TTS such as the Mel Cepstral Distortion, they often correlate poorly with human perception @vasilijevic2011perceptual. As a result, TTS evaluation relies heavily on subjective human judgements, typically in the form of Mean Opinion Scores (MOS) listening tests.

These tests typically involve human evaluators to rate the perceived "quality" of a speech sample on a Likert scale ie. 1 to 5, where higher is better. The resulting metrics can then be scrutinized by paired sample t-tests to check for statistical significance that a particular system is better than another.

Quality in this case can be defined by one or more dimension such as naturalness and intelligibility as mentioned above. Often, researchers tend to make different choices on which dimensions are chosen for evaluation.

In addition, the experimental setup for conducting these tests also vary from formal setups, such as in an controlled lab environments to online crowdsourced efforts such as by using Amazon's Mechanical Turk. This approach is commonly used even in evaluating frontier level TTS architectures such as StyleTTS 2 @li2023styletts2humanleveltexttospeech.

Often times, the latter approach is practically unavoidable due to the need for a large number of participants, time constraints and geographical constraints, such as when there is a limited number of available native speaking evaluators for a low-resource language @wells2024experimental.

This variation poses a challenge as it puts the validity of such test results into scrutiny. Additionally, comparing results for the same dimension such as naturalness across tests with fundamentally different setups can be brought into question @kirkland2023stuck, @chiang2023reportdetailssubjectiveevaluation.

Finally, variance in listener perception due to factors such as in @moore2013introduction:

- Cultural and linguistic background
- Attention and cognitive load
- Environmental factors such as noise and other acoustic conditions
- Proficiency in target languages
- Listening devices

means that for the same sample $x$, the set of ratings $hat(r)_i(x)$ where $i$ refers to the listener ID, may take any value in ${1,2,3,4,5}$. To prescribe MOS as a meaningful metric for evaluation, there is a need to ensure that the pool of listeners $n$ is large enough such that we can confidently assume the sample mean, i.e. $1/n sum_i^n hat(r_i)(x) -> r_i$ will converge to the population mean. For example, recent studies such as @wester15c_interspeech suggests a minimum of $n >= 30$ participants, including a 30 minute test coverage per participant in order to obtain statistical significant results.

== Motivations and Background

The proliferation of Text-to-Speech (TTS) technologies across diverse sectors has underscored the critical need for systematic performance evaluation methodologies. TTS applications have permeated various domains, including:

- Assistive technologies (e.g., aiding individuals with speech impairments to recover their voices)
- Education (facilitating accessible learning materials for visually impaired students)
- Healthcare (providing clear medical instructions and reminders)
- Entertainment and media (generating voiceovers for animations and video games)
- Customer service (powering intelligent virtual assistants)
- Transportation (delivering navigation instructions in vehicles)

This wide-ranging adoption necessitates a robust framework for assessing TTS performance, as different downstream use cases demand optimization along various dimensions, as mentioned above.

Traditional loss functions employed in machine learning, such as MSE and MAE, fail to capture the nuanced aspects of speech quality. These metrics primarily measure how well the generated speech approximates the training data but fall short in evaluating perceptual qualities that are crucial for human listeners. This limitation has given rise to the need for manual assessments based on human judgment, with the MOS emerging as a popular metric for this purpose.

Recent advancements in deep learning have accelerated research on data-driven approaches to predict these subjective metrics with remarkable accuracy. Neural models trained on large datasets of human-rated speech samples have shown promising results in automating the evaluation process. However, serious concerns remain regarding the performance of these supervised neural predictors on out-of-domain data, raising questions about their generalizability and reliability across diverse TTS systems and use cases.

The development of reliable evaluation systems holds significant potential to alleviate the reliance on manual human assessments to a certain extent. Moreover, these systems offer an objective means for evaluating TTS systems at scale, enabling rapid iteration and improvement in TTS technologies. This is particularly crucial as the field continues to evolve at an accelerated pace, with new architectures and techniques emerging regularly.

The challenges associated with MOS-based evaluations have led to the establishment of annually reoccuring competitions and challenges, such as the Blizzard Challenge @king2014measuring and the VoiceMOS Challenge @10389763. These initiatives provide valuable opportunities to train and benchmark automated evaluation systems using standardized datasets and human-generated labels. Indeed, significant progress has been made in this direction, with supervised models like NISQA (Non-Intrusive Speech Quality Assessment), @Mittag_2020 and MOSNet @Lo_2019, and demonstrating the feasibility of automating MOS predictions to a certain extent.

However, it is important to note that while these automated systems show promise, they are not without limitations. Factors such as cultural and linguistic diversity, individual preferences, and the evolving nature of TTS technologies continue to pose challenges for creating universally applicable evaluation metrics. As such, the development of robust, adaptable, and comprehensive evaluation frameworks remains an active area of research in the field of speech synthesis and quality assessment.
