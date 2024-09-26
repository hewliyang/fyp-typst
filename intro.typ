#set heading(numbering: "1.")

#import "@preview/fletcher:0.5.1" as fletcher: diagram, node, edge
#import fletcher.shapes: diamond

= Introduction

== Terminology
#lorem(120)

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
      [Speaker Characteristics], [Voice qualities, speaking styles, accents],
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

== Implications on Evaluation Techniques

== Audio Signals
#lorem(120)

== Representations for Deep Learning
#lorem(120)

== Motivations and Background

The proliferation of TTS across diverse sectors such as

... list the sectors here


necessitates a systematic means for evaluating their performance. Different downstream use cases demand optimizing for different metrics such as


... list the different performance metrics here

Typical loss functions such as MSE and MAE do not measure the afformentioned nuances in speech, but merely how well it approximates the training data. This phenomenon gives rise to the need for manual assessments by human judgement. For example, the Mean Opinion Score (MOS) serves as a popular metric employed for this purpose.

Recent advancements in deep learning has accelerated research on data-driven approaches to predict these subjective metrics with remarkable accuracy. Despite this, serious concerns remain on the performance of these supervised neural predictors on out-of-domain data.

The significance of reliable evaluation systems can alleviate the reliance on manual human assessments to a certain extent, but also offer an objectives means for evaluating TTS systems at scale.

+ *Banana*
  - Apple
  - Pineapple

+ Testing 123

#grid(
  columns: (1.5fr, 2fr),
  column-gutter: 10pt,
  [#figure(
      caption: [An Example Table],
      kind: table,
      [
        #table(
          columns: 3,
          [Column A],
          [Column B],
          [Column C],
          table.hline(start: 0, stroke: 1pt),
          [Hi],
          [There],
          [Can],
          [You],
          [See],
          [This],
          [Table],
          [Or],
          [Not?],
        )
      ],
    )<tab-exampleTable>],
  [#figure(
      caption: [A Second Table],
      kind: table,
      [
        #table(
      columns: 3,
      [*Column A*],
      [Column B],
      [Column C],
      // table.hline(start: 0, stroke: 1pt),
      [#lorem(2)],
      [#lorem(2)],
      [#lorem(2)],
      [#lorem(2)],
      [#lorem(2)],
      [#lorem(2)],
      [#lorem(2)],
      [#lorem(2)],
      [#lorem(2)],
    )
      ],
    )<tab-secondTable>],
)

The results are shown in @tab-exampleTable.

#figure(
  caption: [A graph layout],
  kind: "diagram",
  supplement: [Figure],
  [#diagram(
      node-stroke: .1em,
      node-fill: gradient.radial(blue.lighten(80%), blue, center: (30%, 20%), radius: 80%),
      spacing: 4em,
      edge((-1, 0), "r", "-|>", `open(path)`, label-pos: 0, label-side: center),
      node((0, 0), `reading`, radius: 2em),
      edge(`read()`, "-|>"),
      node((1, 0), `eof`, radius: 2em),
      edge(`close()`, "-|>"),
      node((2, 0), `closed`, radius: 2em, extrude: (-2.5, 0)),
      edge((0, 0), (0, 0), `read()`, "--|>", bend: 130deg),
      edge((0, 0), (2, 0), `close()`, "-|>", bend: -40deg),
    )],
)<fig-banana>

The results are shown in @fig-banana.

#lorem(30)

Here is an in-line equation: $f(x)=sin(x)$ #lorem(10) $g(x)=cos(x)$. #lorem(20).

And here is an out of line equation

$
  f(x) = sin(x)
$<eq-sin>

$
  g(x) = e^(3x pi)
$<eq-exponent>

$
  h(x) = integral_0^1 tan(x) upright(d)x
$<eq-integral>

$
  delta = sqrt(x^2 + y^2)
$<eq-distance>

Refer to @eq-distance.

The first paper is @vaswani2023attentionneed