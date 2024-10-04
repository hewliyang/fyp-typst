#set heading(numbering: "1.")

= Discussion & Related Work (Part II)

As we have seen in the previous sections, purely supervised MOS predictors struggle to perform in out-of-domain evaluations.
This is an active area of research, evidenced by the rise of new challenges such as VoiceMOS @10389763.

Simply gathering more labels is expensive, time-consuming and may not result in better OOD performance which we have shown by training on different densities of data through MOSNet @Lo_2019, NISQA @Mittag_2021 and our reproduction, all achieving nearly identical results. In particular, the system level PRCC of NISQA after pre-training and a far larger train set only slightly outperforms MOSNet which was only trained on VCC 2018.

We suggest that this phenomenon is due to training data bias

Therefore, researchers have come up with incorporating innovative techniques such as self-supervised features to augment such MOS predictors. On the other hand, there are also alternative non-MOS metrics that can be used as a proxy for human perception. These metrics such as SpeechLMScore @maiti2022speechlmscoreevaluatingspeechgeneration can be trained in an unsupervised manner and therefore are not as heavily dependent on labeled datasets.

== UTMOS

UTokyo-SaruLab MOS (UTMOS) @saeki2022utmosutokyosarulabvoicemoschallenge is based on ensemble learning of strong and weak learners. It was proposed in VoiceMOS 2022, and is used as a baseline system for VoiceMOS 2023.

The strong learners are fined-tuned models of self-supervised learning (SSL) models based on Wav2Vec @baevski2020wav2vec20frameworkselfsupervised. UTMOS introduces tricks such as contrastive learning, listener dependency and phoneme encoding to augment SSL outputs with relevant features.

The weak learners use basic machine-learning methods i.e. linear regression & random forest to predict scores based on these SSL features.

== SpeechLMScore

SpeechLMScore @maiti2022speechlmscoreevaluatingspeechgeneration is an unsupervised metric that evaluates generated speech using a speech language model. The metric computes the average log-probability of a speech signal. In language modelling, this is formally known as *perplexity*.

An alternative interpretation would be it measures how likely a sequence of audio tokens is.

For a speech signal $x$ of length $t$, it first converts $x$ into $d = {d_1, ..., d_t}$ using an encoder. Then using a pre-trained speech-unit language model on a large corpus of data, SpeechLMScore is computed as:

$
  "SpeechLMScore"(d|theta) = 1 / T sum_(i=1)^T log(p(d_i | d_(<i), theta))
$

where:

- $T$ is the total number of speech tokens generated by the encoder
- $d_i$ is the current token
- $d_(<i)$ represents all previous tokens
- $theta$ is the speech language model

Higher scores (logprobs) correlate with better perceived speech quality.

Notably, it does not rely on human annotations or task-specific optimization, making it more generalizable across different tasks and domain.