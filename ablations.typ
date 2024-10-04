#set heading(numbering: "1.")
#import "@preview/showybox:2.0.1": showybox

= Ablations

In this section, we describe retraining the NISQA architecture from scratch on naturalness MOS with the newer self-attention layers described in @Mittag_2021 instead of the BiLSTM @Mittag_2020 architecture to investigate it's behavior and sensitivity to different pre-training data. We would also like to observe the impact of removing @prahallad2014blizzard and @prahallad2015blizzard from the train set.

All experiments were conducted on an AWS EC2 `g4dn.xlarge` instance with 4 vCPUs, 16 GB of memory and a single Nvidia Tesla T4 with 16 GB of VRAM.

== Training

Since, the original pre-training corpus was not released in @Mittag_2020, we utilize a similar synthetic corpus described in @Mittag_2021 as the *NISQA Corpus*.

Overall, it includes over 14,000 stimuli with a total of over 97,000 human ratings for MOS-N. But we only use a subset shown below as the other datasets mainly focused on VoIP distortions, which may be out of distribution for our use case.

#figure(
  caption: "Subset of NISQA Corpus used in pre-training",
  kind: table,
  [
    #set text(size: 10.5pt)
    #table(
      columns: 6,
      [Dataset], [Language], [Files], [Individual Speakers], [Files per Condition], [Votes per File],
      [NISQA_TRAIN_SIM], [`en`], [10,000], [2,322], [1], [~5],
      [NISQA_VAL_SIM], [`en`], [2,500], [938], [1], [~5],
      [NISQA_TRAIN_LIVE], [`en`], [1,020], [486], [1], [~5],
      [NISQA_VAL_LIVE], [`en`], [200], [102], [1], [~5],
    )],
)

The hyperparameters for mel-spectrogram construction are the same defaults used in NISQA. In particular:

#showybox(
  frame: (
    border-color: red.darken(50%),
    title-color: red.lighten(60%),
    body-color: red.lighten(80%),
  ),
  title-style: (
    color: black,
    weight: "regular",
    align: center,
  ),
  shadow: (
    offset: 3pt,
  ),
  title: "Mel Spectrogram Hyperparameters",
  ```
  ms_sr: null
  ms_fmax: 20000
  ms_n_fft: 4096
  ms_hop_length: 0.01
  ms_win_length: 0.02
  ms_n_mels: 48
  ms_seg_length: 15
  ms_seg_hop_length: 4
  ms_max_segments: 1300 duration. increase if you apply the model to longer samples
  ms_channel: null
  ```,
)

During pretraining, the `NISQA_VAL_SIM` and `NISQA_VAL_LIVE` are used as validations set to determine early stopping. The BiLSTM decoder network is switched out for the newer self-attention network.

Apart from reducing batch size to 32 due to VRAM limitations, yet again the default hyperparameters are used.

#showybox(
  frame: (
    border-color: red.darken(50%),
    title-color: red.lighten(60%),
    body-color: red.lighten(80%),
  ),
  title-style: (
    color: black,
    weight: "regular",
    align: center,
  ),
  shadow: (
    offset: 3pt,
  ),
  title: "Pre-training Hyperparameters",
  ```
  tr_epochs: 50 # number of max training epochs
  tr_early_stop: 5 # if val RMSE nor correlation does not improve
  tr_bs: 32
  tr_bs_val: 32
  tr_lr: 0.001 # learning rate of ADAM optimiser
  tr_lr_patience: 15 # decrease lr if val RMSE or correlation does not improve
  ```,
)

The model is then fine tuned on the Blizzard Challenge and VCC datasets described in @sec-dataset-curation using the same hyperparameters until convergence. We try to approximate the NISQA training and validation sets as close as possible, using the same train-validation splits.

The loss curves are described in the following chart:

#figure(
  caption: [Train & validation loss curves],
  kind: image,
  [#image("assets/training-run.svg")],
)

== Evaluation & Analysis

The results on the training and validation set at the per-stimuli level are as follows:

The results slightly underperform the original NISQA findings using the CNN-BiLSTM architecture, perhaps due to difference in pretraining data and the absence of the 2014 @prahallad2014blizzard and 2015 @itu_bs1534-3_2015 Blizzard challenges in the train sets.

However, it seems that the phenomenon of poor performance on out-of-distribution data at the stimuli level remains although the in-distribution PRCC is relatively higher - indicative of some overfitting.

#figure(
  caption: "Training and Validation Sets Performance Metrics",
  kind: table,
  [
    #table(
      columns: 3,
      [Dataset], [$r$], [$"RMSE"$],
      table.cell([*Train*]), table.cell(stroke: (left: 0pt), []), table.cell(stroke: (left: 0pt), []),
      [Blizzard_2008_TRAIN], [0.84], [0.48],
      [Blizzard_2009_TRAIN], [0.84], [0.49],
      [Blizzard_2010_TRAIN], [0.95], [0.44],
      [Blizzard_2011_TRAIN], [0.87], [0.43],
      [Blizzard_2013_TRAIN], [0.88], [0.46],
      [Blizzard_2019_TRAIN], [0.91], [0.39],
      [Blizzard_2020_TRAIN], [0.73], [0.51],
      [Blizzard_2021_TRAIN], [0.89], [0.41],
      [VCC_2018_HUB_TRAIN], [0.74], [0.56],
      [VCC_2018_SPO_TRAIN], [0.74], [0.59],
      [*Validation*], table.cell(stroke: (left: 0pt), []), table.cell(stroke: (left: 0pt), []),
      [Blizzard_2008_VAL], [0.70], [0.60],
      [Blizzard_2009_VAL], [0.46], [0.62],
      [Blizzard_2010_VAL], [0.68], [0.67],
      [Blizzard_2012_VAL], [0.50], [0.85],
      [Blizzard_2016_VAL], [0.84], [0.51],
      [VCC_2018_HUB_VAL], [0.64], [0.66],
      [VCC_2018_SPO_VAL], [0.66], [0.65],
    )],
)

As for testing, regretfully, the VCC 2016 @toda2016voice did not distribute stimuli level metrics. Instead, we test on the PhySyQx @7336888 dataset instead. It is important to recall however that it only contains 36 stimuli.

On the PhySyQx dataset, our model achieves an $r = 0.79$ at the per-stimuli level, once again slightly underperforming the original implementation.

Overall, we validate that the work of @Mittag_2020 is indeed reproducible, but are concerned with the poor out of domain performance at the stimuli level, particularly in the validation sets.