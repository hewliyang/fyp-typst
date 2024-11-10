#set heading(numbering: "1.")

= Related Work (Part I)

The main goal of this work is to investigate the feasibility and potential pitfalls of utilizing neural networks as predictors for subjective scores, primarily MOS-N or MOS for naturalness of synthetic speech.

Given a dataset $D = {(x_i,y_i)}_(i=1)^N$ where $x_i$ represents the input speech signal and $y_i in {1,2,3,4,5}$ is corresponding MOS for speech naturalness, we aim to train a neural network $f_theta (x_i)$ such that the predicted score $hat(y)_i = f_theta (x_i)$ minimises the loss

$
  cal(L)(theta) = 1 / N sum_(i=1)^N cal(L)(hat(y)_i, y_i)
$

where $cal(L)$ is the loss function which in this case can be a choice of cross entropy or MSE loss.

== Evaluation Preamble

In TTS evaluations, metrics such as the MOS can be calculated at different granularities. For our investigations, we define them as at the stimulus or system level.

=== Stimulus Level

In this apporach, the MOS is calculated for each stimulus (i.e. each audio sample) seperately. If we have multiple listeners evaluating a single stimulus, we would take the mean score for that stimulus.

Let:

- $x_(i,j)$ be the score given by the $j$-th rather for the $i$-th stimulus
- $m_i$ be the MOS for the $i$-th stimulus
- $n_i$ be the number of listeners for the $i$-th stimulus.

Then the MOS for stimulus $i$ is defined as:

$
  m_i = 1 / n_i sum_(j=1)^n_i x_(i,j)
$

=== System Level

At the system level, the MOS is calculated across all stimuli for a given TTS system, i.e., it aggregates scores across multiple stimuli and listeners to get an overall system score.

Let:

- $N$ be the total number of stimuli
- $m_i$ be the mean score for the $i$-th stimulus from above
- $n_"total" = sum_(i=1)^N n_i$ be the total number of scores across all stimuli and listeners

Then, the system-level MOS is:

$
  M_"system" = 1 / n_"total" sum_(i=1)^N sum_(j=1)^n_i x_(i,j)
$


== MOSNet

MOSNet @Lo_2019 is a deep learning based model designed for assessment of voice conversion (VC) systems, particularly on predicting the MOS of synthesized speech. The model primarily leverages three neural network architectures for feature extraction: CNN's, Bidirectional Long Short Term Memory (BLSTM) and a combination of CNN-BLSTM.

MOSNet follows our formulation above. Interestingly, MOSNet takes a linear spectrogram (direct output of STFT) as input and outputs a floating point number in $[1,5]$, and is trained on a linear combination of regular MSE loss and a frame-level MSE regression objective, i.e.

$
  L(theta) = 1 / N sum_(i=1)^N {(hat(y)_i - y_i)^2 + alpha / T_n sum_(i=1)^(T_n) (hat(q)_(n,i) - y_i)^2}
$

where:

- $hat(q)_(i,n)$ denotes the frame level prediction at time $i$
- $T_n$ denotes the total number of frames in the $n$-th sample
- $N$ is the number of training samples
- $alpha$ is a hyperparameter controlling the impact of the frame-level MSE on the overall loss

In particular,

- The CNN employs 12 convolutional layers to capture temporal information within a 25-frame segment, which is about a 400ms window of the input spectrogram
- The BSLTM, which utilizes a configuration identical to Quality-Net (cite) captures long-term dependencies and sequential characteristics in speech through its bidirectional nature.

The authors conducted ablation studies to investigate performance switching the frame-level MSE objective with a regular MSE objective and found that it significantly improved prediction accuracy from a PRCC at the utterance level of 0.560 against the ground truth to 0.642.

MOSNet was trained and evaluated using a dataset from the Voice Conversion Challenge (VCC) 2018. The dataset comprised of MOS-N assessments for 20,580 utterances.

The train-test-split was:

- 13,580 for training
- 3,000 for evaluation
- 4,000 for testing

All samples were downsampled to 16 kHz before STFT with:
- `sr` = 16,000
- `n_fft` = 512
- `hop_length` = 256

Training was done with the following hyperparameters:

- Learning rate of $1e-4$
- Dropout rate of 0.3
- Early stopping with 5 epochs patience on vanilla MSE of validation set

Evaluations show that MOSNet performs remarkably at the system-level with a PRCC of 0.957 on the held-out test set at the system-level, but poorly on the stimuli-level with only 0.642 against ground truths. The authors attribute this discrepancy to variance in listener perception for each stimuli.

The authors claim that the model generalizes to unseen data by providing evidence that testing on the VCC 2016 dataset, a PRCC of 0.917 was obtained at the system-level.

== NISQA <section-nisqa>

NISQA for assessing synthetic speech MOS the work of @Mittag_2020, and is not be confused with their later work under the same name @Mittag_2021. Like MOSNet, NISQA also utilizes a CNN-BiLSTM framework but later in @Mittag_2021 introduces a self-attention architecture, replacing the BiLSTM module to better capture temporal interactions.

NISQA was initially used for predicting MOS scores in the telecommunications field, particularly to evaluate and detect degradations in signal quality post-transmission. Due to architectural similarities with MOSNet, the authors also found it to be suitable for neural speech applications.

In contrast to MOSNet, NISQA utilises mel spectrograms instead of linear spectrograms. They are divided into 150ms segments with 10ms hop length which are fed into the CNN encoder. The encoder output for each segment is a 20-dimensional feature vector which is then used as input for a bidirectional LSTM or self-attention network. Each framewise prediction is then average-pooled at the final output layer.

Assuming a $f_s$ of 16 kHz, the mel spectrograms are firstly created using the following parameters:

- `n_mels` = 48
- `n_fft` = 4048
- `win_length` = 320
- `hop_length` = 160

NISQA introduces a far larger training corpus compared to MOSNet, incorporating data from the 2008 to 2019 Blizzard Challenge @king2008blizzard @king2009blizzard @king2010blizzard @king2011blizzard @king2012blizzard @king2013blizzard @prahallad2014blizzard @prahallad2015blizzard @king2016blizzard @wu2019blizzard, in addition to VCC 2016 @toda2016voice and 2018 @lorenzotrueba2018voiceconversionchallenge2018. The additional datasets that were used were PhySyQX @7336888 for testing, and three in house TU Berlin / Kiel University German datasets for both training and testing. The datasets combined cover 12 different languages, primarily English, German, Chinese and dialects from India.

Interestingly, NISQA also introduced the usage of an pretraining dataset - which is unreleased. This set consisted of 100,000 stimuli based on 5,000 English and German speech files on which distortions were augmented into in order to simulate artifacts in synthesized speech. As these stimuli did not have labels, the authors utilized the objective metric, Perceptual Objective Listening Quality Analysis (POLQA), which is an evolution of PESQ as a proxy for MOS.

The model was firstly trained on the pre-training dataset for 24 epochs, after which full parameter fine tuning or transfer learning was done on the actual training set. The train-test split was not done randomly in this case. The entire training process utilized the Adam optimizer at a learning rate of $1e-3$ with a regular MSE objective.

- Train set includes Blizzard Challenges 2008 to 2019 ad VCC 2018 Training sets
- Validation set included subtasks from Blizzard Challenges 2008, 2009, 2010, 2012 and 2016 that were unused in the train set, and VCC 2018 Validation
- Test set included VCC 2016, PhySyQX and two in-house German sets

In particular, the authors wanted to use VCC 2016 as a direct comparison to MOSNet, which similarly used it as an out-of-distribution test.

In terms of results, NISQA demonstrates similar behavior to MOSNet with spectacular system-level PRCC's (0.89) but still suffers from moderate correlation to ground truths at the stimuli-level (0.65) across the test set.

On VCC 2016 however, NISQA with transfer learning outperforms MOSNet with a system-level correlation of 0.96 compared to 0.92. The authors conduted ablations and found that the model without pre-training achieves a similar correlation of 0.93.

Note that the results claimed were computed based on the earlier CNN-BiLSTM architecture.

While the relatively poorer stimuli level performance on unseen data puts into question the reliability of it's predictions, in practice it is still useful to use in conjunction with other objective or subjective metrics in a multivariate evaluation system.