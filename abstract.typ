#align(
  center + horizon,
  [
    = Abstract

    #v(3em)

    The evaluation of synthetic speech presents unique challenges in today's rapidly evolving technological landscape. While the Mean Opinion Score (MOS) remains the predominant metric for assessing speech quality, its inherent subjectivity raises concerns about reliability. This study examines the effectiveness of supervised neural models—specifically NISQA and MOSNet—in predicting MOS scores for speech naturalness.

    To facilitate this investigation, we compiled an extensive dataset from the Blizzard Challenge and Voice Conversion Challenge, incorporating MOS labels and commercial use permissions. These datasets are now publicly accessible via the HuggingFace Hub. Our experimental framework focused on reproducing and analyzing NISQA's performance, incorporating both pre-training and transfer learning approaches.

    Our results indicate that supervised MOS predictors face significant challenges in out-of-domain evaluations, primarily due to training data bias. Even with expanded datasets and pre-training strategies, the models demonstrated limited generalization capabilities. These findings suggest the need for fundamental changes in how we evaluate synthetic speech.

    We explore promising alternatives to traditional MOS evaluation, including SpeechLMScore—an unsupervised metric leveraging speech language models—and UTMOS, which employs ensemble learning techniques. These approaches show potential in addressing MOS limitations and providing more reliable quality assessments. Our research highlights the critical need for evaluation frameworks that can adapt to advancing speech synthesis technologies while maintaining objectivity and reliability.

  ],
)