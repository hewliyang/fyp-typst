#set heading(numbering: "1.")

= Conclusion

To recap, in this work, we have achieved the following:

1. Establish the basics for understanding speech processing and TTS
2. Curated a large dataset for training NISQA & MOSNet style of supervised models
3. Published these datasets for use by the public on the HuggingFace Hub
4. Reproduced the results of NISQA including pre-training and transfer learning
5. Analysed the potential pitfalls of MOS predictors in out-of-distribution scenarios
6. Explored alternative metrics and the use of self-supervised features in existing architectures.

Our work has demonstrated the inherent limitations of relying on Mean Opinion Score (MOS) as the sole metric for evaluating synthetic speech quality, particularly in light of the rapid advancements in modern speech synthesis technologies. While MOS has served as a convenient benchmark for many years, its relative nature and susceptibility to various biases, including the absence of well-defined anchors @chiang2023reportdetailssubjectiveevaluation, render it an unreliable measure of absolute quality. Our experiments highlight the significant influence of lower-quality systems acting as anchors, the impact of introducing higher-quality systems on historical ratings, and the potential saturation of the MOS scale in the face of increasingly natural-sounding synthetic speech.

These findings underscore the urgent need for a paradigm shift in speech synthesis evaluation methodologies. Relying solely on MOS, especially in the context of training automatic MOS predictors, risks perpetuating the problem of reliably predicting an unreliable score. Future work should focus on developing more robust and nuanced evaluation protocols that address the limitations of ACR. This includes exploring the use of standardized anchors in ACR, investigating alternative protocols like MUSHRA with its hybrid rating/ranking approach @le_maguer_limits_2024, and incorporating public leaderboards testing such as the TTS Arena. By embracing a more holistic approach, we can better characterize the true advancements in speech synthesis technology and move beyond the limitations of MOS.

The codebases used through this work can be accessed at:

- Training & evaluation codebase - #link("https://github.com/hewliyang/nisqa")
- Typesetting (Typst) and diagrams - #link("https://github.com/hewliyang/fyp-typst")
- Blizzard Challenge datasets from 2008 to 2023 - #linebreak()#link("https://huggingface.co/datasets/hewliyang/nisqa-blizzard-challenge-mos")
- VCC datasets - #link("https://huggingface.co/datasets/hewliyang/nisqa-vcc-mos")