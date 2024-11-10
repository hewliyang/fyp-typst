= Appendix

== Revisions

1. Elaboration on the initial purpose / motivation / use-case of NISQA *(@section-nisqa)*

  - Appended additional context on the usability of NISQA naturalness MOS scores in combination with other objective or subjective metrics, despite the low out-of-distribution performance at stimuli levels

  - Enriched section explaining the emergence of NISQA usage in synthetic speech context, and it's original use in traditional telecommunications.

2. Clarify use of `listener_id` and `domain_id` in UTMOS system *(@section-utmos)*

  - Appended missing context regarding jointly encoding listener and system information during training in order to improve OOD performance relative to systems such as NISQA and MOSNet

3. Clarify difference in training methodology between original NISQA experiment and our own ablation experiments. *(@section-training)*

  - Augmented section with directed charts illustrating the different training processes, particularly the difference in pre-training dataset & time-dependency architecture @figure-training