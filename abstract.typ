#align(center + horizon,
[
  = Abstract

  #v(3em)

   Pruning techniques at initialization have gained prominence for their ability
   to reduce the size of neural networks while preserving performance. However, a
   puzzling phenomenon observed in such techniques is their insensitivity to layerwise
   shuffling, as highlighted in the seminal work of Frankle et al. (2021). Particularly,
   the performance of SynFlow-pruned networks at extreme sparsities improved
   after shuffling. Our investigation not only reaffirms the presence of neuron
   collapse, originally proposed by Frankle et al. (2021) as the cause of this improved
   performance, but also uncovers its occurrence in SNIP and GraSP. Additionally,
   we introduce novel scoring adjustments, such as SynFlow-Square, SynFlow-Cube,
   and SynFlow-Exp, which significantly increase the percentage of effective neurons.
   While these adjustments lead to sensitivity to layerwise shuffling at 99% sparsity
   in certain networks, the same effect does not manifest at 90% sparsity, which
   shows that neuron collapse does not fully explain the improved performance post
  shuffling. To further investigate this puzzle, we conduct a case study involving
   magnitude-based pruning. Contrary to expectations, shuffling magnitude-pruned
   networks results in increased accuracy, despite these networks having more
   effective paths and neurons before shuffling. These findings underscore the
   complexity of the relationship between neuron collapse, the number of effective
   paths and neurons, and sensitivity to layerwise shuffling

])