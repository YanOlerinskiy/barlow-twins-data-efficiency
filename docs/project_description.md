# Reducing data in visual AI
## Prerequisites
Experience with a Deep Learning framework, such as PyTorch is needed; (otherwise it will take too much time to also learn PyTorch in addition to doing the research)

## Background and motivation
Data is powering AI. Most data, however, is in the hands of LargeCompany^TM, creating a privileged few that have huge data, and a long tail of universities, SMEs and researchers, that have limited access. Moreover, large dataset are hard to process and curate, making it difficult to control for fairness, copyright, and data biases.

In this project we will explore how to more efficiently use data to train visual AI. Specifically, we will investigate the effect of the amount of data on current visual foundation models [1, 2, 3] typically a vision transformer (VIT) [4, 12] trained self-supervised [5, 6, 7, 14] which is fine-tuned by PEFT [13] on down-stream tasks, eg [11].
In this project we will evaluate how to scale-down the huge data problem, to a manageable, but still representative, setting where we can tackle the underlying research problem, without being constrained by huge compute, see the "scaling down deep learning" [9] paper for a motivation. A possible starting architecture is [8].

## Research Questions for the Sub-Projects
Whole group. The research question for the whole group is how to evaluate data efficiency by learning curves [10] for small-compute self-supervised pre-trained visual foundation models, which are then evaluated on a set of down-stream tasks.

For each of the 5 sub-projects. The research question for each individual student, is to investigate, implement, and critically evaluate a popular self-supervision method on data-efficiency. Each student has 1 method, and this method is compared, and evaluated, to the other 4 methods.

## References
[1] On the Opportunities and Risks of Foundation Models https://arxiv.org/abs/2108.07258

[2] https://en.wikipedia.org/wiki/Foundation_model

[3] https://crfm.stanford.edu/2021/10/18/reflections.html

[4] An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale https://arxiv.org/abs/2010.11929

[5] Masked Autoencoders Are Scalable Vision Learners https://arxiv.org/abs/2111.06377

[6] Bootstrap your own latent: A new approach to self-supervised Learning https://arxiv.org/abs/2006.07733

[7] A Simple Framework for Contrastive Learning of Visual Representations https://arxiv.org/abs/2002.05709

[8] Escaping the Big Data Paradigm with Compact Transformers https://arxiv.org/abs/2104.05704

[9] Scaling down Deep Learning https://greydanus.github.io/2020/12/01/scaling-down/ and https://github.com/greydanus/mnist1d

[10] https://en.wikipedia.org/wiki/Learning_curve_(machine_learning)

[11] Visual Task Adaptation Benchmark (VTAB) https://github.com/google-research/task_adaptation

[12] Data-Efficient architectures and training for Image classification https://github.com/facebookresearch/deit

[13] Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey https://arxiv.org/abs/2403.14608

[14] A Survey of the Self Supervised Learning Mechanisms for Vision Transformers https://arxiv.org/abs/2408.17059