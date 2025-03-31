# Code of InfoReg
This is the official PyTorch implementation of "Adaptive Unimodal Regulation for Balanced Multimodal Information Acquisition".

## Authors:
Chengxiang Huang, [Yake Wei](https://echo0409.github.io/), Zequn Yang and [Di Hu](https://dtaoo.github.io/index.html)

## Abstract:
Sensory training during the early ages is vital for human development. Inspired by this cognitive phenomenon, we observe that the early training stage is also important for the multimodal learning process, where dataset information is rapidly acquired. We refer to this stage as the prime learning window. However, based on our observation, this prime learning window in multimodal learning is often dominated by information-sufficient modalities, which in turn suppresses the information acquisition of information-insufficient modalities.
To address this issue, we propose **Info**rmation Acquisition **Reg**ulation (InfoReg), a method designed to balance information acquisition among modalities. Specifically, InfoReg slows down the information acquisition process of information-sufficient modalities during the prime learning window, which could promote information acquisition of information-insufficient modalities. This regulation enables a more balanced learning process and improves the overall performance of the multimodal network. Experiments show that InfoReg outperforms related multimodal imbalanced methods across various datasets, achieving superior model performance.

For more details of our paper, please refer to  our [CVPR 2025 paper](https://arxiv.org/abs/2503.18595)
