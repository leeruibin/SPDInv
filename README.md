# SPDInv
SPDInv: Source Prompt Disentangled Inversion for Boosting Image Editability with  Diffusion Models

<a href='https://arxiv.org/abs/2311.16518'><img src='https://img.shields.io/badge/arXiv-2311.16518-b31b1b.svg'></a> &nbsp;&nbsp;

[Ruibin Li](https://github.com/leeruibin)<sup>1</sup> | [Ruihuang Li](https://scholar.google.com/citations?user=8CfyOtQAAAAJ&hl=zh-CN)<sup>1</sup> |[Song Guo](https://scholar.google.com/citations?user=Ib-sizwAAAAJ&hl=en)<sup>2</sup> | [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang/)<sup>1*</sup>

<sup>1</sup>The Hong Kong Polytechnic University, <sup>2</sup>The Hong Kong University of Science and Technology.

:star: If SeeSR is helpful to your images or projects, please help star this repo. Thanks! :hugs:

## 🔎 Overview framework

Pipelines of different inversion methods in text-driven editing. (a) DDIM inversion inverts a real image to a latent noise code, but the inverted noise code often results in large gap of reconstruction $D_{Rec}$ with higher CFG parameters. (b) NTI optimizes the null-text embedding to narrow the gap of reconstruction $D_{Rec}$, while NPI further optimizes the speed of NTI. (c) DirectInv records the differences between the inversion feature and the reconstruction feature, and merges them back to achieve high-quality reconstruction. (d) Our SPDInv aims to minimize the gap of noise $D_{Noi}$, instead of $D_{Rec}$, which can reduce the impact of source prompt on the editing process and thus reduce the artifacts and inconsistent details encountered by the previous methods.

![SPDInv](figures/methods.png)

## 📷 Editing cases with P2P, MasaCtrl, PNP, ELITE

![P2P](figures/cases_P2P.jpg)

![MasaCtrl](figures/cases_MasaCtrl.jpg)

![PNP](figures/cases_PNP.jpg)

![ELITE](figures/cases_ELITE.jpg)

## TODO



