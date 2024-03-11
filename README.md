# SPDInv
SPDInv: Source Prompt Disentangled Inversion for Boosting Image Editability with  Diffusion Models

<a href='xxx'><img src='https://img.shields.io/badge/arXiv-xxx-b31b1b.svg'></a> &nbsp;&nbsp;

[Ruibin Li](https://github.com/leeruibin)<sup>1</sup> | [Ruihuang Li](https://scholar.google.com/citations?user=8CfyOtQAAAAJ&hl=zh-CN)<sup>1</sup> |[Song Guo](https://scholar.google.com/citations?user=Ib-sizwAAAAJ&hl=en)<sup>2</sup> | [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang/)<sup>1*</sup>

<sup>1</sup>The Hong Kong Polytechnic University, <sup>2</sup>The Hong Kong University of Science and Technology.

:star: If SeeSR is helpful to your images or projects, please help star this repo. Thanks! :hugs:

# 🔎 Overview framework

Pipelines of different inversion methods in text-driven editing. (a) DDIM inversion inverts a real image to a latent noise code, but the inverted noise code often results in large gap of reconstruction $D_{Rec}$ with higher CFG parameters. (b) NTI optimizes the null-text embedding to narrow the gap of reconstruction $D_{Rec}$, while NPI further optimizes the speed of NTI. (c) DirectInv records the differences between the inversion feature and the reconstruction feature, and merges them back to achieve high-quality reconstruction. (d) Our SPDInv aims to minimize the gap of noise $D_{Noi}$, instead of $D_{Rec}$, which can reduce the impact of source prompt on the editing process and thus reduce the artifacts and inconsistent details encountered by the previous methods.

![SPDInv](figures/methods.png)

# ⚙️ Dependencies and Installation
TODO

# 🚀 Quick Inference
TODO

# 📷 Editing cases with P2P, MasaCtrl, PNP, ELITE
## Editing cases with P2P
![P2P](figures/cases_P2P.jpg)
## Editing cases with MasaCtrl
![MasaCtrl](figures/cases_MasaCtrl.jpg)
## Editing cases with PNP
![PNP](figures/cases_PNP.jpg)
## Editing cases with ELITE
![ELITE](figures/cases_ELITE.jpg)

# TODO



