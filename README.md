# Laplacian Canonization: A Minimalist Approach to Sign and Basis Invariant Spectral Embedding

Source code for the paper "**[Laplacian Canonization: A Minimalist Approach to Sign and Basis Invariant Spectral Embedding](https://openreview.net/forum?id=1mAYtdoYw6)**", NeurIPS 2023.

The implementation of our MAP algorithm is in ```data/map.py```, along with codes for verifying correctness and counting the ratio of assumption violations.

To download the datasets, run ```data/script_download_ZINC.sh```. The OGBG-MOL* datasets are automatically downloaded from OGB.
To run the experiments, use the scripts in ```scripts/```.

**Attribution**: Our code is built on top of the [[SignNet repo](https://github.com/cptq/SignNet-BasisNet)] by Lim et al. in 2022, which in turn builds off of the setup in [[LSPE repo](https://github.com/vijaydwivedi75/gnn-lspe)] by Dwivedi et al. in 2021.

If you use our code, please cite
```
@inproceedings{laplacian-canonization,
    title={{Laplacian Canonization: A Minimalist Approach to Sign and Basis Invariant Spectral Embedding}},
    author={Ma, George and Wang, Yifei and Wang, Yisen},
    booktitle={NeurIPS},
    year={2023}
}
```
