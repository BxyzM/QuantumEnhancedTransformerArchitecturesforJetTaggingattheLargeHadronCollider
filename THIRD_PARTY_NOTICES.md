# Third-party notices

Parts of `thesis_models/` are copied or adapted from the projects below. All of them are MIT-licensed, and the MIT license requires its copyright and permission notice to be kept with copies or substantial portions of the code. The full license texts are in `LICENSES/`.

| File | Derived from | License |
|---|---|---|
| `thesis_models/part.py` | [weaver-core](https://github.com/hqucms/weaver-core), `weaver/nn/model/ParticleTransformer.py` (commit 3231e58): `Embed`, `PairEmbed`, `Block`, `SequenceTrimmer`, `ParticleTransformer` | MIT, Copyright (c) 2020 Huilin Qu, see `LICENSES/weaver-core.txt` |
| `thesis_models/part.py` | [particle_transformer](https://github.com/jet-universe/particle_transformer), `networks/example_ParticleTransformer.py` (model wrapper and defaults) | MIT, Copyright (c) 2022 Huilin Qu, Congqiao Li, Sitian Qian, see `LICENSES/particle_transformer.txt` |
| `thesis_models/vqc.py` | 1P1Q variational circuit code (encoding, CNOT ring, trainable rotations, readouts) | MIT, Copyright (c) 2024 Aritra, see `LICENSES/1P1Q.txt` |

`thesis_models/qdimenet.py` only imports [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) (MIT) as an installed dependency and copies none of its code. The other dependencies (NumPy, scikit-learn, PyTorch, PennyLane, Jupyter) are installed from `requirements.txt` and are not redistributed here.

## Data

The demo inputs in `data/` are small subsets of public datasets. Cite them when you use the data:

- **JetClass:** H. Qu, C. Li, S. Qian, *JetClass: A Large-Scale Dataset for Deep Learning in Jet Physics*, Zenodo (2022), [doi:10.5281/zenodo.6619768](https://doi.org/10.5281/zenodo.6619768). License: Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). The demo files contain a selection of jets plus QFIMs computed from them.
- **QM9:** R. Ramakrishnan, P. O. Dral, M. Rupp, O. A. von Lilienfeld, *Quantum chemistry structures and properties of 134 kilo molecules*, Scientific Data 1, 140022 (2014), [doi:10.1038/sdata.2014.22](https://doi.org/10.1038/sdata.2014.22); data on figshare, [doi:10.6084/m9.figshare.978904](https://doi.org/10.6084/m9.figshare.978904).
- **JetsGame:** simulated with Pythia 8.223 at truth level.
