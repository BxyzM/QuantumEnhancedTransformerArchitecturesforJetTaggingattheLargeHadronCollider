# Thesis notebooks: inference with the trained models

Companion notebooks for the master's thesis *Quantum-Enhanced Transformer Architectures for Jet Tagging at the Large Hadron Collider*.

Each of the three self-contained notebooks loads a small demo set, rebuilds the models, loads their trained weights and runs inference. Together they reproduce the numbers of the thesis on demo-sized data. The model code is in `thesis_models/`, so you only need the packages in `requirements.txt`. You don't need the original repositories or the full datasets.

| Notebook | Thesis part | Models | Demo data |
|---|---|---|---|
| `01_particle_transformer_quiver.ipynb` | QFIM-augmented Particle Transformer (QUIVER) on JetClass | ParT and QUIVER (PConcat) for Top and H→cc̄ vs QCD, with kinematic and full features, at 100K / 500K / 5M samples per epoch: 24 models, one seed each | 10,000 signal + 10,000 QCD test jets per task, with their QFIMs |
| `02_qm9_qdimenet.ipynb` | QM9 HOMO–LUMO gap | DimeNet++ and QDimeNet++ (run006, seed 42) | 5,000 QM9 test molecules (heavy atoms), with their QFIMs |
| `03_variational_circuits.ipynb` | Variational quantum circuits | Chapter 4: the 1P1Q circuits for Top (epoch 2) and H→cc̄ (epoch 9). Chapter 5: the grid of re-uploading depths L = 1–16, readouts (⟨Z₀⟩, all ⟨Z_i⟩, + ⟨Z_iZ_j⟩), the 1P1Q (CNOT ring) and R_YY(λ_ij ΔR_ij) circuits, BCE and MSE, on JetClass and JetsGame, one seed per cell (300 circuits) | 10,000 + 10,000 test jets per dataset and task |

Every notebook has the same four steps:
1. Setup.
2. Load the data.
3. Build the models and load the weights.
4. Run inference and print a table of the results next to the thesis values.

In Chapter 5 every cell exists at 7 qubits. Three readout and circuit combinations also have 10-qubit runs, all trained with BCE:
- R_YY with ⟨Z_i⟩ + ⟨Z_iZ_j⟩ (the qubit comparison of the thesis);
- 1P1Q with ⟨Z₀⟩;
- 1P1Q with all ⟨Z_i⟩.

## Installation

You need Python 3.10–3.12, because the pinned `numpy==1.26.4` has no wheels for newer versions. The notebooks were validated with Python 3.10. On Linux and Apple silicon:

```bash
git clone https://github.com/BxyzM/QuantumEnhancedTransformerArchitecturesforJetTaggingattheLargeHadronCollider.git
cd QuantumEnhancedTransformerArchitecturesforJetTaggingattheLargeHadronCollider
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

- **Apple silicon:** use a native arm64 Python. `python -c "import platform; print(platform.machine())"` must print `arm64`. An x86_64 Python running under Rosetta (for example from an Intel Homebrew) only offers PyTorch up to 2.2.2.
- **Intel Mac:** PyTorch is only available up to 2.2.2. Change `torch==2.9.1` to `torch==2.2.2` in `requirements.txt` before installing. The notebooks were also checked with this version in a clean environment and give the same results.

## Running

Start Jupyter from the environment, so that the notebooks run with its packages and not with another Jupyter installation on your system:

```bash
python -m jupyter notebook        # open a notebook and run all cells
```

To run all notebooks without the browser and keep the executed copies in `executed/`:

```bash
for nb in 01_particle_transformer_quiver 02_qm9_qdimenet 03_variational_circuits; do
  python -m jupyter nbconvert --to notebook --execute "$nb.ipynb" \
      --output-dir executed --ExecutePreprocessor.timeout=-1
done
```

All notebooks run on a CPU. Runtimes on 4 cores:

| Notebook | Runtime |
|---|---|
| 01 | ~6 min for all 24 transformers |
| 02 | ~1 min |
| 03 | ~4 min for the default circuit selection |

You can add any other circuit in `weights/vqc/index.json` to the run list in notebook 03. A 10-qubit, 16-layer circuit takes about 1 min per 1,000 jets.

## Expected results

The demo sets are small, so AUCs and MAEs scatter around the full-test-set values of the thesis. Every model shifts the same way on the same jets, so comparisons between models still hold.

| Notebook | Demo vs full test set | Notebook vs original pipeline, same inputs |
|---|---|---|
| 01 | Demo AUC within 0.002 of the thesis AUC of the same seed (mean difference −0.0007). | Scores bitwise identical to weaver's own inference for all 24 checkpoints. |
| 02 | Demo MAE 73.0 / 68.9 meV (DimeNet++ / QDimeNet++) vs 71.45 / 67.99 meV on the full test set (seed 42). | Predictions match the thesis predictions to 0.007 meV. |
| 03, Ch. 5 | Demo AUCs agree with the stored full-test AUCs within the scatter of a small sample. | 1P1Q: scores identical to 10⁻¹⁴. R_YY: identical to ≤ 2·10⁻⁵, because training evaluated ΔR_ij in single precision on a GPU. |
| 03, Ch. 4 | — | The notebook computes exact expectation values, while the thesis used 10⁴ shots. The difference from the thesis scores is the expected shot noise (RMS 0.005). A 10⁴-shot run is included. |

The verification results are in `verification/` and `weights/vqc/verification.json`.

## Layout

```
01_…, 02_…, 03_….ipynb   the notebooks
thesis_models/           part.py (ParT, QUIVER) · qdimenet.py (DimeNet++, QDimeNet++) · vqc.py (Ch. 4 and Ch. 5 circuits)
data/                    demo inputs (.npz), preprocessed exactly as the models saw them in training
weights/                 part/ (24 state dicts) · qm9/ (2 checkpoints) · vqc/ (all circuit weights + index.json)
verification/            comparison of the notebook outputs with the original pipelines
LICENSES/                license texts of the third-party code (see THIRD_PARTY_NOTICES.md)
```

## Notes on the data and weights

- **QFIMs:** the stored matrices are the Fubini–Study metric g from PennyLane's `qml.metric_tensor` (`hybrid=True`, `approx=None`), so the QFIM is F = 4g. The models receive g unchanged.
  - **JetClass:** the 30×30 matrix of a jet is stored as 90 arrays `qf0`…`qf89` of length 10. They are a row-major reshape: F[r, s] = qf_c[p] with 30r + s = 10c + p. QUIVER token p is (qf₀[p], …, qf₈₉[p]).
  - **QM9:** the per-atom-pair blocks are 6×6 and come from the 60×60 rotation block of the stored 101×101 metric.
- **The QUIVER QFIMs come from separate productions.** The Top and H→cc̄ datasets were produced with different trained 1P1Q circuits, so the same QCD jet carries different QFIMs in the two demo files. The Top QFIMs were not computed with the Chapter 4 Top model.
- **Which weights are used:**
  - Chapter 5 circuits: the weights after the last epoch, as in the thesis.
  - Chapter 4 circuits: the epoch with the best validation AUC, as in the thesis.
  - ParT and QUIVER: weaver's best-epoch checkpoint.

## Licenses and data sources

Parts of `thesis_models/` are derived from MIT-licensed code:
- [weaver-core](https://github.com/hqucms/weaver-core) and [particle_transformer](https://github.com/jet-universe/particle_transformer) for `part.py`;
- the 1P1Q circuit code for `vqc.py`.

Their license texts are in `LICENSES/`. `THIRD_PARTY_NOTICES.md` lists which file derives from which project.

The demo data are small subsets of JetClass ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), [doi:10.5281/zenodo.6619768](https://doi.org/10.5281/zenodo.6619768)), QM9 ([doi:10.1038/sdata.2014.22](https://doi.org/10.1038/sdata.2014.22)) and JetsGame. Please cite the original datasets when you use them. The references are in `THIRD_PARTY_NOTICES.md`.
