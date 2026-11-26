# Thesis notebooks: inference with the trained models

Three self-contained notebooks that load a small demo set, rebuild a model, load its trained weights and run inference. They reproduce the numbers of the thesis on demo-sized data. The model code is in `thesis_models/`, so the notebooks need only the packages in `requirements.txt`. No original repository or full dataset is required.

| Notebook | Thesis part | Models | Demo data |
|---|---|---|---|
| `01_particle_transformer_quiver.ipynb` | QFIM-augmented Particle Transformer (QUIVER) on JetClass | ParT and QUIVER (PConcat) for Top and H→cc̄ vs QCD, kinematic and full features, 100K / 500K / 5M samples per epoch; 24 models, one seed each | 10,000 signal + 10,000 QCD test jets per task with their QFIMs |
| `02_qm9_qdimenet.ipynb` | QM9 HOMO–LUMO gap | DimeNet++ and QDimeNet++ (run006, seed 42) | 5,000 QM9 test molecules (heavy atoms) with their QFIMs |
| `03_variational_circuits.ipynb` | Variational quantum circuits | Chapter 4: 1P1Q circuits for Top (epoch 2) and H→cc̄ (epoch 9). Chapter 5: the full grid of re-uploading depths, readouts (⟨Z₀⟩, all ⟨Z_i⟩, + ⟨Z_iZ_j⟩) and the R_YY(λ_ij ΔR_ij) circuit, BCE and MSE, 7 and 10 qubits, JetClass and JetsGame, one seed per cell (300 circuits) | 10,000 + 10,000 test jets per dataset and task |

Every notebook has four steps: setup, load the data, build the models and load the weights, then run inference with a table of the results next to the thesis values.

## Running

```bash
pip install -r requirements.txt
jupyter notebook          # open a notebook and run all cells
# or, without Jupyter: run all code cells in order
python tools/run_notebook.py 02_qm9_qdimenet.ipynb
```

All notebooks run on a CPU. Runtimes on 4 cores:

| Notebook | Runtime |
|---|---|
| 01 | ~6 min for all 24 transformers |
| 02 | ~1 min |
| 03 | ~4 min for the default circuit selection |

Any other circuit in `weights/vqc/index.json` can be added to the run list in notebook 03. A 10-qubit, 16-layer circuit takes about 1 min per 1,000 jets.

## Demo data and expected agreement

The demo sets are small, so AUCs and MAEs scatter around the full-test thesis values. Every model shifts the same way on the same jets, so comparisons between models are preserved.

| Notebook | Demo vs full test set | Original pipeline vs notebook, same jets |
|---|---|---|
| 01 | demo AUC within 0.002 of the thesis AUC of the same seed | scores bitwise identical to weaver's own inference for all 24 checkpoints |
| 02 | demo MAE 73.0 / 68.9 meV vs 71.45 / 67.99 meV (seed 42) | predictions match the thesis predictions to 0.007 meV |
| 03, Ch. 5 | demo AUCs agree with the stored full-test AUCs to within the scatter of a small sample | 1P1Q: scores identical to 10⁻¹⁴. R_YY: identical to ≤ 2·10⁻⁵, because training evaluated ΔR_ij in single precision on a GPU |
| 03, Ch. 4 | — | the notebook computes exact expectation values; the thesis used 10⁴ shots. The difference to the thesis scores is the expected shot noise (RMS 0.005). A 10⁴-shot run is included |

The verification results are in `verification/` and `weights/vqc/verification.json`.

## Layout

```
thesis_models/   part.py (ParT, QUIVER) · qdimenet.py (DimeNet++, QDimeNet++) · vqc.py (Ch. 4 and Ch. 5 circuits)
data/            demo inputs (.npz), already preprocessed exactly as the models saw them in training
weights/         part/ (24 state dicts) · qm9/ (2 checkpoints) · vqc/ (all circuit weights + index.json)
tools/           export_*.py: provenance scripts that produced data/ and weights/ from the original runs
                 build_nb*.py: notebook generators · run_notebook.py: run a notebook without Jupyter
verification/    comparison of notebook outputs with the original pipelines
```

## Notes on the data

- **QFIMs:** the stored matrices are the Fubini–Study metric g from PennyLane's `qml.metric_tensor` (`hybrid=True`, `approx=None`), so the QFIM is F = 4g. The models receive g unchanged.
  - **JetClass:** the 30×30 matrix of a jet is stored as 90 arrays `qf0`…`qf89` of length 10, a row-major reshape F[r, s] = qf_c[p] with 30r + s = 10c + p. QUIVER token p is (qf₀[p], …, qf₈₉[p]).
  - **QM9:** the per-atom-pair blocks are 6×6, from the 60×60 rotation block of the stored 101×101 metric.
- **QUIVER QFIMs come from separate productions:** the Top and H→cc̄ datasets were produced with different trained 1P1Q circuits, so the same QCD jet carries different QFIMs in the two demo files. The Top QFIMs were not computed with the Chapter 4 Top model.
- **Weights:**
  - Chapter 5 circuits use the weights after the last epoch, as in the thesis.
  - Chapter 4 circuits use the epoch with the best validation AUC, as in the thesis.
  - ParT and QUIVER use weaver's best-epoch checkpoint.
