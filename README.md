# CLEAR: Closed-Loop, Erasure-Aware CSI Feedback

Code for the paper *CLEAR: Closed-Loop, Erasure-Aware CSI Feedback for FDD Massive MIMO over Multiple Blockage-Prone Uplinks*.

A UE feeds back downlink CSI with a fixed budget of K analog channel uses that it can split over L uplinks (one per TRP). Each uplink has its own SNR and may be blocked during the report. CLEAR has three parts:

- **Codec.** A state-conditioned bidirectional Mamba-2 encoder/decoder, trained jointly over a library of five candidate splits.
- **OFDR router.** It picks the split with the smallest predicted expected NMSE, computed exactly over all 2^L erasure patterns with a learned per-pattern critic. Both ends know the link state, so the choice needs no signaling.
- **Closed-loop decoder.** The CPU re-encodes its estimate with the UE encoder, forms the innovation on each link that arrived, and back-projects it through the aggregator Jacobian.

## Requirements

- Linux with an NVIDIA GPU. The Mamba-2 kernels of `mamba_ssm` need CUDA, so CPU execution is not supported.
- Python 3.10. We used PyTorch 2.9.1 with CUDA 12.8.

```bash
pip install -r requirements.txt   # install a CUDA build of torch first if needed
```

`matplotlib` is only needed for the figures. The shell scripts use the `python` on `PATH`. Set `CONDA_ENV=<name>` to have them activate a conda environment first.

## Data

We use the COST2100 dataset preprocessed by the CsiNet authors (C.-K. Wen, W.-T. Shih, and S. Jin, IEEE WCL 2018): 32 antennas and 32 delay taps, 100,000 training, 30,000 validation and 20,000 test samples per scenario. It can be downloaded from the link in the [CRNet README](https://github.com/Kylin9511/CRNet#a-data-preparation). Put the files in `./COST2100`:

```
COST2100/DATA_Htrain{in,out}.mat  DATA_Hval{in,out}.mat  DATA_Htest{in,out}.mat  DATA_HtestF{in,out}_all.mat
```

`in` is the indoor scenario (5.3 GHz) and `out` the outdoor scenario (300 MHz).

## Quick start

Train CLEAR (indoor, 200 epochs), refit its router on the frozen codec, and evaluate the exact expected NMSE on the validation split:

```bash
python main.py --data-dir ./COST2100 --scenario in --epochs 200 --batch-size 128 --workers 8 \
    --scheduler cosine --total-budget 512 --max-blockage 0.3 --safe-link-prob 0.5 --seed 7 --gpu 0 \
    --model mode_select --bidirectional --token-ffn 2048 \
    --refine-steps 2 --refine-step learned --refine-refiner \
    --output-dir runs/clear

python analysis/fit_router.py --checkpoint runs/clear/checkpoints/last.pth \
    --output runs/clear/checkpoints/last_router.pth --scenario in

python analysis/eval_exact.py --checkpoint runs/clear/checkpoints/last_router.pth \
    --output-dir runs/clear/eval_val --split val --scenario in
```

`bash scripts/clear_smoke.sh` runs every pipeline for 2 epochs to check the installation. Its numbers are meaningless.

## Reproducing the paper

Every checkpoint stores its model settings, so the analysis scripts rebuild the model from the checkpoint alone. Results go to `./clear_results` (indoor) and `./clear_results_out` (outdoor).

| Step | Command |
|---|---|
| Main schemes and ablations (indoor) | `SEEDS=7 EPOCHS=200 PARALLEL=4 FINAL_TEST=1 TEST_EVAL_SAMPLES=20000 TEST_GRID_SAMPLES=20000 bash scripts/clear_full.sh` |
| Extra seeds (Table of seeds) | the same with `SEEDS="8 9" ARMS="greedy equal risk_switch M1 A1c_bi_ffn A3c_bi_ffn_loop"` |
| Literature codecs, CLEAR without state FiLM, equal split with our backbone | `bash baselines/setup_baselines.sh`, then `BASELINE_GROUPS="multilink clean nostate bimamba" FINAL_TEST=1 bash scripts/clear_baselines.sh` |
| Robustness (state errors, correlated blockage) | `SPLIT=test NUM_SAMPLES=5000 bash scripts/clear_robustness.sh` |
| Outdoor scenario | `FINAL_TEST=1 bash scripts/clear_outdoor.sh` |
| Figure data | `python analysis/paper_figure_data.py --split test` |
| Tables and quoted numbers | `python analysis/make_paper_tables.py`, `python analysis/band_table.py`, `python analysis/outdoor_table.py` |
| Latency | `python analysis/measure_latency.py` |
| Figures | `python analysis/plot_paper_figures.py --split test` |

Every design decision in the paper was made on the validation split. The scripts evaluate the test split only when `FINAL_TEST=1`.

### Run names

The scripts use internal run names. The paper uses the names below.

| Run | Paper name |
|---|---|
| `greedy`, `equal`, `risk_switch` | Greedy, Equal, Risk-switch |
| `A3c_bi_ffn_loop` | CLEAR |
| `A1c_bi_ffn` | CLEAR-1S (one-shot) |
| `A2c_bi_ffn_no_reencode` | CLEAR-NR (loop without re-encoding) |
| `A3c_nostate` | CLEAR-NS (no state FiLM) |
| `A3_closed_loop` | CLEAR-U (unidirectional backbone, no token FFN) |
| `M1` | CLEAR-U-1S |
| `M1_risk_rule` | CLEAR-U-1S with the risk rule in place of the router |
| `A8_fixed_loop` | CLEAR-U-1S + CL |
| `A2_no_reencode` | CLEAR-U-NR |
| `AU_untied` | CLEAR-U-UT (untied encoder copy) |
| `A1_flop_control` | CLEAR-U-DN (extra dense CPU blocks) |
| `A1b_token_ffn` | CLEAR-UF-1S |
| `B_bimamba_equal` | Equal split with CLEAR's backbone (BiMamba) |
| `B_<codec>_equal`, `B_<codec>_clear` | literature codec with the equal split / with CLEAR |
| `C_<codec>` | literature codec as a noiseless single-link autoencoder |
| suffix `_T2` | the same checkpoint with the untrained closed loop at test time (+ CL) |
| suffix `_T0` | the same checkpoint decoded one-shot |

## Repository layout

```
main.py                     training entry point
models/                     CLEAR model (mamba_multi_link_csi.py), Mamba-2 backbone, literature codecs as backbones
evaluation/exact_eval.py    exact expected NMSE over the erasure patterns, grid cells, SNR bands
analysis/                   router refit, evaluation, paired bootstrap tables, figure data, figures, latency
scripts/                    experiment pipelines (clear_full.sh, clear_baselines.sh, clear_outdoor.sh, ...)
baselines/                  commits of the official literature code and a script that clones it
utils/, dataloader/         training loop, arguments, COST2100 loader
```

## Literature codecs

CRNet, CLNet and TransNet are built from their official PyTorch code (MIT licensed). `baselines/setup_baselines.sh` clones it at the commits listed in `baselines/COMMITS.txt`. We replace each codec's two fully connected layers with the link projection and the aggregator of the multi-link pipeline. The official CsiNet code is Keras, so `models/baseline_backbones.py` contains a layer-by-layer PyTorch port.

## Citation

```bibtex
@article{clear2026,
  title   = {{CLEAR}: Closed-Loop, Erasure-Aware {CSI} Feedback for {FDD} Massive {MIMO} over Multiple Blockage-Prone Uplinks},
  author  = {TBD},
  journal = {TBD},
  year    = {2026}
}
```
