# PA-AgentFCL: Privacy-Aware Telemetry for Agentic Federated Continual Learning

An experimental framework for studying **telemetry-driven autonomous control in federated continual learning (FCL)** and the effect of Gaussian telemetry perturbation on controller decisions, learning accuracy, and forgetting.

PA-AgentFCL operates on a sequence of visual domains and uses aggregated client-side measurements of **representation drift** and **replay error** to select a transparent training policy. The research implementation compares non-private and telemetry-perturbed control with federated optimization and replay baselines, with paired-seed evaluation and detailed diagnostics.

## Highlights

- **Adaptive federated continual learning:** a three-action controller (Plastic, Balanced, Retention) selects local epochs, replay ratios, and proximal strengths.
- **Telemetry-focused privacy analysis:** two-dimensional, norm-clipped client telemetry is aggregated and Gaussian-perturbed, with zero-concentrated differential privacy (zCDP) calibration under an explicitly scoped accounting model.
- **Reproducible Digits-Five protocol:** five-domain stream, non-IID Dirichlet client partitions, partial participation, client dropout, and a convolutional classifier.
- **Systematic comparisons:** FedAvg, FedProx, FedReplay, two-epoch FedReplay, non-private AgentFCL, and PA-AgentFCL.
- **Detailed evaluation:** paired statistical inference, policy-component ablations, privacy-budget sweeps, action disagreement, compute accounting, threshold sensitivity, domain-order changes, longer adaptation, and additional client-population settings.

## Experimental setting

| Setting | Configuration |
| --- | --- |
| Benchmark | Digits-Five: MNIST, USPS, SVHN, MNIST-M, SynthDigits |
| Reference domain order | MNIST → USPS → SVHN → MNIST-M → SynthDigits |
| Reference client population | 10 non-IID clients |
| Domain exposure | 4 communication rounds per domain (20 total) |
| Participation | 70% client selection; independent 10% dropout; at least 2 active clients |
| Data heterogeneity | Dirichlet label partitioning, α = 0.30 |
| Client sample budget | Up to 256 current-domain examples per round |
| Replay memory | Up to 200 examples per client |
| Model | Three-channel CNN with GroupNorm |
| Optimization | SGD; learning rate 0.03; batch size 64 |
| Main evaluation | 10 paired seeds |
| Supplementary analyses | Typically 5 paired seeds |
| Telemetry budgets | Nominal ε ∈ {1, 3, 5, 10}, δ = 10⁻⁵ |

### Adaptive action space

| Action | Replay ratio | Proximal coefficient | Local epochs |
| --- | ---: | ---: | ---: |
| Plastic | 0.15 | 0.00 | 2 |
| Balanced | 0.35 | 0.01 | 1 |
| Retention | 0.55 | 0.03 | 2 |

## Repository contents

A minimal release can be organized as follows:

```text
PA-AgentFCL/
├── README.md
├── PA_AgentFCL_Final.ipynb
└── pa_agentfcl_complete_reviewer_results.zip
```

- **`PA_AgentFCL_Final.ipynb`** — complete Google Colab research workflow, including the original experiments and additional diagnostic experiments in the same notebook.
- **`pa_agentfcl_complete_reviewer_results.zip`** — archived outputs including per-seed CSV summaries, statistical tests, saved run summaries and histories, plots, and additional experiment results.
- **`README.md`** — project overview, execution guide, and interpretation notes.

If the archive is extracted, its contents are stored under `pa_agentfcl_outputs/`, including the `remaining_reviewer_experiments/` directory. The ZIP is supplied as a convenient, independently inspectable record of the completed runs; it is not a substitute for executing the notebook when independent reproduction is required.

**Before publishing:** use the filenames shown here or update this section to match the actual uploaded files. In particular, make sure the notebook uploaded as `PA_AgentFCL_Final.ipynb` includes the final experiments.

## Run in Google Colab

1. Open `PA_AgentFCL_Final.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Choose a GPU-enabled runtime if available. The code detects CUDA and can also use CPU, though full experiments may be significantly slower.
3. In the initial configuration cell, select `RUN_MODE = "paper"` for the complete reference evaluation or `RUN_MODE = "smoke"` for a small functional check. Inspect the optional-experiment flags before starting.
4. Select **Runtime → Run all** for a fresh end-to-end reproduction. This downloads and verifies the dataset, executes the enabled experiments, generates summaries and figures, and exports results.
5. Outputs are written to `/content/pa_agentfcl_outputs/` by default. Google Colab's temporary storage does not survive every runtime reset; copy important files to persistent storage after a run.

### Dataset integrity

The notebook obtains the public preprocessed Digits-Five benchmark from a FedBN dataset mirror:

- Source: [FedBN dataset mirror](https://huggingface.co/datasets/Jemary/FedBN_Dataset)
- Archive: `digit_dataset.zip`
- Expected SHA-256: `6c006e41ce16404aab520895a5c510166453c8b58ed2e7eb7e23e133e1fa4221`

The notebook checks the archive's SHA-256 digest before loading data. Dataset download requires internet access and is subject to upstream availability.

### Dependencies

The notebook uses Python, PyTorch, NumPy, pandas, SciPy, Matplotlib, requests, and `huggingface_hub`. The installation cell installs supported missing packages in Colab. Exact dependency versions are not pinned; recorded environment details in the run manifest should be retained for reproducibility.

## Experiments and outputs

The reference workflow produces, among other artifacts:

| Output | Description |
| --- | --- |
| `main_per_seed_results.csv` | Per-seed results for the principal comparison methods |
| `main_summary.csv` | Means and standard deviations across main-evaluation seeds |
| `main_round_history.csv` | Training and evaluation history over the domain stream |
| `headline_paired_statistics.csv` | Paired statistical analysis of the principal method comparison |
| `factorial_2x2x2_results.csv` | Eight policy-component configurations |
| `factorial_all_effects_including_interactions.csv` | Factorial main and interaction effects |
| `epsilon_sweep_results.csv` | Telemetry privacy-budget sweep |
| `private_controller_action_confusion.csv` | Private versus non-private controller decisions |
| `remaining_reviewer_experiments/remaining_reviewer_results.csv` | Additional experiment runs and per-seed measurements |
| `remaining_reviewer_experiments/remaining_reviewer_summary.csv` | Aggregated additional experiment summaries |

The archive also contains supporting figures, run histories and checkpoint summaries. These files are provided to facilitate independent inspection of the reported findings.

## Evaluation and interpretation

Key reported metrics include final average accuracy across observed domains, forgetting, action selection/disagreement, local computation (client-epochs, SGD steps, training-example exposures), and uncertainty estimates across paired seeds. The expanded experiments explore the sensitivity of findings to controller thresholds, domain sequencing, exposure duration, and client-population size.

Results from the 50- and 100-client settings should be interpreted as **exploratory**: the same benchmark data are repartitioned among more clients, without increasing the total data proportionally. Differences in performance cannot be attributed to client population size alone.

Likewise, fixed-epoch and shuffled-epoch controls offer useful computation-aware comparisons but do **not** match every SGD step and training-example exposure exactly. Findings should not be interpreted as establishing a strict equal-workload advantage unless supported by an appropriately matched analysis.

## Privacy and security scope

**The telemetry mechanism is not end-to-end differentially private federated learning.** The reported Gaussian noise calibration and zCDP accounting apply to a clipped, aggregated two-dimensional telemetry release **conditional on a fixed model trajectory** and the stated replacement-adjacency and participation assumptions. They do not establish a client-level DP guarantee for the full adaptive training transcript.

Model updates and the final global model are **not** protected by this telemetry mechanism. The research simulation does not provide production-grade cryptographic secure aggregation, collusion resistance, or a deployed trusted privacy gateway. Any practical system would require an explicit threat model and additional security mechanisms and proofs.

## Reproducibility notes

- Reference methods within each paired seed share the corresponding initialization, data partitioning, participation schedule, and sampling construction.
- The updated representation-drift computation evaluates current and comparison examples with the **same broadcast model** (`DRIFT_MODE = "same_model"`). The `legacy` option exists for comparison with earlier behavior and should not be mixed into revised result tables.
- The main evaluation uses ten paired seeds, while several supplementary analyses use five.
- Outputs and numerical tables should be regenerated from actual stored experimental results rather than transcribed as fixed constants.
- Runs can be computationally intensive; persistent checkpoints and result archives should be retained for reproducibility.

## Citation

If you use this implementation or its experimental protocol, please cite the accompanying research paper:

*Privacy-Aware Telemetry for Agentic Federated Continual Learning.* INMIC 2026.

Full bibliographic details can be added after the final publication record is available.

## License

No software license is asserted in this README. Reuse and redistribution permissions are governed by any license file separately included in the repository. The external dataset remains subject to its original terms.
