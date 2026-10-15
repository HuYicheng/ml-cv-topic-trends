# Precision review of the topic rules

Every topic's rules were checked by reading random papers they match, in two periods (2020-2023 and 2024-2026) and two strata:
papers matched by the **title** rule (T) and papers matched **only by the abstract** rule (A). Labels: Y = the paper engages with
the topic in the topic's sense; P = topic's sense, passing mention only (counted as correct, as in the share definition);
N = wrong sense or not the topic; ? = cannot tell. Every item was judged from the paper's abstract (user, 2026-10-15:
"你不能只按title来判断，你要看abstract"). Reading was split among sub-agents; every wrong item they reported was read again
by the analyst before a rule was changed.

- **Round 1** (seed 0, up to 10 + 10 items per topic and period): wrong senses found; about 120 rule changes measured on
  the full data (papers removed / added, removed papers read) and adopted.
- **Round 2** (seed 1, fresh samples after the round-1 changes, 6 + 6 items per topic and period, full abstract for every
  item): the table below. Causes shared by two or more wrong items were fixed afterwards (about 45 changes, e.g. the
  quantization rule now needs a model context in abstracts, "tabular" in the RL sense, "long sequence" for long videos,
  "factual knowledge", conversational agents); the round-2 counts are for the rules *before* those last fixes.
- Sample sizes are small: read the counts as "high / medium / low", not as measured rates. Recall was not measured;
  coverage: 94.8% (2020) to 98.4% (2026) of papers belong to at least one topic.

| topic | 2020-23 T Y/P/N | 2020-23 A Y/P/N | 2024-26 T Y/P/N | 2024-26 A Y/P/N |
|---|---|---|---|---|
| 3D 4D generation | 6/0/0 | 4/0/2 | 6/0/0 | 5/0/1 |
| 3D Gaussian splatting | 0/0/0 | 0/0/3 | 6/0/0 | 5/1/0 |
| 3D humans avatars motion | 6/0/0 | 5/0/1 | 5/0/1 | 6/0/0 |
| 3D reconstruction SfM SLAM | 6/0/0 | 4/2/0 | 6/0/0 | 5/1/0 |
| AI for science molecules proteins materials | 6/0/0 | 0/4/2 | 6/0/0 | 1/3/2 |
| Active learning data selection | 6/0/0 | 2/0/4 | 6/0/0 | 5/1/0 |
| Adversarial robustness attacks | 6/0/0 | 5/0/1 | 6/0/0 | 5/1/0 |
| Agents tool use | 4/0/0 | 3/0/3 | 6/0/0 | 4/2/0 |
| Alignment RLHF preference optimization | 4/0/2 | 4/0/2 | 6/0/0 | 4/1/1 |
| Autonomous driving | 6/0/0 | 5/1/0 | 5/0/1 | 3/3/0 |
| Autoregressive models tokenizers | 6/0/0 | 3/3/0 | 6/0/0 | 4/2/0 |
| Backdoor data poisoning | 6/0/0 | 5/0/1 | 6/0/0 | 6/0/0 |
| Bandits online learning | 6/0/0 | 5/1/0 | 6/0/0 | 6/0/0 |
| Bayesian probabilistic inference | 6/0/0 | 6/0/0 | 6/0/0 | 4/2/0 |
| Causal inference discovery | 6/0/0 | 3/0/3 | 6/0/0 | 5/0/1 |
| Clustering unsupervised learning | 4/0/2 | 4/2/0 | 6/0/0 | 4/2/0 |
| Code generation software | 6/0/0 | 5/1/0 | 6/0/0 | 4/2/0 |
| Combinatorial optimization scheduling | 6/0/0 | 3/3/0 | 6/0/0 | 4/1/1 |
| Continual incremental learning | 6/0/0 | 6/0/0 | 6/0/0 | 6/0/0 |
| Convolutional networks | 4/0/2 | 3/3/0 | 6/0/0 | 3/2/1 |
| Datasets benchmarks | 4/0/2 | 6/0/0 | 5/0/1 | 6/0/0 |
| Depth stereo optical flow matching | 5/0/1 | 5/0/1 | 6/0/0 | 4/2/0 |
| Diffusion models score based | 5/0/1 | 5/1/0 | 6/0/0 | 6/0/0 |
| Domain adaptation generalization shift | 6/0/0 | 5/1/0 | 6/0/0 | 4/2/0 |
| Efficient inference sampling | 3/1/2 | 4/2/0 | 5/1/0 | 5/1/0 |
| Energy based models | 6/0/0 | 6/0/0 | 6/0/0 | 6/0/0 |
| Equivariant geometric deep learning | 5/0/1 | 5/1/0 | 6/0/0 | 5/1/0 |
| Evaluation of LLMs LLM as judge | 6/0/0 | 5/0/0 (1?) | 6/0/0 | 6/0/0 |
| Event cameras computational imaging | 5/0/1 | 6/0/0 | 6/0/0 | 6/0/0 |
| Face analysis | 6/0/0 | 3/3/0 | 6/0/0 | 5/1/0 |
| Fairness bias | 6/0/0 | 5/0/0 (1?) | 6/0/0 | 3/2/1 |
| Federated learning | 6/0/0 | 6/0/0 | 6/0/0 | 5/1/0 |
| Few shot zero shot open vocabulary | 6/0/0 | 4/2/0 | 6/0/0 | 6/0/0 |
| Fine tuning PEFT prompting | 6/0/0 | 4/2/0 | 6/0/0 | 2/4/0 |
| Flow matching normalizing flows | 6/0/0 | 6/0/0 | 6/0/0 | 6/0/0 |
| Foundation models | 6/0/0 | 5/1/0 | 6/0/0 | 6/0/0 |
| GANs | 6/0/0 | 5/1/0 | 6/0/0 | 3/3/0 |
| Graph learning knowledge graphs | 5/0/1 | 5/0/1 | 6/0/0 | 5/1/0 |
| Graph neural networks | 6/0/0 | 5/0/1 | 6/0/0 | 5/1/0 |
| Hallucination factuality | 2/0/0 | 4/0/2 | 5/0/1 | 1/4/1 |
| Healthcare clinical | 6/0/0 | 4/2/0 | 6/0/0 | 6/0/0 |
| Human pose body hands | 6/0/0 | 5/1/0 | 6/0/0 | 5/1/0 |
| Image classification recognition | 6/0/0 | 0/6/0 | 6/0/0 | 1/5/0 |
| Image generation editing | 6/0/0 | 3/1/2 | 6/0/0 | 6/0/0 |
| Image restoration low level vision | 6/0/0 | 6/0/0 | 6/0/0 | 1/4/1 |
| Image retrieval metric learning | 6/0/0 | 4/1/1 | 6/0/0 | 4/2/0 |
| In context learning | 6/0/0 | 4/1/1 | 6/0/0 | 4/2/0 |
| Interpretability explainability | 4/0/1 (1?) | 6/0/0 | 4/0/2 | 5/1/0 |
| Kernel methods Gaussian processes | 6/0/0 | 6/0/0 | 6/0/0 | 4/0/2 |
| Knowledge distillation | 5/0/0 (1?) | 4/2/0 | 6/0/0 | 4/1/1 |
| LLM safety jailbreak | 3/0/1 | 2/4/0 | 5/0/1 | 6/0/0 |
| LLMs language models | 5/0/1 | 5/1/0 | 6/0/0 | 5/1/0 |
| Learning theory generalization | 5/0/1 | 6/0/0 | 6/0/0 | 6/0/0 |
| Long context | 6/0/0 | 5/0/1 | 5/0/1 | 2/2/2 |
| Mathematical reasoning theorem proving | 6/0/0 | 5/0/1 | 5/0/1 | 2/4/0 |
| Medical imaging | 6/0/0 | 1/3/2 | 6/0/0 | 4/2/0 |
| Meta learning | 6/0/0 | 6/0/0 | 6/0/0 | 4/2/0 |
| Mixture of experts | 6/0/0 | 5/0/1 | 6/0/0 | 5/1/0 |
| Model merging editing unlearning | 6/0/0 | 4/2/0 | 6/0/0 | 4/1/1 |
| Multi agent learning game theory | 6/0/0 | 5/0/1 | 6/0/0 | 5/1/0 |
| Multi task transfer learning | 6/0/0 | 5/0/1 | 5/0/1 | 5/0/1 |
| Multimodal learning cross modal | 6/0/0 | 6/0/0 | 6/0/0 | 5/0/1 |
| NLP tasks | 5/0/1 | 2/3/1 | 4/0/2 | 3/3/0 |
| Neural ODEs dynamical systems | 6/0/0 | 6/0/0 | 6/0/0 | 6/0/0 |
| Neural architecture search AutoML | 5/0/1 | 6/0/0 | 6/0/0 | 4/2/0 |
| Neural fields NeRF | 6/0/0 | 5/1/0 | 6/0/0 | 5/1/0 |
| Novel view synthesis neural rendering | 6/0/0 | 6/0/0 | 6/0/0 | 6/0/0 |
| OCR documents charts | 6/0/0 | 1/2/3 | 6/0/0 | 4/2/0 |
| OOD anomaly detection | 6/0/0 | 5/0/1 | 6/0/0 | 5/1/0 |
| Object 6D pose estimation | 6/0/0 | 4/1/1 | 6/0/0 | 2/1/3 |
| Object detection | 5/0/1 | 4/2/0 | 6/0/0 | 3/3/0 |
| Offline RL imitation learning | 6/0/0 | 6/0/0 | 6/0/0 | 6/0/0 |
| Optimal transport | 6/0/0 | 6/0/0 | 6/0/0 | 5/1/0 |
| Optimization | 5/0/1 | 3/2/1 | 6/0/0 | 5/1/0 |
| Point clouds 3D understanding | 6/0/0 | 4/2/0 | 6/0/0 | 5/1/0 |
| Privacy | 6/0/0 | 4/2/0 | 6/0/0 | 4/2/0 |
| Quantization pruning compression | 6/0/0 | 4/1/1 | 5/0/1 | 0/0/6 |
| Reasoning chain of thought | 4/0/2 | 4/1/1 | 6/0/0 | 3/3/0 |
| Recommendation information retrieval | 6/0/0 | 2/4/0 | 4/0/2 | 2/3/1 |
| Recurrent networks LSTMs | 6/0/0 | 3/3/0 | 6/0/0 | 2/4/0 |
| Reinforcement learning | 6/0/0 | 6/0/0 | 6/0/0 | 5/1/0 |
| Remote sensing earth observation | 6/0/0 | 6/0/0 | 6/0/0 | 4/2/0 |
| Retrieval augmented generation | 6/0/0 | 4/2/0 | 6/0/0 | 5/1/0 |
| Robotics embodied AI | 6/0/0 | 2/4/0 | 5/0/1 | 1/5/0 |
| Scaling laws pretraining | 6/0/0 | 1/5/0 | 6/0/0 | 4/2/0 |
| Scientific ML PDEs physics | 6/0/0 | 4/0/2 | 5/0/1 | 4/2/0 |
| Segmentation | 6/0/0 | 4/2/0 | 6/0/0 | 1/3/2 |
| Self supervised contrastive learning | 6/0/0 | 5/1/0 | 5/0/1 | 4/1/1 |
| Semi weakly supervised noisy labels | 6/0/0 | 5/1/0 | 6/0/0 | 5/0/1 |
| Speech audio music | 6/0/0 | 4/2/0 | 5/0/1 | 1/3/2 |
| Spiking neuromorphic networks | 6/0/0 | 5/0/1 | 6/0/0 | 4/2/0 |
| State space models Mamba linear attention | 5/0/1 | 5/0/1 | 6/0/0 | 5/1/0 |
| Statistical estimation high dimensional statistics | 6/0/0 | 4/2/0 | 4/0/2 | 4/2/0 |
| Synthetic data data augmentation | 6/0/0 | 4/2/0 | 6/0/0 | 3/2/1 |
| Tabular data | 6/0/0 | 2/0/4 | 6/0/0 | 2/2/2 |
| Time series forecasting | 6/0/0 | 5/1/0 | 6/0/0 | 2/3/0 (1?) |
| Tracking re identification | 6/0/0 | 4/2/0 | 5/0/1 | 5/0/1 |
| Transformers attention | 6/0/0 | 6/0/0 | 6/0/0 | 6/0/0 |
| Uncertainty calibration conformal | 6/0/0 | 4/0/2 | 4/0/2 | 6/0/0 |
| VAEs vector quantized models | 6/0/0 | 6/0/0 | 6/0/0 | 4/2/0 |
| Video generation editing | 6/0/0 | 4/0/2 | 6/0/0 | 4/2/0 |
| Video understanding action recognition | 6/0/0 | 2/4/0 | 6/0/0 | 5/1/0 |
| Vision language multimodal LLMs | 6/0/0 | 6/0/0 | 6/0/0 | 6/0/0 |
| Visual QA captioning grounding | 6/0/0 | 3/3/0 | 6/0/0 | 3/2/1 |
| Watermarking deepfakes forensics | 5/0/1 | 3/2/1 | 6/0/0 | 2/4/0 |
| World models video prediction | 6/0/0 | 5/1/0 | 4/0/2 | 5/1/0 |

Round 2 totals: T: 1258 read, 95.8% correct (Y+P), 95.6% Y; A: 1269 read, 92.2% correct (Y+P), 71.2% Y.

