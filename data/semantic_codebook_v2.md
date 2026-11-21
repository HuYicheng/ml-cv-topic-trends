# Topic codebook for semantic labelling, ML/CV, v2 (2026-11-20, full run)

v2 settles the boundary cases the 500-paper pilot exposed (its labellers' notes and the blind re-judgement of
disagreements); changes from v1 are marked [v2].

The same scheme as the robotics study (robotics-trends/data/semantic_codebook_v2.md), applied to the 106 topics of the
ML/CV keyword rules (mlcv-trends/scripts/04_topics.py), in the same order and with the same names. The six "[term]" word
counts of that study are not topics and are not labelled here.

Read the paper's title and abstract and decide which topics it is about. Two labels per paper:

- **involves**: the paper's own work (its method, model, theory, task, data or evaluation) involves the topic.
  A topic named only as background, motivation, contrast ("unlike X", "outperforms X-based methods"), related work or
  future work does NOT count. A paper usually involves several topics.
- **main**: the 1 to 3 topics that best describe the paper's main contribution. Every main topic is also an involves topic.
  If no topic fits the contribution, "main" may be empty.

Judge the meaning, not the words: a paper counts even if it never uses the topic's usual words, and does not count when a
word is used in another sense. When unsure, decide by the definition; do not count a topic just because a word appears.
- A well-known model named in the abstract counts for what it is when the paper's own work uses, adapts or studies it
  (BERT and ViT are transformers; GPT-4, LLaMA and Qwen are LLMs; CLIP and LLaVA are vision-language models; Stable
  Diffusion is a diffusion model; ResNet and U-Net are convolutional networks; SAM and DINOv2 are vision foundation
  models). The paper's own method name or acronym is not evidence of anything.
- A routine ingredient named in passing counts only where the topic's definition says so (training with Adam is not
  Optimization; a "Bayesian" hyperparameter search is not Bayesian inference).
- Decide from the title and abstract only. [v2] What follows directly from the abstract counts (a nearest-neighbour
  scene-flow loss on LiDAR implies point clouds); do not guess further, and never from a method's own name. A generative
  model of unstated kind ("a pretrained text-to-image model") gets no model-family topic (1-5).
- [v2] Tasks and data used in the evaluation: a task topic (68-92, 96, 97) counts when the paper's experiments evaluate on
  that task with a named benchmark or a clearly stated task (ImageNet -> 68, GLUE -> 97, Cityscapes segmentation -> 70);
  a vague mention ("vision and language tasks") does not. An application domain (80-82, 93-102) needs the paper to
  target or frame its work in that domain; one benchmark from the domain among generic test sets does not count.
- [v2] In a benchmark or evaluation paper, the models it evaluates are its subject and count (an MLLM benchmark ->
  19); in a method paper, models named only as baselines do not. Possible extensions or future work never count.
- [v2] When no topic fits, involves and main may both be empty.

## A. Architectures and model families

1. **Diffusion models / score-based**: denoising diffusion and score-based generative models (DDPM, DDIM, latent
   diffusion, consistency models), for any purpose: training, sampling, guidance, distillation, analysis or use as a
   prior or component. Not: physical, heat, chemical, graph, information or opinion diffusion; diffusion MRI or tensors;
   diffusion maps or diffusion as feature propagation; Langevin dynamics or SDEs without a diffusion generative model.
   [v2] Flow-matching or rectified-flow generators described in diffusion terms (denoising steps, e.g. FLUX, SD3) count
   for both 1 and 2. Diffusion bridges and Schrodinger bridges count for 1 only when built as a denoising generative
   model.
2. **Flow matching / normalizing flows**: flow matching, rectified flow, stochastic interpolants, normalizing flows and
   continuous normalizing flows. Not: GFlowNets; optical, scene or traffic flow; "flow" as a pipeline or process.
   [v2] A paper about flow matching itself (theory, objectives, samplers) is 2; also 1 only if diffusion models are
   part of it.
3. **GANs**: generative adversarial networks (StyleGAN, CycleGAN, GAN inversion, adversarially trained generators).
   Not: adversarial examples, attacks or robustness training (49); a discriminator used only for domain alignment (36).
   [v2] Count only when adversarial generator-discriminator training is described, not from a method name.
4. **VAEs / vector-quantized models**: variational autoencoders and vector-quantized latent models (VQ-VAE, VQ-GAN,
   learned codebooks of discrete latents). Not: plain autoencoders; quantizing weights or activations for compression (58).
5. **Autoregressive models / tokenizers**: autoregressive generation (next-token or next-scale prediction) as a modelling
   choice that the work builds on or studies, for images, video, audio, 3D, actions, molecules or text; tokenizers that turn
   images, video or audio into discrete tokens; analyses of next-token prediction. Not: merely using an LLM (18); classical
   statistical AR / ARMA time-series models.
   [v2] Text tokenization (tokenizer design, tokenizer-free or byte-level LMs) counts too.
6. **Transformers / attention**: the paper's own model is built on, modifies or analyses transformers or attention
   (self- or cross-attention, ViT, Swin, DETR, BERT-type encoders, positional encodings, attention mechanisms, theory of
   transformers). Not: LLMs, VLMs or diffusion models used without reference to their architecture; "attention" in the
   loose sense of focus.
7. **State-space models / Mamba / linear attention**: structured or selective state-space sequence models (S4, S5,
   Mamba), linear attention, linear recurrent models (RWKV, RetNet) and other sub-quadratic sequence architectures.
   Not: classical state-space models in control or statistics (Kalman filters, HMM state spaces).
8. **Mixture of experts**: mixture-of-experts layers or models, sparse gating and routing among experts, expert
   parallelism. Not: ensembles; "experts" meaning people or expert demonstrations.
   [v2] Routing between whole models (LLM routers, cascades) and dynamic merging of task modules are not 8 unless
   described as a mixture of experts.
9. **Graph neural networks**: neural networks on graphs: GNNs, graph convolutions, graph attention, message passing
   networks, graph transformers, and their theory (expressivity, oversmoothing). Not: graph algorithms without neural
   networks; scene graphs used without a GNN. [v2] Mesh or graph convolutions on meshes count for 9 and 13.
10. **Convolutional networks**: the paper's own model is a convolutional network or contains a stated convolutional design
    (CNNs, ResNet, U-Net, ConvNeXt, convolution layers or kernels), or the paper studies convolutional architectures or their
    theory. Not: graph or mesh convolutions (9); convolution as a mathematical operation outside a network.
11. **Recurrent networks / LSTMs**: RNNs, LSTMs, GRUs and their training or theory. Not: linear recurrent SSM-type models
    (7); "recurrent" meaning repeated or iterative refinement.
12. **Neural ODEs / dynamical systems**: neural ODEs, SDEs and CDEs, continuous-time neural models, Koopman operators;
    learning, modelling or analysing dynamical systems with ML; analysing learning itself as a dynamical system (e.g.
    gradient-flow dynamics). Not: diffusion SDEs used only as the generative process (1).
    [v2] Ordinary networks used for forecasting (weather, ocean, traffic) are 94/95, not 12. Theoretical or empirical
    studies of training dynamics (gradient flows, phases of training, grokking, how circuits form during training)
    count.
13. **Equivariant / geometric deep learning**: models with built-in symmetry (equivariance or invariance to groups such as
    E(3), SE(3), SO(3), permutations, scaling), and deep learning on manifolds, meshes or other geometric domains. Not:
    SE(3) or SO(3) used only to describe poses or rotations.
    [v2] Topological deep learning (e.g. persistent-homology layers) counts here.
14. **Energy-based models**: energy-based models and energy-based training or generation. Not: physical energy;
   "energy efficiency".
15. **Spiking / neuromorphic networks**: spiking neural networks and neuromorphic computing or hardware for them.
    Not: analysis of recorded neural spiking data without an SNN; event cameras alone (83).
    [v2] Brain-inspired analog or physical neural hardware counts as neuromorphic computing.
16. **Neural fields / NeRF**: neural radiance fields, implicit neural representations, coordinate networks, neural signed
    distance or occupancy functions.
17. **3D Gaussian splatting**: 3D/4D Gaussian splatting and other Gaussian-splat scene representations.

## B. Large and foundation models

18. **LLMs / language models**: text language models (large language models or smaller pretrained LMs such as BERT or
    GPT-2) that the work uses, trains, adapts, analyses or evaluates. A multimodal LLM counts here only when the work
    concerns its language-model part or its text-only use; otherwise it is 19 alone.
    [v2] Speech-text LLMs without vision are 18 and 96 (and 47), not 19.
19. **Vision-language / multimodal LLMs**: vision-language models (CLIP-type dual encoders, captioning or VQA models
    pretrained on image-text data), multimodal LLMs (LLaVA, GPT-4V, Qwen-VL), audio-visual or omni-modal LLMs,
    vision-language pretraining. Not: multimodal fusion without language (47 only).
    [v2] Every 19 paper also counts for 47. Vision-language-action (VLA) models count for 19 (they are built on a VLM).
20. **Foundation models**: large pretrained general-purpose models treated as such: vision foundation models (SAM,
    DINOv2, large pretrained ViTs), foundation models for other modalities or sciences (time series, medical, molecules,
    graphs, weather, robotics), generalist multi-task models, and studies of foundation models as a class. Using an LLM or
    VLM counts here only when the paper treats it as a general-purpose foundation model for another task or domain.
    [v2] Count when the work builds, adapts, evaluates or analyses a model that the abstract presents as a foundation,
    generalist or large general-purpose pretrained model (other than plain LLM or VLM use), or uses an LLM or VLM as a
    general-purpose foundation model in a non-language domain (robotics, medicine, science, time series). Fine-tuning
    large pretrained ViTs or SAM-type models counts.
21. **Agents / tool use**: agents built on LLMs or VLMs (web, GUI, computer-use, coding, research or data-science agents,
    agentic workflows, multi-agent LLM systems) and tool use or function calling by models. Not: RL agents that are not
    LLM-based (42, 44).
    [v2] LLM- or VLM-driven embodied agents count for 21 and 98. A dialogue model called an agent, with no tools,
    actions or environment, does not count.
22. **Reasoning / chain-of-thought**: explicit reasoning by a model as a capability or method: chain-of-thought,
    reasoning models, test-time compute or scaling, reasoning traces, logical, commonsense, spatial or visual reasoning
    tasks for models. Not: "reasoning" as a loose word for inference or computation; probabilistic inference (63).
    [v2] Test-time scaling without reasoning (best-of-N sampling of a generator) does not count. Every 23 paper also
    counts for 22. Puzzles used as reasoning tests are 22, not 101.
23. **Mathematical reasoning / theorem proving**: models solving mathematics problems, formal theorem proving (Lean, Coq,
    Isabelle), autoformalization, formal verification of code or proofs with models. Not: the paper's own mathematical
    theory or proofs (61, 62).
24. **Retrieval-augmented generation**: models whose generation or prediction uses retrieved documents, memories or
    examples (RAG, retrieval-augmented LMs or VLMs, knowledge-augmented generation). Not: retrieval systems without a
    generator or predictor that consumes the results (102).
    [v2] Retrieval used only to collect training data is 41, not 24. A retriever that is the contribution of a RAG
    system counts for 24 and 102.
25. **Alignment / RLHF / preference optimization**: aligning models with human preferences, values or instructions:
    RLHF, DPO and other preference optimization, reward models for LLMs or generators, constitutional AI, value alignment.
    Not: "alignment" of features, domains, modalities, images or distributions.
    [v2] Counts for any model trained from human or AI preference or feedback signals (pairwise or ranked outputs,
    reward models of preferences), including preference-based RL in control, and for value or safety alignment.
    Training from task rewards alone is 42 only; instruction following alone is not 25.
26. **Fine-tuning / PEFT / prompting**: adapting a pretrained model by fine-tuning, parameter-efficient tuning (LoRA,
    adapters, prompt or prefix tuning), prompt engineering or prompt learning, or instruction tuning, when this is part of
    the paper's own method or study. Not: fine-tuning named only as a baseline or in background.
    [v2] Prompt design or learning counts as a method even without training (incl.
    visual prompts for VLMs); merely calling a model with a prompt, or using it as is, does not. Post-training a
    pretrained model by RL or preference optimization also counts (with 42 or 25). Fine-tuning an ordinary pretrained
    network inside the method counts.
27. **In-context learning**: in-context learning by LLMs or transformers (learning from demonstrations in the prompt,
    few-shot prompting), its mechanisms and theory, in-context regression.
28. **Hallucination / factuality**: unfaithful, fabricated or factually wrong content produced by language or
    vision-language models (incl. object hallucination), and factuality or faithfulness of generated content.
29. **Code generation / software**: models that generate, complete, translate, repair or reason about code; program
    synthesis; software-engineering tasks (SWE-bench); code agents. Not: "code" as a codebook, binary code or hash code.
30. **Model merging / editing / unlearning**: merging model weights, task arithmetic, knowledge or model editing, machine
    unlearning. Not: "unlearning" a bias or spurious feature in the loose sense (52).
    [v2] Concept erasure counts for 30, and for 56 too when it removes unsafe or protected content. Merging adapters or
    modules counts.
31. **Long context**: handling long inputs in sequence models: long-context LLMs, extending the context window, length
    generalization or extrapolation, long-range sequence benchmarks. Not: long videos or long horizons as such.
    [v2] Long videos count when the paper handles their long token sequences or context (memory, context length) in a
    sequence model.
32. **Scaling laws / pretraining**: scaling laws and compute-optimal training, pretraining data, corpora, mixtures and
    recipes for large models, emergent abilities, large-scale pretraining as the object of study. Not: self-supervised
    pretraining objectives alone (33).
    [v2] A general method only evaluated in a pretraining setting (an optimizer, sparse training) does not count;
    pretraining itself (data, recipes, scaling) must be studied.
## C. Learning paradigms

33. **Self-supervised / contrastive learning**: learning representations without labels: self-supervised and contrastive
    learning (SimCLR, MoCo, CLIP-style contrastive objectives), masked modelling (MAE), representation learning as the
    subject. Not: contrastive decoding, contrastive divergence, contrastive explanations.
    [v2] Self-supervised training of a task model (e.g. self-supervised depth or scene flow) counts. A contrastive or
    masked-modelling loss that is a named part of the method counts.
34. **Semi- / weakly supervised & noisy labels**: semi-supervised learning, weak supervision, pseudo-labels, learning with
    noisy, partial or positive-unlabelled labels.
35. **Few-shot / zero-shot / open-vocabulary**: learning or recognising with few or no labelled examples of the target
    classes or tasks: few-shot learning, zero-shot transfer or recognition, open-vocabulary and open-set recognition of
    novel categories. Not: "zero-shot" meaning data-free (zero-shot quantization or NAS) or training-free (use the
    method's actual topics).
    [v2] Counts when a model is applied to classes, tasks, identities or domains it was not trained on, without
    per-target training (incl. zero-shot speaker or identity personalization). "Zero-shot" meaning training-free or
    unsupervised does not. Naming novel classes is 35; rejecting unknowns is 55.
36. **Domain adaptation / generalization / shift**: domain adaptation (incl. unsupervised and source-free), domain
    generalization, distribution or covariate shift, test-time adaptation or training, out-of-distribution
    generalization. Not: detecting OOD inputs (55).
    [v2] Counts when robustness or generalization under a distribution shift is a stated aim or a reported result.
    Continual test-time adaptation counts for 36 and 37. Narrowing a sim-to-real gap counts.
37. **Continual / incremental learning**: continual, lifelong or class/task-incremental learning; catastrophic forgetting.
    [v2] A model must learn over a sequence of tasks or a data stream; a growing retrieval index alone does not count.
38. **Meta-learning**: learning to learn (MAML-type methods, meta-training across tasks, learned optimizers or priors).
    [v2] Well-known meta-learning model families (Neural Processes) count.
39. **Federated learning**: federated or decentralised collaborative training across clients.
40. **Knowledge distillation**: teacher-student distillation of models (incl. LLM, diffusion-step and self-distillation)
    and dataset distillation. Not: "distil" in the loose sense of summarising; score distillation sampling for 3D (90).
    [v2] Distilling a diffusion model into a few-step student counts; score distillation (SDS-type) losses that
    optimize an output (3D asset, image) do not. Dataset or graph condensation counts.
41. **Active learning / data selection**: active learning, data selection, data pruning, data curation, coresets, data
    valuation, choosing which samples to train or label on.
    [v2] Curriculum learning that selects or orders training samples counts.
42. **Reinforcement learning**: learning from rewards by trial and error in any form: deep RL, policy gradient,
    Q-learning, actor-critic, MDPs, RL fine-tuning of LLMs (PPO, GRPO, RLVR), offline RL (also 43), inverse RL, RL theory.
43. **Offline RL / imitation learning**: offline (batch) RL, imitation learning, behaviour cloning, inverse RL, learning
    from demonstrations.
44. **Multi-agent learning / game theory**: multi-agent RL and learning, game theory, equilibria, mean-field games,
    mechanism design, auctions, learning in games.
    [v2] Shapley values or game-theoretic interactions used to explain models are 53, not 44.
45. **Bandits / online learning**: multi-armed and contextual bandits, online learning and online convex optimization,
    regret minimization, Thompson sampling, UCB, best-arm identification.
46. **Multi-task / transfer learning**: multi-task learning, transfer learning, transferability of representations or
    models, auxiliary tasks. Not: transferability of adversarial attacks (49).
    [v2] Sim-to-real counts here only when framed as transfer or pretrain-then-fine-tune.
47. **Multimodal learning / cross-modal**: learning from or across two or more modalities (image, text, audio, video,
    depth, LiDAR, sensors, tables...): fusion, alignment, cross-modal generation, translation or retrieval, missing
    modalities. VLM and MLLM papers count here too. Not: "multimodal" meaning multiple modes of a distribution, objective
    or behaviour.
    [v2] Text-conditioned generation (text-to-image, -video, -3D, -audio) counts only when the work concerns the
    relation between the condition and the output (alignment, compositional prompt following, combining condition
    modalities) or generates several modalities jointly, not when text is only the prompt. Different views or
    resolutions of one modality are not different modalities.
48. **Clustering / unsupervised learning**: clustering, unsupervised learning without labels (incl. unsupervised
    segmentation, discovery or anomaly detection), dimensionality reduction, manifold learning. Not: self-supervised
    representation learning alone (33); unsupervised domain adaptation alone (36).

## D. Trustworthy ML

49. **Adversarial robustness / attacks**: adversarial examples, attacks and defences, adversarial training for
    robustness, certified robustness, randomized smoothing, adversarial prompts or perturbations against any model.
    Not: GAN-style adversarial training (3); domain-adversarial training (36).
50. **Backdoor / data poisoning**: backdoor or trojan attacks and defences, data poisoning. Not: the "backdoor criterion"
    of causal inference (64).
    [v2] Unlearnable examples (availability poisoning to protect data) count, with 51 when framed as privacy.
51. **Privacy**: privacy of data or models: differential privacy, DP-SGD, membership inference, data extraction or
    reconstruction attacks, model inversion, private training or inference.
    [v2] Model stealing or extraction counts. Privacy only as a motivation of the setting (federated, black-box) does
    not.
52. **Fairness / bias**: fairness, social or demographic bias, stereotypes, protected attributes, group robustness,
    spurious correlations. Not: statistical bias of an estimator; inductive bias; bias terms.
    [v2] Popularity bias toward items in recommenders counts. Position, length or style bias of models or judges does
    not.
53. **Interpretability / explainability**: explaining or interpreting models: feature attribution and saliency maps,
    mechanistic interpretability, probing, concept-based explanations, sparse autoencoders for interpretability,
    counterfactual explanations, inherently interpretable models. Not: salient object detection; "interpretable" used only
    as a loose adjective for the method's output.
    [v2] Physically meaningful or identifiable parameters called "interpretable" do not count. Analyses of what a model
    stores or computes internally count.
54. **Uncertainty / calibration / conformal**: estimating or quantifying predictive uncertainty (epistemic, aleatoric),
    confidence calibration, conformal prediction, uncertainty-aware methods. Not: camera or sensor calibration.
55. **OOD / anomaly detection**: detecting out-of-distribution inputs, anomalies, outliers or novelties (incl. industrial
    and video anomaly detection, open-set rejection, failure detection). Not: generalizing under shift (36).
    [v2] Detecting misclassified or failed predictions of a predictor counts. Hallucination detection in generative
    models is 28 (and 54 if by uncertainty), not 55.
56. **LLM safety / jailbreak**: safety of LLMs and other generative models: jailbreaks, red-teaming, harmful or unsafe
    content, prompt injection, refusal, guardrails, safety alignment, concept erasure for safety.
57. **Watermarking / deepfakes / forensics**: watermarking of models or content, deepfake or AI-generated content
    detection, image or video forensics and forgery detection, copyright protection, model or content fingerprinting.
    [v2] Steganography counts.

## F. Efficiency

58. **Quantization / pruning / compression**: compressing models: quantization of weights or activations, pruning,
    sparsification, low-bit or binary networks, low-rank model compression, lottery tickets. Not: vector-quantized latents
    (4); data pruning (41); token pruning (59).
    [v2] Reducing trainable parameters during fine-tuning is 26, not 58; compressing the model or its weight updates
    counts. Sparsity or quantization used to speed up training counts.
59. **Efficient inference / sampling**: making inference or generation faster or cheaper: efficient attention, speculative
    decoding, KV-cache methods, token pruning or merging, early exit, few-step or accelerated sampling of diffusion models,
    serving systems. Not: efficient training alone.
    [v2] Counts when lower inference or generation cost is a stated goal of the method or of a designed component, not
    when efficiency is a side claim.
60. **Neural architecture search / AutoML**: neural architecture search, AutoML, hyperparameter optimization.

## G. Theory and foundations

61. **Learning theory / generalization**: theoretical analysis of learning: generalization bounds, sample complexity, PAC
    and VC theory, Rademacher complexity, benign overfitting, double descent, implicit bias, NTK, minimax rates and lower
    bounds, other theorems about learning. Not: empirical "generalization" without theory.
    [v2] Theorems about learning (generalization, sample complexity, regret, convergence of learning in RL or games,
    expressivity) are 61; rates of optimization algorithms on objectives are 62; give both when both are present.
62. **Optimization**: optimization as the subject: optimizers (SGD, Adam variants, second-order methods), convergence
    analysis, convex and non-convex optimization, bilevel and min-max problems, zeroth-order methods, learning-rate
    schedules, optimization theory of training. Not: training a model with a standard optimizer; "optimize" meaning
    improve; combinatorial optimization problems (101).
    [v2] A method merely posed as a bilevel or min-max problem and solved by standard means does not count. Empirical
    studies of optimization behaviour (loss landscapes, mode connectivity, training instabilities) count.
63. **Bayesian & probabilistic inference**: Bayesian methods and probabilistic inference: posteriors, variational
    inference, MCMC and Langevin sampling, sequential Monte Carlo, importance sampling, probabilistic models and
    programming, Bayesian neural networks, Bayesian optimization as a method studied. Not: "Bayesian" named only for a
    routine tool.
    [v2] Posterior sampling for inverse problems (e.g. with a diffusion prior) counts. Amortized samplers such as
    GFlowNets count here.
64. **Causal inference / discovery**: causal effects and treatment effects, causal discovery and structure learning,
    causal representation learning, counterfactual inference, confounding, instrumental variables, interventions.
    Not: causal (masked) attention or causal language modelling; "causal" as a loose word.
65. **Optimal transport**: optimal transport, Wasserstein distances, Sinkhorn, Schrödinger bridges.
66. **Kernel methods / Gaussian processes**: kernel methods (RKHS, kernel ridge regression, SVMs, random features, kernel
    mean embeddings) and Gaussian processes.
    [v2] Bayesian optimization counts only when a Gaussian process or kernel surrogate is stated.
67. **Statistical estimation / high-dimensional statistics**: statistical estimation and inference: high-dimensional
    regression, sparse recovery, Lasso, matrix or tensor completion and factorization, robust statistics, hypothesis
    testing, estimation rates.
    [v2] A matrix or tensor factorization used as a computational tool inside another method does not count.

## H. Vision tasks

68. **Image classification / recognition**: classifying or recognising images (incl. fine-grained and long-tailed
    recognition); image classification as a task the paper addresses or evaluates on (e.g. ImageNet accuracy).
    [v2] Long-tailed or class-imbalanced recognition counts only for images.
69. **Object detection**: 2D and 3D object detection, open-vocabulary and oriented detection, detectors (DETR, YOLO),
    human-object interaction detection. Not: salient or camouflaged object detection (70); anomaly or OOD detection (55).
70. **Segmentation**: semantic, instance, panoptic, interactive, referring and medical segmentation; salient and
    camouflaged object segmentation; matting; promptable segmentation (SAM). Not: action, temporal, word or speaker
    segmentation.
71. **Tracking / re-identification**: object and multi-object tracking, point tracking, video object segmentation,
    person or vehicle re-identification, gait recognition.
72. **Human pose / body / hands**: human pose estimation, body shape and mesh recovery (SMPL), hand pose and hand
    reconstruction, human keypoints. Not: camera or object pose (84, 73).
    [v2] Human body, hand or face reconstruction goes to 72, 91 or 74, not 84.
73. **Object 6D pose estimation**: estimating or tracking the 6D (6-DoF) pose of objects, category- or instance-level. [v2] Rotation-only object pose
    counts.
74. **Face analysis**: faces: recognition, detection, landmarks, attributes, expressions, face generation or editing,
    talking heads, lip sync, portraits, face anti-spoofing.
75. **Image restoration / low-level vision**: super-resolution, denoising, deblurring, dehazing, deraining, low-light
    enhancement, inpainting, image quality assessment, image compression, HDR, and other low-level image processing.
76. **Video understanding / action recognition**: understanding video: action recognition, detection or anticipation,
    temporal grounding and localization, video QA or captioning, video LLMs, long-video understanding, video
    representation learning.
    [v2] Video QA, captioning, grounding and text-video retrieval are 76, not 77 or 79. Video quality assessment is 75.
77. **Visual QA / captioning / grounding**: visual question answering, image captioning, visual grounding and referring
    expressions, image-text retrieval, visual reasoning benchmarks, scene graphs, visual dialog.
    [v2] 3D visual grounding and 3D QA count for both 77 and 87.
78. **OCR / documents / charts**: text recognition and spotting, scene text, document understanding and layout, charts,
    tables in images, handwriting.
79. **Image retrieval / metric learning**: image retrieval, metric learning, hashing for retrieval, place recognition,
    sketch-based and composed image retrieval.
    [v2] Metric learning counts for any data, not only images. Person re-identification and person search are 71.
80. **Medical imaging**: medical images and their analysis (MRI, CT, X-ray, ultrasound, pathology slides, retinal
    images, endoscopy), medical image segmentation, reconstruction or generation, medical VLMs on images.
81. **Remote sensing / earth observation**: satellite, aerial or hyperspectral imagery, SAR, earth observation, land
    cover, geospatial imagery.
82. **Autonomous driving**: autonomous or self-driving vehicles and driving scenes: perception (incl. BEV, LiDAR),
    motion or trajectory prediction of traffic agents, planning and end-to-end driving, driving simulation and datasets.
    [v2] Perception on driving scenes counts even when "driving" is not written (LiDAR or multi-camera 3D detection,
    BEV, on KITTI, nuScenes or Waymo); generic segmentation or depth evaluated on Cityscapes or KITTI among other
    benchmarks does not.
83. **Event cameras / computational imaging**: event cameras, computational imaging and photography, lensless, single-
    photon, light-field, coded-aperture or polarization imaging, rolling shutter, snapshot compressive imaging.

## I. 3D, video and generation

84. **3D reconstruction / SfM / SLAM**: reconstructing 3D or 4D scenes, objects or surfaces; structure from motion,
    multi-view stereo, SLAM, visual localization and odometry, camera pose estimation and calibration, bundle adjustment.
85. **Depth / stereo / optical flow / matching**: depth estimation and completion, stereo matching, optical and scene
    flow, feature or keypoint detection and matching, dense correspondence. [v2] Image registration counts.
86. **Novel view synthesis / neural rendering**: novel view synthesis, neural, inverse and differentiable rendering,
    relighting.
87. **Point clouds / 3D understanding**: point clouds, LiDAR data, voxels or meshes analysed for 3D scene or shape
    understanding (3D detection, segmentation, classification, 3D visual grounding or QA, shape correspondence).
    [v2] 3D scene understanding from images or video, without point clouds, counts too.
88. **Image generation / editing**: generating, editing or translating images (text-to-image, personalized or
    subject-driven generation, controllable generation, image editing, style transfer, image-to-image translation).
89. **Video generation / editing**: generating or editing videos (text- or image-to-video, video diffusion, video editing,
    video generative world models).
90. **3D / 4D generation**: generating 3D or 4D content: objects, scenes, assets, textures, meshes (text- or image-to-3D,
    score distillation sampling, 4D generation).
91. **3D humans / avatars / motion**: 3D humans and avatars, clothed-human reconstruction, human motion capture,
    generation, prediction or retargeting, text-to-motion, human-object or human-scene interaction.
92. **World models / video prediction**: learned models that predict how an environment or video evolves (latent or
    pixel dynamics) for planning, policy learning, simulation or evaluation; video prediction; studies of whether models
    hold an internal world model. Not: "world" meaning real-world; open-world recognition.

## J. Applications

93. **AI for science: molecules / proteins / materials**: molecules, drugs, proteins, chemistry, crystals and materials,
    DNA and genomics, single-cell biology. [v2] LLM or benchmark work on chemistry or biology problems counts.
94. **Scientific ML / PDEs / physics**: solving or learning PDEs, neural operators, physics-informed networks, fluid
    dynamics, climate and weather modelling, physical simulation, other scientific computing with ML.
    [v2] LLM or benchmark work on physics or earth-science problems counts. A simulator counts when it is the
    contribution.
95. **Time series / forecasting**: time-series modelling, classification, imputation, anomaly detection and
    forecasting (weather, traffic, energy, finance, demand), spatio-temporal forecasting.
    [v2] Neuroscience or finance data count only when modelled as time series.
96. **Speech / audio / music**: speech, audio, sound and music: recognition, synthesis (TTS), enhancement, separation,
    speaker tasks, audio generation, audio-language models.
97. **NLP tasks**: language tasks: text question answering, machine translation, summarization, natural language
    inference, text classification, named entities and information extraction, sentiment, dialogue systems.
    [v2] Translation of sign language or speech counts with 96 where speech is involved. Multilingual and cross-lingual
    transfer counts.
98. **Robotics / embodied AI**: robots and embodied agents: manipulation, grasping, locomotion, navigation of embodied
    agents, vision-language-action models, sim-to-real, robot learning.
    [v2] VLA papers count for 98, 19 and 47; 20 when presented as a generalist or foundation robot model; 43 only when
    trained on demonstrations. Standard simulated control benchmarks (MuJoCo locomotion) alone do not count.
99. **Healthcare / clinical**: healthcare and medicine beyond images: clinical data and EHRs, patients, diseases,
    diagnosis, survival analysis, epidemiology, medical QA or reports, biomedical text.
100. **Tabular data**: learning on tabular data (tables of rows and columns). Not: the "tabular" setting of RL or MDPs.
     [v2] Table understanding or generation by LLMs counts; tables used only to report or store data do not.
101. **Combinatorial optimization / scheduling**: combinatorial optimization problems (TSP, vehicle routing, SAT, MILP,
     graph problems), scheduling, neural or learned solvers for them.
     [v2] Counts when solving such a problem is part of the contribution (incl. scheduling inside systems); a DP or ILP
     step used as a tool does not.
102. **Recommendation / information retrieval**: recommender systems, collaborative filtering, information retrieval,
     search engines, dense retrieval, learning to rank, query rewriting, click models. Not: "recommendations" meaning advice.
103. **Graph learning / knowledge graphs**: learning on graph-structured data: node or graph classification, link
     prediction, graph generation, graph matching, graph structure learning, knowledge graphs, hypergraphs, heterogeneous
     graphs. GNN papers on graph data count here too.
     [v2] Molecules or meshes count when the paper treats them as graphs (a GNN on molecular graphs). GNNs on graphs
     constructed from non-graph data (token or KNN graphs) are 9 only. Causal DAG learning is 64.

## K. Data and evaluation

104. **Datasets / benchmarks**: the paper contributes a new dataset or benchmark, or a large benchmarking study. Not:
     only evaluating on existing datasets; data collected for the paper's own experiments unless presented as a
     contribution (released, described as a dataset or benchmark).
     [v2] A large human evaluation study is not a benchmark unless presented as one.
105. **Synthetic data / data augmentation**: data augmentation methods, mixing methods (mixup, CutMix), generating
     synthetic data or samples for training or evaluation, self-generated training data.
     [v2] Synthetic toy data used only as a testbed does not count. Self-generated training data (self-training,
     self-evolution) counts, with 34 when it labels unlabelled data.
106. **Evaluation of LLMs / LLM-as-judge**: evaluating LLMs, VLMs or foundation models: benchmarks and evaluation
     methodology for them, LLM-as-a-judge, data contamination, leaderboards and arenas.
     [v2] Large text-to-image or video generators count as foundation models here. LLM-as-a-judge used only as one
     metric of the paper's own system does not count; studying or contributing the judging method does.
