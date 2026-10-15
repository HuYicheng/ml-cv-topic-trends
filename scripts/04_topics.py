"""Topic rules for the ML / CV study (title regexes, case-insensitive) and the title-only paper x topic matrix.

Same framework as the robotics study (~/robotics-trends/scripts/04_topics.py): a paper counts for a topic when the
title rule matches its title or the abstract rule matches its abstract (12_title_abstract.py); words that are generic
inside abstracts get stricter abstract rules (ABSTRACT_OVERRIDES). Topics were drawn from the most frequent and the
fastest-changing title n-grams of the six venues (06_ngrams.py) and checked by reading samples
(results/precision_review.md). Family codes: A architectures, B large models, C learning paradigms, D trustworthy ML,
F efficiency, G theory and foundations, H vision tasks, I 3D / video / generation, J applications, K data and evaluation;
"E term" = vocabulary terms tracked alongside, not topics.
Output: data/topic_matrix.csv (title only)
"""
import csv
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# non-generative senses of "diffusion" (physics, graphs, consensus, imaging, sampling dynamics)
DIFF_NOT = (r"(?<!anisotropic )(?<!label )(?<!advection[-– ])(?<!reaction[-– ])(?<!convection[-– ])(?<!thermal )(?<!heat )"
            r"(?<!gas )(?<!chemical )(?<!molecular )(?<!distributed )(?<!information )(?<!langevin )(?<!graph heat )"
            r"(?<!drift[- ])(?<!drift and )(?<!opinion )(?<!innovation )(?<!influence )(?<!graph neural )(?<!itô )(?<!ito )(?<!heat-)(?<!cascade )(?<!feature )diffusion"
            r"(?![- ](tensor|weighted|mri|coefficients?|equations?|terms?\b|convolutions?|maps?\b|wavelets?|kernels?|distance|operators?\b|covariances?|matri(?:x|ces)|theor\w*|patterns?\b)| of\b|[ -]?net\b)")

TOPICS = [
    # A. architectures and model families
    ("Diffusion models / score-based", "A architectures", DIFF_NOT + r"|score[- ]based (generative|models?|diffusion)|denoising (diffusion|score)|\bddpms?\b|\bddim\b|consistency models?"),
    ("Flow matching / normalizing flows", "A architectures", r"^(?!.*(?:gflownet|generative flow network)).*?flow[- ]matching|rectified flows?|normali[sz]ing flows?|continuous normali[sz]ing|stochastic interpolants?"),
    ("GANs", "A architectures", r"generative adversarial(?! (?:attacks?|perturbations?|transferability|examples?|patch(?:es)?|noise)\b)|\bgans?\b|gan-based|stylegan|\bcyclegan"),
    ("VAEs / vector-quantized models", "A architectures", r"variational auto-?encoders?|\bvaes?\b|vq-?vae|vq-?gan|vector[- ]quantiz"),
    ("Autoregressive models / tokenizers", "A architectures", r"(?<!vector )(?<!vector-)autoregressive(?![- ](?:moving|integrated|conditional heteroske?dastic|processes?\b|models? of order))|next[- ](token|scale) prediction|visual tokeni[sz]|image tokeni[sz]|video tokeni[sz]|discrete tokens?"),
    ("Transformers / attention", "A architectures", r"transformers?\b|self[- ]attention|vision transformers?|\bvits?\b|attention (mechanisms?|layers?|heads?)|cross[- ]attention|multi[- ]head attention"),
    ("State-space models / Mamba / linear attention", "A architectures", r"\b(?:structured|selective|diagonal|linear) state[- ]space|^(?=.*(?:state[- ]space|\bssms?\b)).*?\b(?:s4|s5)\b|\bmamba|state[- ]space (?:models?|layers?) \((?:ssms?|s4|s5)\)|(?<!bi)(?<!multi-)(?<!multi)(?<!x-)(?<!non)linear attention|\brwkv\b|(?<!non)(?<!non-)linear recurren|sub-?quadratic(?:ally)?[^.]{0,80}?(?:attention|transformers?|sequence|architectures?|alternatives?|language models?|recurren\w*|state[- ]space|ssms?|hyena)|(?:attention|transformers?|sequence|architectures?|alternatives?|language models?|recurren\w*|state[- ]space|ssms?|hyena)[^.]{0,80}?sub-?quadratic"),
    ("Mixture of experts", "A architectures", r"mixture[- ]of[- ]experts?|\bmoe\b|sparse(ly)?[- ]gated|expert (routing|parallel)"),
    ("Graph neural networks", "A architectures", r"graph neural|\bgnns?\b|\bgcns?\b|graph convolution|graph attention|(?<!approximate )(?<!attention )(?<!propagation-based )message[- ]passing(?! protocols?\b)|graph transformers?"),
    ("Convolutional networks", "A architectures", r"(?<!graph )(?<!graph-)(?<!edge )(?<!mesh )convolutional (neural )?networks?|\bcnns?\b|convnets?|(?<!graph )(?<!graph-)convolution(al)? (layers?|filters?|kernels?)"),
    ("Recurrent networks / LSTMs", "A architectures", r"recurrent neural|\brnns?\b|\blstms?\b|long short[- ]term memory|\bgrus?\b(?! ?-?\d)"),
    ("Neural ODEs / dynamical systems", "A architectures", r"neural (odes?|sdes?|ordinary differential|stochastic differential|controlled differential)|continuous[- ]time (neural|models?|dynamics)|koopman|dynamical systems?"),
    ("Equivariant / geometric deep learning", "A architectures", r"(?<!zero-)equivarian|geometric deep learning|group[- ]invariant|\bse\(3\)(?![- ]?(?:poses?\b|pose (?:setting|sequences?|estimation)|motion\b|b-spline|field\b|mappings?|multi-view critic|bijective|manifold steering|synchronization|end-effector))|\bso\(3\)(?![- ]?(?:poses?\b|pose (?:setting|sequences?|estimation)|motion\b|b-spline|field\b|mappings?|multi-view critic|bijective|manifold steering|synchronization|end-effector))|\be\(3\)(?![- ]?(?:poses?\b|pose (?:setting|sequences?|estimation)|motion\b|b-spline|field\b|mappings?|multi-view critic|bijective|manifold steering|synchronization|end-effector))|lie groups?"),
    ("Energy-based models", "A architectures", r"energy[- ]based (models?|learning|training|generative)|\bebms?\b"),
    ("Spiking / neuromorphic networks", "A architectures", r"(?<!neural )(?<!neuronal )spiking(?! (?:activity|data|rates?|statistics|recordings?|responses?|trains?|patterns?|variability|counts?)\b)|\bsnns?\b(?! ?\(siamese)|neuromorphic(?!-paris)(?![- ](?:event[- ]based |vision |visual |image |imaging |polarization )?(?:cameras?|sensors?|events?|data(?:sets?)?|imaging|vision|x-ray|signals?|streams?)\b)"),
    ("Neural fields / NeRF", "A architectures", r"\bnerfs?\b|neural radiance|radiance fields?|neural (implicit|fields?)|implicit neural (representations?|functions?)|signed distance (functions?|fields?)|occupancy networks?"),
    ("3D Gaussian splatting", "A architectures", r"gaussian[- ]splatting|\b3dgs\b|\b4dgs\b|gaussian splats?|(?<!lift, )(?<!lift-)(?<!softmax )(?<!many-to-many )(?<!surface )splat(?!planner)(?=.*gaussian)|(?<=gaussian)\w*splat"),
    # B. large and foundation models
    ("LLMs / language models", "B llm", r"large language models?|\bllms?\b|(?<!vision-)(?<!vision )(?<!visual-)(?<!visual )(?<!vision-and-)(?<!vision and )language models?|\bgpt[- ]?\d|chatgpt|\bllama|\bmistral\b|\bqwen|\bdeepseek"),
    ("Vision-language / multimodal LLMs", "B llm", r"vision[- ]language|visual[- ]language|vision[- ]and[- ]language (models?|pre-?training|pretrained|representations?|transformers?)|multi-?modal (large )?(language models?|llms?|reasoning models?|foundation models?)|omni-?modal (large )?(language models?|llms?|models?)|audio-visual (large )?(language models?|llms?)|\bmllms?\b|\blvlms?\b|\bvlms?\b|large multi-?modal models?|\blmms?\b|(?-i:\bCLIP\b)|image[- ]text (models?|pre-?training|alignment|pairs)|language[- ]image pre-?training"),
    ("Foundation models", "B llm", r"foundation models?|generalist models?|general[- ]purpose (models?|vision)|large[- ]scale pre-?trained (models?|vision)|segment anything|\bdinov[23]\b"),
    ("Agents / tool use", "B llm", r"\b(llm|lm|language|multimodal|web|gui|software|coding|computer[- ]use|tool[- ]using|mobile|llm-based|data[- ]science|deep[- ]research|research|scientific) agents?\b|language models? agents?|\((llms?|lms?|vlms?|mllms?)\) agents?|(llm|lm|vlm|mllm|language model)[- ](powered|driven|based) agents?|agentic(?! transformer)|(?<!emergent )(?<!surgical )tool[- ](use|usage|learning|calling|augmented)\b|function calling"),
    ("Reasoning / chain-of-thought", "B llm", r"chain[- ]of[- ]thought|\bcot\b|test[- ]time (compute|scaling)|\bthink(ing)?\b|^(?=.*\b(?:question|answer\w*|qa|vqa|language|llms?|vlms?|mllms?|lvlms?|knowledge|commonsense|logic\w*|benchmark\w*|inference|rules?|abstract|puzzles?|ratiocination|capabilit\w*|text|chain-of-thought|cot|math\w*|in-context|transformers?|instruction\w*|agents?|agentic|code|programs?|multimodal|vision-language)\b).*?reasoning|(logical|visual|physical|mathematical|math|multi-?step|commonsense|common-sense|spatial|temporal|abductive|deductive|inductive|compositional|analogical|symbolic|multi-?hop|geometric|multimodal|video|3d|chart|table|numerical) reasoning|reasoning (abilities|capabilit\w*|strategies|tasks?|models?|chains?|traces?|steps?|benchmarks?)"),
    ("Mathematical reasoning / theorem proving", "B llm", r"theorem prov(?:ing|ers?)\b|formal (?:proofs?|mathematics)|(?s:^(?=.*\b(?:llms?|language models?|theorems?|proofs?|provers?|lean|coq|isabelle|dafny|verus|code generation|code synthesis|programs?|verilog|software|math\w*)\b).*?formal verification)|\blean ?4|(?<=-)lean\b|\blean-\w|(?:(?<=\bin )|(?<=\bvia )|(?<=\bfor )|(?<=\bof )|(?<=\bthe )|(?<=\band )|(?<=\busing )|(?<=\bfrom )|(?<=\binto ))lean\b|\blean (?:theorems?|proofs?|provers?|code|verifier|compiler|repositor\w*|projects?|workbook|dojo|copilot|hammer|environment|kernel|tactics?|formali\w*|problems?|statements?|version)\b|\bmathlib|math(?:ematical)? (?:reasoning|word problems?|benchmarks?)|\bmath problems?|(?<!\ba )(?<!\bas )(?<!underlying )(?<!\bthat )(?<!\bthis )mathematical problems?|\bgsm8k\b|olympiad|auto-?formali[sz]|math(?:ematical)? problem[- ]solving"),
    ("Retrieval-augmented generation", "B llm", r"retrieval[- ]augmented|\brag\b|retrieval[- ]enhanced|knowledge[- ]augmented generation"),
    ("Alignment / RLHF / preference optimization", "B llm", r"\brlhf\b|human feedback|preference (?:optimi[sz]ation|alignment|tuning)|\bdpo\b|(?:llm|language model|value|human|ai) alignment|aligning (?:llms?|language models?|large|ai)|constitutional ai|superalignment|(?:^(?!.*\b(?:bandits?|regret|mdps?|pomdps?|model-based (?:rl|reinforcement)|imitation|world models?|offline (?:rl|reinforcement)|inverse reinforcement|q-learning|value iteration|elections?|social choice|mallows|rankings?|robot\w*|manipulation|locomotion)\b)|^(?=.*\b(?:llms?|language models?|rlhf|human feedback|human preferences?|alignment|aligned|aligning|diffusion|text-to-\w+|generative|chatbots?|assistants?|instruction\w*|mllms?|vlms?|lvlms?|lmms?|dpo|grpo|verifiers?|preference-based|pbrl|prompts?|responses?|chain-of-thought|multimodal|generation|gui|group relative|contrastive preference)\b)).*?(?:reward model(?:s|l?ing)?\b|preference (?:learning|data))"),
    ("Fine-tuning / PEFT / prompting", "B llm", r"fine[- ]?tun|parameter[- ]efficient|\bpeft\b|\blora\b|low[- ]rank adaptation|adapters?\b|prompt (tuning|learning|engineering|optimi[sz]ation)|visual prompt (?:tuning|learning)|soft prompts?|instruction[- ]tun|^(?=.*\b(?:llms?|language models?|multi-?modal|mllms?|vlms?|lvlms?|chat\w*|alignment|tuning)\b).*?instruction following"),
    ("In-context learning", "B llm", r"in[- ]context learning|\bicl\b(?!-)|in[- ]context (examples?|demonstrations?|abilit|regression)|few[- ]shot prompting|learns? in[- ]context|learning in[- ]context"),
    ("Hallucination / factuality", "B llm", r"factuality|factual (?:consistency|errors?|accuracy)|hallucinat\w*(?:\W+\w+){0,5}?\W+(?:spurious|non-existent|nonexistent|implausible|incorrect|unfaithful|faithfulness|erroneous\w*|fabricat\w*|inconsisten\w*|absent\b|undesir\w*|unwanted|unreliab\w*)|(?:spurious|non-existent|nonexistent|implausible|incorrect|unfaithful|faithfulness|erroneous\w*|fabricat\w*|inconsisten\w*|absent\b|undesir\w*|unwanted|unreliab\w*)(?:\W+\w+){0,4}?\W+hallucinat|(?:risk|prone|susceptible|suffers?|suffering|severe)\W+(?:\w+\W+){0,3}?hallucinat|(?<!face )(?<!feature )(?<!texture )(?<!style )(?<!depth )hallucinations\b|object hallucination|non-hallucinat|hallucination[- ](?:free|resistant|resilient|aware|detection|mitigation|evaluation|benchmark\w*|rates?|control|reduction|problems?|issues?)|(?:mitigat|reduc|alleviat|suppress|prevent|eliminat|avoid|curb|combat|address|tackl|minimi[sz])\w*\s+(?:\S+\s+){0,4}hallucinat|(?:video|contextual|model|object|cross-modal|multi-?modal|knowledge|factual|audio|textual|relation\w*|reasoning|spatial|temporal|action|stroke) hallucination|hallucinated (?:artifacts?|structures?|content|objects?|information|details?|regions?|responses?|answers?|facts?|text|tokens?|actions?)|collapse to hallucination|suffers? (?:from )?(?:\w+ )?hallucination|(?:why|do|does|can|models?|llms?|vlms?|mllms?|lvlms?) hallucinate|hallucinate (?:when|less|more|about|because)\b|\bor hallucination\b|^(?=.*(?:(?:(?:llms?|vlms?|mllms?|lvlms?|lmms?)\b|\b(?:chatgpt|gpt-?\d\w*|chatbots?|(?:large )?(?:vision[- ])?language models?|multi-?modal (?:large )?language|large multi-?modal|multi-?modal models?|unified models?|foundation models?|captioning|question[- ]answering|vqa|summar\w+|generated (?:text|answers?|responses?)|report generation|text generation|knowledge editing|factual\w*|dpo|rlhf|preference (?:optimi[sz]ation|alignment|data)|trustworth\w*|text-to-video|video-to-text|retrieval[- ]augmented|rag|refusal|jailbreak\w*)\b)|\btransformers?\b|\bgrounding\b|(?:mitigat|reduc|alleviat|detect|evaluat|benchmark|assess|measur|correct|suppress|curb|combat|address|tackl|eliminat|diagnos|quantif|understand|analy[sz]|investigat|interpret|explain|prevent)\w*\s+(?:\S+\s+){0,3}hallucinat)).*?(?<!face )(?<!feature )hallucinat"),
    ("Code generation / software", "B llm", r"(?<!compiler automatic )(?<!compiler )(?<!motion )(?<!latent )(?<!hash )code (?:generation|llms?|models?|completion|synthesis|editing|repair)(?! capabilit\w* of compilers)|program synthesis|programming (?:tasks|problems)|software engineering|swe-bench|code agents?"),
    ("Model merging / editing / unlearning", "B llm", r"model merging|\bmerging (models|weights|llms?)|task arithmetic|knowledge editing|model editing|^(?!.*debias).*?(?<!bias )(?<!noise-)(?<!noise )(?<!alleviating )unlearning(?! (?:the |of )?(?:biases|bias\b|spurious))|machine unlearn"),
    ("Long context", "B llm", r"long[- ]context|context (length|window)s?|length (generali[sz]ation|extrapolation)|long[- ]range arena|long[- ]sequence(?![- ](point|video|frames?|actions?|trajector\w*|motion|tasks?))"),
    ("Scaling laws / pretraining", "B llm", r"scaling (laws?|behaviou?r)|compute[- ]optimal|pre-?training (data|corpus|corpora|recipe)|(?<!streaming-)data mixtures?(?! models?)|emergent abilit"),
    # C. learning paradigms
    ("Self-supervised / contrastive learning", "C paradigms", r"self[- ]supervised|contrastive(?![- ](decoding|divergence|explanations?|preference|activation|examples?|analysis|reasoning|policy[- ]optimi[sz]ation))|representation learning|masked (image|autoencoders?|modell?ing|language modell?ing)|\bmae\b"),
    ("Semi- / weakly supervised & noisy labels", "C paradigms", r"semi[- ]supervised|weakly[- ]supervised|weak supervision|noisy labels?|label noise|pseudo[- ]label|positive[- ]unlabeled|partial labels?|learning with noisy"),
    ("Few-shot / zero-shot / open-vocabulary", "C paradigms", r"few[- ]shot|zero[- ]shot(?![- ](quantization|quantized|nas\b|neural architecture search|knowledge distillation))|open[- ]vocabulary|^(?=.*\b(detect|segment|recogni|classif|identif|semi-supervised|discover|count|understand|categor|class|label|unknown|novel|objectness|anomal|ground|retriev|metric learning|parsing|attribution|out-of-distribution|ood\b|open[- ]set|noisy|tracking|localization|perception|interpretation|uncertainty|recommendation|feature|learning\b|scene graph|prompt tuning|prototype|amodal|object extraction|semantic|relation|concept))(?:.*?)open[- ]world|open[- ]set|novel (classes|categories)|unseen (classes|categories)"),
    ("Domain adaptation / generalization / shift", "C paradigms", r"domain adaptation|domain generali[sz]ation|distribution(al)? shifts?|covariate shift|test[- ]time (adaptation|training)|domain shift|source[- ]free|out[- ]of[- ]domain|domain[- ]invariant"),
    ("Continual / incremental learning", "C paradigms", r"continual learning|lifelong learning|incremental learning|class[- ]incremental|catastrophic forgetting|task[- ]incremental"),
    ("Meta-learning", "C paradigms", r"meta[- ]learn|learning[- ]to[- ]learn\b(?! (representations?|features?|embeddings?|policies|a |the ))|meta[- ]train|model[- ]agnostic meta|\bmaml\b"),
    ("Federated learning", "C paradigms", r"federated"),
    ("Knowledge distillation", "C paradigms", r"distill(ation|ing|ed)?\b|teacher[- ]student|dataset distillation"),
    ("Active learning / data selection", "C paradigms", r"\bactive learning|data selection|coresets?\b|data pruning|data curation|sample selection|data valuation"),
    ("Reinforcement learning", "C paradigms", r"reinforcement learning|\b(deep |online |inverse |multi-agent |model-based )?rl\b|policy (optimi[sz]ation|gradients?|iteration)|actor[- ]critic|q[- ]learning|\bppo\b|markov decision|\bmdps?\b|\bgrpo\b"),
    ("Offline RL / imitation learning", "C paradigms", r"offline (rl|reinforcement)|batch rl|imitation learning|behaviou?r(al)? cloning|inverse reinforcement|learning from demonstrations?|\birl\b"),
    ("Multi-agent learning / game theory", "C paradigms", r"multi[- ]agent (reinforcement|rl|learning|cooperation|coordination)|\bmarl\b|game[- ]theor|nash equilibri|equilibri(um|a) (computation|learning|finding)|mean[- ]field games?|mechanism design|auctions?\b|(markov|zero-sum|general-sum|two-player|multi-player|extensive[- ]form|normal[- ]form|potential|stochastic|cooperative|non-cooperative|bayesian|stackelberg|matrix|congestion|repeated|differential|imperfect[- ]information|sequential) games?\b"),
    ("Bandits / online learning", "C paradigms", r"bandits?\b|online learning|regret (bounds?|minimi[sz]ation)|online convex|thompson sampling|\bucb\b|best[- ]arm"),
    ("Multi-task / transfer learning", "C paradigms", r"multi[- ]task|transfer learning|^(?!.*\b(adversarial|attacks?|perturbations?|jailbreaks?|backdoors?|black-?box)\b).*?transferab|auxiliary tasks?"),
    ("Multimodal learning / cross-modal", "C paradigms", r"multi-?modal(?! (data )?(distributions?|posteriors?|densit\w*|landscapes?|functions?|objectives?|optimi[sz]ation|behaviou?rs?|action distributions?|targets?|trajector\w*|motion (prediction|forecasting)|futures?|predictions?|outputs?|solutions?|modes?|prototypes?)\b)|multi-modality|cross-?modal|audio-?visual|omni-?modal|modalit(y|ies)\b|(sensor|modality|multi-sensor|rgb-?d|rgb-?t) fusion"),
    ("Clustering / unsupervised learning", "C paradigms", r"clustering|^(?!.*\bdomain[- ]adapt(ation|ive)\b).*?unsupervised(?!(?:[- ][a-z]+){0,3}[- ](domain[- ]adaptation|adaptation|personali[sz]ation)\b)|dimensionality reduction|manifold learning|k[- ]means"),
    # D. trustworthy ML
    ("Adversarial robustness / attacks", "D trustworthy", r"adversarial (examples?|attacks?|robustness|perturbations?|defen[cs]es?|vulnerabilit)|certified robust|robustness certificat|randomi[sz]ed smoothing|evasion attacks?|adversarially robust|(?<!domain[- ])(?<!non-)adversarial training"),
    ("Backdoor / data poisoning", "D trustworthy", r"back-?door(?![- ](adjust|path|criteri|confound|deconfound|intervention)|,? (and |or )?front-?door)|trojan|data poisoning|poisoning attacks?|poisoned"),
    ("Privacy", "D trustworthy", r"privacy|differential(ly)? privat|\bdp-sgd\b|membership inference|(data|training data) (extraction|reconstruction) attacks?|model inversion|private (learning|training|inference|data)"),
    ("Fairness / bias", "D trustworthy", r"fairness|\bfair (classif|represent|learning|ranking|machine|ml|allocation|clustering|regression)|algorithmic fairness|demographic (parity|groups?|bias)|social bias|gender bias|racial bias|stereotyp|protected (attributes?|groups?)|equal(i[sz]ed)? (odds|opportunity)|group robustness|spurious correlation"),
    ("Interpretability / explainability", "D trustworthy", r"interpretab|explainab(?!leml)|\bxai\b|saliency(?![- ](detection|detector|prediction|predictor|object|ranking|model))|feature attribution|attribution (methods?|maps?)|mechanistic|sparse auto-?encoders?|concept bottleneck|counterfactual explanations?|probing (classifiers?|tasks?|experiments?|analysis)|(linear|structural) probes?"),
    ("Uncertainty / calibration / conformal", "D trustworthy", r"uncertainty (quantification|estimation|calibration|aware)|uncertainty-aware|conformal|predictive uncertainty|epistemic|aleatoric|(confidence|model|probability) calibration|(?<!camera )(?<!sensor )(?<!extrinsic )calibration errors?|well[- ]calibrated|miscalibrat"),
    ("OOD / anomaly detection", "D trustworthy", r"(out[- ]of[- ]distribution|\bood\b)(\W+\w+){0,3}?\W+(detect\w*|identif\w*|reject\w*|scor\w*|discriminat\w*|flag\w*|filter\w*|separat\w*|segment\w*|locali[sz]\w*|abstain\w*|abstention|guard\w*|safeguard\w*|monitor\w*)|(detect\w*|identif\w*|reject\w*|scor\w*|discriminat\w*|flag\w*|filter\w*|separat\w*|segment\w*|locali[sz]\w*|abstain\w*|abstention|guard\w*|safeguard\w*|monitor\w*)(\W+\w+){0,3}?\W+(out[- ]of[- ]distribution|\bood\b)|anomal(y|ies|ous)|novelty detection|outlier detection|open[- ]set recognition"),
    ("LLM safety / jailbreak", "D trustworthy", r"jailbreak|red[- ]teaming|harmful (content|outputs?|requests?|prompts?)|prompt injection|refusal|guardrails?|(llm|ai|model) safety(?![- ]gym)|safety (alignment|of (llms?|language|large))|unsafe (content|outputs?|prompts?)"),
    ("Watermarking / deepfakes / forensics", "D trustworthy", r"watermark|deepfake|forgery|forensic|(ai|machine)[- ]generated (images?|text|content|videos?) detection|detect(ing|ion of)? (ai|machine)[- ]generated|copyright|(model|content|image|video|audio|text|llm|gan|generator|generative|network|deepfake|camera|device|ownership)s?[- ]([\w-]+[- ])?fingerprint\w*|fingerprint\w* ([\w-]+ ){0,2}(of|for|to|in) (the )?([\w-]+ ){0,2}(models?|llms?|generat\w*|images?|content|gans?|networks?|ownership|attribut\w*|ip)\b|fingerprint\w* (llms?|models?|large language|generative|gans?|diffusion|deepfakes?)|(content|image|video|media|text|output)[- ]provenance|provenance (of|for) (the )?(ai|machine|generated|synthetic|images?|content|media|text|outputs?|videos?)|provenance (tracking|tracing|verification|detection|attribution)"),
    # F. efficiency
    ("Quantization / pruning / compression", "F efficiency", r"(?<!vector[- ])(?<!vector-)(?<!de)(?<!scalar )(?<!product )(?<!residual )(?<!color )(?<!semantic )(?<!lookup-free )quantiz|(?<!branch-and-bound )(?<!tree )(?<!node )(?<!search )(?<!beam )(?<!candidate )(?<!path )(?<!data )(?<!dataset )(?<!ensemble )pruning|sparsif|low[- ]bit|binar(y|ized) neural|model compression|compress(ing|ion of) (the )?(models?|networks?|llms?|transformers?)|lottery ticket"),
    ("Efficient inference / sampling", "F efficiency", r"efficient (inference|serving|decoding|attention|generation|sampling)|speculative decoding|kv[- ]cache|token (pruning|merging|reduction|compression|dropping)|early[- ]exit|accelerat(e|ing|ion) (inference|sampling|diffusion|generation|decoding)(?![\w -]{0,40}\btraining\b(?!-free))|inference (acceleration|efficiency|latency|cost)|few[- ]step (generation|sampling|diffusion)|one[- ]step (generation|diffusion|sampling)"),
    ("Neural architecture search / AutoML", "F efficiency", r"architecture search|\bnas\b|automl|hyper-?parameter (optimi[sz]ation|search)"),
    # G. theory and foundations
    ("Learning theory / generalization", "G theory", r"generali[sz]ation (bounds?|error|gap|theory|guarantees?)|pac[- ]bayes|\bpac\b|sample complexity|vc[- ]dimension|rademacher|statistical learning theory|benign overfitting|double descent|implicit (bias|regularization)|neural tangent|\bntk\b|minimax (rates?|optimal)|(?<!evidence )(?<!variational )lower bounds?(?! (on|of|for) (the )?(log-?likelihood|marginal likelihood|mutual information|evidence))|\blearnability|provabl"),
    ("Optimization", "G theory", r"stochastic gradient|\bsgd\b|convergence (rates?|analysis|guarantees?)|non-?convex|convex (optimi[sz]ation|problems?|functions?)|optimi[sz]ers?\b|\badam(w)?\b|first[- ]order (methods?|optimi)|gradient descent|second[- ]order (methods?|optimi)|learning rate|bilevel|min-?max (optimi[sz]ation|problems?)|variational inequalit|zeroth[- ]order|saddle[- ]point"),
    ("Bayesian & probabilistic inference", "G theory", r"bayesian|variational inference|\bmcmc\b|markov chain monte carlo|langevin|posterior (sampling|inference|approximation|distribution)|(?<!diffusion )probabilistic (models?|programming|inference|circuits?)|sequential monte carlo|importance sampling|stein variational|expectation[- ]maximi[sz]ation"),
    ("Causal inference / discovery", "G theory", r"causal (inference|discovery|effects?|structure|models?|representation|reasoning|graphs?|learning)|causality|treatment effects?|confound|instrumental variables?|interventional|counterfactual (inference|reasoning|prediction|estimation|outcomes?)"),
    ("Optimal transport", "G theory", r"optimal transport|wasserstein|sinkhorn|earth mover|schr[oö]dinger bridge"),
    ("Kernel methods / Gaussian processes", "G theory", r"kernel (methods?|ridge|regression|learning|machines?|mean embedding)|reproducing kernel(?! particle)|\brkhs\b|support vector|\bsvms?\b|random fourier features|random features?(?= (for|of|approximation|regression|kernel|model))|gaussian process"),
    ("Statistical estimation / high-dimensional statistics", "G theory", r"high[- ]dimensional (statistics|regression|estimation|inference)|sparse (regression|recovery|estimation|linear(?! (systems?|solvers?|layers?|attention|algebra|encoding|transforms?|surrogates?|rnns?|concept)))|\blasso\b|matrix (completion|factori[sz]ation|sensing|estimation)|tensor (completion|factori[sz]ation)|robust (statistics|estimation|mean estimation|regression)|hypothesis test|two[- ]sample test|(?<!crowd )(?<!object )density estimation|covariance estimation|mean estimation"),
    # H. vision tasks
    ("Image classification / recognition", "H vision", r"image (classification|recognition)|visual recognition|fine[- ]grained (visual )?(classification|recognition|categori[sz]ation)|long[- ]tail(ed)? (recognition|classification|learning|distributions?)|imagenet"),
    ("Object detection", "H vision", r"(?<!salient )(?<!camouflaged )(?<!co-salient )object detect|detection transformers?|\bdetr\b|object detectors?|(2d|3d) (object )?detection|open[- ]vocabulary (object )?detection|oriented (object )?detection|\byolo|human[- ]object interaction (detection|recognition)|\bhoi detection"),
    ("Segmentation", "H vision", r"(?<!action )(?<!temporal )(?<!word )(?<!speaker )(?<!model )(?<!syllabic )(?<!syllable )segmentation|segment anything|(?<!cmu )(?<!cmu-)panoptic(?![- ]studio)|matting|segmenter|(semantic|instance) masks?"),
    ("Tracking / re-identification", "H vision", r"(object|visual|multi[- ]object|point|single[- ]object|pedestrian) tracking|\btrackers?\b|video object segmentation|re[- ]?identification|\bre-?id\b|person retrieval|gait recognition"),
    ("Human pose / body / hands", "H vision", r"^(?!.*\b(cameras?|rays|objects?|vehicles?|grasp|robot|category-level|instance-level|6d|6-?dof|nerf|radiance|splatting|gaussians?|view synthesis|sparse[- ]views?|sfm|structure[- ]from[- ]motion|bundle adjustment|spacecraft|satellite|articulated|assembly|registration|localization)\b).*?pose estimation|human pose|body (pose|shape|mesh|models?)|\bsmpl|hand (pose|reconstruction|mesh|tracking)|hands?[- ](and[- ])?objects? interactions?|(human|body|whole-body|animal|hand) keypoints?|human mesh"),
    ("Object 6D pose estimation", "H vision", r"(object|category-level|instance-level) pose (estimation|tracking)|(?<!non-rigid )(?<!geocentric )object pose|(6d|6-?dof) pose (estimation|tracking)|6d pose|6-?dof (object )?pose"),
    ("Face analysis", "H vision", r"(?-i:(?!FACES?\b))(?<!hugging )\bfaces?\b(?! (of|the|a|an|challenges?|difficult|significant|limitations|issues|problems|interaction graph|selection)\b)(?![- ]to[- ]face)|\bfacial|talking[- ](heads?|faces?)|lip[- ]sync|portrait"),
    ("Image restoration / low-level vision", "H vision", r"(?<!label )(?<!audio )(?<!speech )super[- ]resolution(?! (for|of) (time series|climate|audio|speech))|(image|video|real|raw|burst|blind|low-light|self-supervised|real-world|single-image) denois|denois(e|ing) (of )?(images?|videos?|raw)|deblur|dehaz|derain|low[- ]light|image (restoration|enhancement|quality assessment|compression)|inpainting|demosaic|\bhdr\b|video (restoration|super-resolution|frame interpolation|enhancement|compression)|blind (image|restoration|super)"),
    ("Video understanding / action recognition", "H vision", r"video (understanding|recognition|classification|question answering|grounding|captioning|retrieval|representation|summari[sz]ation|llms?|language models?|reasoning)|\baction (recognition|detection|localization|segmentation|anticipation)|temporal (action|grounding|localization|sentence grounding)|activity recognition|long[- ]form video|video[- ]language|egocentric"),
    ("Visual QA / captioning / grounding", "H vision", r"visual question answering|\bvqa\b|image captioning|captioning|visual grounding|referring (expressions?|segmentation)|phrase grounding|image[- ]text retrieval|cross[- ]modal retrieval|text[- ]to[- ](image|video) retrieval|visual reasoning|scene graphs?|visual dialog"),
    ("OCR / documents / charts", "H vision", r"\bocr\b|optical character(?!istic)|(?<!ai )(?<!ai-generated )(?<!machine-generated )(?<!llm-generated )(?<!generated )(?<!erroneous )text (recognition|detection|spotting)|scene text(?!ur)|document (understanding|parsing|images?|layout|visual|ai|vqa)|(?<!uv )(?<!minimal )(?<!coordinate )\bcharts?\b|table (understanding|recognition|parsing)|handwrit|layout analysis"),
    ("Image retrieval / metric learning", "H vision", r"image retrieval|\bmetric learning|(deep|cross-modal|image|semantic|binary|supervised|unsupervised|video) hashing|content[- ]based retrieval|instance retrieval|place recognition|sketch[- ]based( fine[- ]grained)?( image| 3d shape| 3d| shape| video| scene| object)? retrieval|\bsbir\b|composed image retrieval"),
    ("Medical imaging", "H vision", r"medical (imag|image|vision|segmentation|visual)|\bmri\b|(?-i:\bCT\b)|computed tomography|x-?rays?\b|radiolog|histopatholog|(computational|digital|clinical) patholog|patholog(y|ists?)\b|pathological (images?|regions?|tissues?|slides?|lesions?|features?|areas?)|whole[- ]slide|ultrasound|\bfmri\b|retinal(?! (ganglion|mosaics?|periphery|representation manifold|circuits?|neurons?|cells?|coding|responses?|computations?))|fundus|dermatolog|endoscop|brain (imaging|tumou?r|mri)|mammogra|clinical imag"),
    ("Remote sensing / earth observation", "H vision", r"remote sensing|satellite (images?|imagery|data)|aerial (images?|imagery)|earth observation|hyperspectral|synthetic aperture radar|geospatial|land (cover|use)"),
    ("Autonomous driving", "H vision", r"autonomous driving|self[- ]driving|(?<!avatar )(?<!signal )\bdriving\b(?! (engine|human|cortical|videos?\b))|autonomous vehicles?|bird.?s[- ]eye[- ]view|\bbev\b|(?<!human )(?<!3d human )(?<!body )motion forecasting|trajectory (prediction|forecasting)|traffic (scenes?|scenarios?|participants?|simulation|agents?)|nuscenes|waymo|\bcarla\b|lane detection|occupancy (prediction|forecasting)"),
    ("Event cameras / computational imaging", "H vision", r"event[- ](based (vision|cameras?|data|sensors?)|cameras?|streams?)|computational (imaging|photography)|lensless|single[- ]photon|(?<!neural )(?<!surface )(?<!hdr )(?<!5d )(?<!deformable )light fields?(?! (transformers?|networks?|estimation))|polari[sz]ation (imaging|cues)|coded (aperture|exposure)|rolling shutter|snapshot compressive"),
    # I. 3D, video and generation
    ("3D reconstruction / SfM / SLAM", "I 3d", r"3d reconstruction|4d reconstruction|(multi[- ]view|surface|scene|mesh|object) reconstruction|structure[- ]from[- ]motion|\bsfm\b|\bslam\b|bundle adjustment|visual (localization|odometry)|camera (pose|relocali[sz]ation|calibration)|multi[- ]view stereo|\bmvs\b"),
    ("Depth / stereo / optical flow / matching", "I 3d", r"depth (estimation|prediction|completion|maps?)|monocular depth|stereo (matching|depth|vision)|optical flow|scene flow|feature matching|image matching|(?<!human )keypoint (matching|description|detection)|dense correspondences?|(?<!global and )(?<!global-)(?<!global & )(?<!global-to-)(?<!non-)local features?\b(?! (selection|swapping|enhancement|fusion|alignment|coupling))"),
    ("Novel view synthesis / neural rendering", "I 3d", r"novel[- ]view synthesis|view synthesis|neural rendering|inverse rendering|relighting|differentiable rendering|(scene|appearance) rendering"),
    ("Point clouds / 3D understanding", "I 3d", r"point clouds?|lidar|3d (object detection|semantic|scene understanding|shape analysis|understanding|perception|vision[- ]language|grounding|visual grounding|question answering|points?(?![- ](track|trajector)))|voxel|occupancy (grids?|networks?)|mesh (segmentation|analysis)|shape (correspondence|matching|analysis)"),
    ("Image generation / editing", "I 3d", r"text[- ]to[- ]image|(?<!view )image (generation|synthesis|editing|manipulation|translation|outpainting)|\b(controllable|conditional|personali[sz]ed|subject[- ]driven) image generation|\b(personali[sz]ed|subject[- ]driven) generation|(?<!text )(?<!textual )(?<!language )(?<!sentence )(?<!formality )(?<!sentiment )(?<!voice )(?<!speech )(?<!prosody )(?<!music )(?<!motion )(?<!timbre )(?<!audio )(?<!soundscape )(?<!dance )(?<!gait )(?<!writing )(?<!linguistic )(style transfer|stylization)(?![^.]{0,80}\b(sentences?|formality|sentiment|nlp|speech|voices?|music|audio|motions?|gestures?|prosody)\b)|image[- ]to[- ]image(?! (architectures?|networks?|segmentation|registration))"),
    ("Video generation / editing", "I 3d", r"video (generation|synthesis|editing|diffusion|prediction|generative|world models?)|text[- ]to[- ]video(?![- ]((temporal )?(retrieval|matching|search|grounding|moment)|re-?ranking))|(?<!from )image[- ]to[- ]video(?![- ](transfer|adaptation|knowledge|re-?identification|re-?id\b|retrieval|matching|learning|distillation|person|model adaptation|domain|representation|transformers?|auto-?augment\w*|setting|attacks?|\(i2v\) attack))|video generative|animat(e|ion|ing) (images?|characters?|portraits?)"),
    ("3D / 4D generation", "I 3d", r"3d (generation|generative|content creation|assets?|object generation|shape generation|scene generation)|text[- ]to[- ]3d|image[- ]to[- ]3d(?! correspondences?)|4d (generation|content|(scenes?|objects?|assets?|avatars?|dynamics|videos?|worlds?|motions?) (generation|synthesis|creation))|(text|image|video)[- ]to[- ]4d|(creat\w*|generat\w*|synthesi[sz]\w*) ([\w-]+ ){0,6}?4d (scenes?|gaussians?|objects?|content|assets?|avatars?|dynamics|videos?|motions?)|mesh generation|score distillation sampling|variational score distillation"),
    ("3D humans / avatars / motion", "I 3d", r"avatars?|human motion (generation|synthesis|prediction|capture)|\b(?<!video )(?<!scene )motion (generation|synthesis)(?! (in|for|of) (video|camera))|\bmotion (capture|diffusion|retargeting)|text[- ]to[- ]motion|human[- ]object interactions?\b(?! (\(hois?\) )?(detection|recognition))|clothed humans?|digital humans?|3d humans?|human (reconstruction|digitization)|gesture generation|dance generation"),
    ("World models / video prediction", "I 3d", r"(?<!real-)(?<!real )(?<!open-)(?<!open )world[- ]?model(s|l?ing)?\b|video prediction"),
    # J. applications
    ("AI for science: molecules / proteins / materials", "J applications", r"molecul|(?<!carbohydrates, )protein|\bdrugs?\b|(?<!histo)chemi(cal|stry)|materials? (discovery|design|science|property prediction)|predict\w* (\w+ )?materials? propert|crystal|interatomic|single[- ]cell(?! (activity|recordings?|responses?|stereotypes?|neurons?|firing|electrophysiolog))|(?<!-)\bdna\b|(?<!visual )(?<!action )(?<!cancer )\bgenom|(?<!-)\brna\b|antibod|enzyme|peptide|ligand|docking|retrosynthesis"),
    ("Scientific ML / PDEs / physics", "J applications", r"partial differential|\bpdes?\b|neural operators?|physics[- ]informed|\bpinns?\b|fluid (dynamics|simulation|flows?)|navier|climate[- ](models?|modell?ing|data|datasets?|forecast\w*|predictions?|projections?|emulat\w*|simulat\w*|science|dynamics|variables?|analysis|downscal\w*|extremes?|systems?|fields?|reanalysis)|\bclimax\b|sub-?seasonal|weather and climate|climate and weather|(?<!adverse )(?<!bad )(?<!diverse )(?<!extreme )(?<!all-)(?<!all )weather (forecast|prediction|models?|data)|numerical weather|turbulen|scientific (machine learning|discovery|computing)|(?<!cyber-)(?<!cyber )physical (systems?|simulation)|computational fluid"),
    ("Time series / forecasting", "J applications", r"time[- ]series|(weather|traffic|demand|sales|energy|load|stock|epidemic|financial|probabilistic|long-term|multivariate) forecasting|spatio[- ]temporal (forecasting|prediction)"),
    ("Speech / audio / music", "J applications", r"(?<!part-of-)(?<!part of )(?<!hate )speech|audio|\bsounds?\b|music|speaker|\bvoices?\b|acoustic|\basr\b|text[- ]to[- ]speech|\btts\b|spoken"),
    ("NLP tasks", "J applications", r"^(?!.*\b(visual|video|image|chart|vision|multimodal|3d|scene|egocentric|vlms?|mllms?|text-to-image)\b).*?question answering|machine translation|(?<!video )(?<!data )summari[sz]ation|natural language (understanding|inference|generation|processing)|\bnlp\b|text classification|named entity|sentiment|(?<!talking )dialog(ue)? (systems?|generation|models?)|information extraction|translation (quality|models?)"),
    ("Robotics / embodied AI", "J applications", r"\brobot|embodied|robotic|vision[- ]language[- ]action|\bvlas?\b|locomotion|(?<!image )(?<!photo )(?<!face )(?<!text )manipulation (tasks?|policies|skills?)|(visual|vision[- ]language|object[- ]goal) navigation|sim[- ]to[- ]real|(?<!human )(?<!whole-body )(?<!full-body )(?<!hand-object )grasp(?!\s?:)(?! (the|any|video)\b)"),
    ("Healthcare / clinical", "J applications", r"clinical|healthcare|health care|\behrs?\b|electronic health|patients?\b|diseases?\b|mortality|survival analysis|epidemi(?!c (cascades?|learning))|biomedical|medical (records?|questions?|diagnosis|knowledge|qa|reports?)"),
    ("Tabular data", "J applications", r"(?<!horizon )(?<!episodic )(?<!finite )tabular(?![- ](setting|settings|case|cases|mdps?\b|markov|rl\b|reinforcement|environments?|q-learning|q-functions?|nas\b|polic(y|ies)|value functions?|games?\b|pomdps?|bandits?|regime|search|partially observable|domains?|atari))(?!,? (and|or)? ?(linear|function approximation|low-rank|continuous|atari|mujoco)\b)"),
    ("Combinatorial optimization / scheduling", "J applications", r"combinatorial (optimi[sz]ation|problems?)|traveling sales(man|person)|(?<!\()\btsp\b(?!\))(?!-[a-z])|vehicle routing|(?<!\bnoise )(?<!\bcovariance )(?<!\btoken )(?<!\breward )(?<!\bcurriculum )(?<!\bdata )(?<!\brate )(?<!\blr )(?<!\bdecay )(?<!\bdropout )(?<!\bentropy )(?<!\btimestep )(?<!\btime-step )(?<!\bstep )(?<!\bguidance )(?<!\bquantization )(?<!\bpatch )(?<!\btile )(?<!\bblock )(?<!\bdilated )(?<!\bvisual )(?<!\bsample )(?<!\bsize )(?<!\bwindow )(?<!\bcyclical )(?<!\bdiscount )(?<!\btemperature )(?<!\bflow )(?<!\bproximity )(?<!\bweight )(?<!\brehearsal )(?<!\bcognitive )(?<!\bgradient-aware )(?<!\brelaxed )(?<!\bupdate )(?<!\bfusion )(?<!\bview )(?<!\bthreshold )(?<!\bparameter )(?<!\boptimization )(?<!\blinear )(?<!\bgranularity )(?<!\bfast-slow )(?<!\bdual-timestep )(?<!\bexploration )(?<!\bregularization )(?<!\bkeyframe )(?<!\bmask )(?<!\bmasking )(?<!\bsampling )(?<!\bstability )(?<!\bloss )(?<!\baugmentation )(?<!\bmulti-objective )(?<!\btraining )(?<!\bdifficulty )(?<!\bramp )(?<!\btime-)(?<!\bepsilon-)(?<!\bmanual )scheduling(?! (of (the )?(memory|data|causal|learning|noise|guidance|classifier|weights?|thoughts)|weight|thoughts|your llm|strateg))|mixed[- ]integer|\bmilp\b|boolean satisfiability|\bsat solv|integer programming|neural solvers?"),
    ("Recommendation / information retrieval", "J applications", r"recommend(ation|er|ers|ations)\b|recommender|^(?!.*\b(super-?resol\w*|denois\w*|restor\w*|spike|image quality|iqa)\b).*?collaborative filtering|information retrieval|search engines?|dense retrieval|^(?!.*\b(super-?resol\w*|denois\w*|restor\w*|spike|image quality|iqa)\b).*?learning to rank|query (expansion|rewriting)|click[- ]through|retrievers?\b"),
    ("Graph learning / knowledge graphs", "J applications", r"knowledge graphs?|link prediction|node classification|(?<!scene )graph (generation|matching|clustering|representation|structure learning|classification|learning)|heterogeneous graphs?|hypergraph|molecular graphs?"),
    # K. data and evaluation
    ("Datasets / benchmarks", "K data", r"(?<!offline )(?<!imbalanced )(?<!private )(?<!decentralized )(?<!clean )(?<!multi-)(?<!multiple )(?<!cross-)(?<!small )(?<!noisy )(?<!context )\bdataset\b(?! (shift|distillation|condensation|pruning))|(?<!offline )(?<!imbalanced )(?<!private )(?<!decentralized )(?<!clean )(?<!multi-)(?<!multiple )(?<!cross-)(?<!small )(?<!noisy )(?<!context )(?<!image )(?<!diverse )(?<!unannotated )(?<!unlabeled )(?<!large )(?<!large-scale )(?<!synthetic )\bdatasets\b(?! (shift|distillation|condensation|pruning))|\bbenchmarks?\b|benchmarking"),
    ("Synthetic data / data augmentation", "K data", r"data augmentation|synthetic (data|images?|datasets?|samples?)|augmentation (strategies|polic|methods?)|\bmixup\b|cutmix|data generation|generated data|self[- ]generated (data|training)"),
    ("Evaluation of LLMs / LLM-as-judge", "K data", r"llm[- ]as[- ]a[- ]judge|evaluat(ing|ion of|e) (large language|llms?|language models?|foundation models?|multimodal(?!ity)|vlms?)|\bllm evaluation|benchmark(ing)? (llms?|large language|multimodal(?!ity))|data contamination|(llm|chatbot|model) (leaderboards?|arena)"),
    # E. vocabulary terms (not topics)
    ("[term] efficient", "E term", r"efficien"),
    ("[term] robust", "E term", r"\brobust"),
    ("[term] unified / universal / general-purpose", "E term", r"unified|universal|general[- ]purpose|generalist|one model"),
    ("[term] training-free / zero-shot", "E term", r"training[- ]free|tuning[- ]free|plug[- ]and[- ]play|zero[- ]shot"),
    ("[term] provable / theoretical", "E term", r"provabl|theoretical|theorem|guarantee"),
    ("[term] real-world", "E term", r"real[- ]world"),
]
COMPILED = [(n, f, re.compile(p, re.I)) for n, f, p in TOPICS]

# Abstract versions of rules whose title words are generic inside abstracts
ABSTRACT_OVERRIDES = {
    "Datasets / benchmarks": r"\bwe (introduce|present|release|propose|collect|construct|build|create|curate|open-source)\w* (a |an |the |our )?((new|novel|large-scale|first|open-source|public|publicly available|real-world|multi-?modal|comprehensive|diverse|challenging|synthetic|annotated|unified|high-quality)[ ,]+)*(dataset|benchmark)s?\b|(this|our) (paper|work|letter|article) (introduces|presents|proposes|releases) (a |an |the )?([\w-]+ ){0,4}(dataset|benchmark)\b|the proposed (dataset|benchmark)\b|\b(new|novel|first) (large-scale |public |open-source )?(dataset|benchmark) (for|of|that|which|named|called|comprising|containing)\b",
    "Transformers / attention": r"transformers?\b|self[- ]attention|vision transformers?|\bvits?\b|attention (mechanisms?|layers?|heads?|maps?|weights?|modules?|blocks?|scores?)|cross[- ]attention|multi[- ]head attention",
    "Reasoning / chain-of-thought": r"chain[- ]of[- ]thought|\bcot\b|reasoning (capabilit|abilit|chains?|traces?|steps?|process|tasks?|benchmarks?|paths?|performance|skills?)|(?<!counterfactual )reasoning models?|(?:mathematical|math)(?<!by strict mathematical)(?<!by rigorous mathematical)(?<!by mathematical) reasoning|(logical|commonsense|common-sense|spatial|visual|multi-?step|complex|deductive|inductive|abductive|temporal|symbolic|multi-?hop|compositional|analogical|physical|geometric|multimodal|video) reasoning(?! (?:modules?|networks?|layers?|blocks?|units?|branch\w*|graphs?)\b)|(llms?|vlms?|mllms?|language models?|models?)'?s? reasoning|reasoning[- ](intensive|heavy|enhanced|oriented)|test[- ]time (compute|scaling)|long[- ]cot|large reasoning models?",
    "Agents / tool use": r"\b(llm|lm|language|multimodal|web|gui|software|coding|computer[- ]use|tool[- ]using|llm-based|language model|data[- ]science|deep[- ]research|research|scientific) agents?\b|language models? agents?|\((llms?|lms?|vlms?|mllms?)\) agents?|(llm|lm|vlm|mllm|language model)[- ](powered|driven|based) agents?|agentic(?! transformer)|(?<!emergent )(?<!surgical )(?<!robotic )(?<!robot )tool[- ](use|usage|learning|calling|augmented)\b|function calling",
    "Fine-tuning / PEFT / prompting": r"(?<!without )(?<!without any )(?<!without further )(?<!without additional )(?<!no )(?<!need for )fine[- ]?tun|parameter[- ]efficient|\bpeft\b|\blora\b|low[- ]rank adaptation|\badapters?\b (module|layer|tuning|for)|prompt (tuning|learning|engineering|optimi[sz]ation)|soft prompts?|instruction[- ]tun",
    "Optimization": r"stochastic gradient|\bsgd\b|non-?convex (optimi[sz]ation|problems?|objectives?|functions?)|convex (optimi[sz]ation|problems?|functions?)|\badam(w)?\b|first[- ]order (methods?|optimi)|second[- ]order (methods?|optimi)|learning rate (schedules?|warm-?up|tuning)|bilevel optimi[sz]ation|min-?max (optimi[sz]ation|problems?)|variational inequalit|zeroth[- ]order|saddle[- ]point",
    "Learning theory / generalization": r"generali[sz]ation (bounds?|error|gap|theory|guarantees?)|pac[- ]bayes|\bpac\b|sample complexity|vc[- ]dimension|rademacher|statistical learning theory|benign overfitting|double descent|implicit (bias|regularization)|neural tangent|\bntk\b|minimax (rates?|optimal)|(?<!evidence )(?<!variational )lower bounds?(?! (on|of|for) (the )?(log-?likelihood|marginal likelihood|mutual information|evidence))|\blearnability|we prove (that|a|an|the|tight|matching|upper|lower|novel|new)|we (establish|derive) (that|tight|matching|upper|lower)|we (establish|derive) ([\w-]+ ){0,3}?(bounds?|rates?|guarantees?|regret|complexity|convergence|theorems?|identifiab\w*|consisten\w*|optimality|minimax|separation|impossibility|hardness|equivalence|characteri[sz]ation|inequalit\w*|limits?|lemma|stability|risk|error|competitive ratio|trade-?offs?|duality)\b",
    "Bayesian & probabilistic inference": r"bayesian (inference|neural|optimi[sz]ation|models?|methods?|approach|deep|learning|posterior|framework|nonparametric)|variational inference|\bmcmc\b|markov chain monte carlo|langevin|posterior (sampling|inference|approximation)|(?<!diffusion )probabilistic (models?|programming|inference|circuits?)|sequential monte carlo|stein variational|expectation[- ]maximi[sz]ation|simulation[- ]based inference|likelihood[- ]free inference",
    "Clustering / unsupervised learning": r"clustering|unsupervised (learning|representation|methods?|setting|manner|fashion|pre-?training|clustering|segmentation|discovery|anomaly)|dimensionality reduction|manifold learning|k[- ]means",
    "Multi-task / transfer learning": r"multi[- ]task (learning|models?|training|setting|optimi[sz]ation|networks?|benchmarks?)|transfer learning|transferab(le|ility) (of|to|across)(?! (the )?(adversarial|attacks?|perturbations?|black-?box|((unseen|different|other|multiple|various|diverse) )?(target|victim|surrogate|black-?box) models?))|auxiliary tasks?",
    "Knowledge distillation": r"knowledge distillation|distill(ation|ing|ed)? (from|into|the knowledge|knowledge|a (large|smaller|student))|teacher[- ]student|dataset distillation|(consistency|progressive|adversarial|step) distillation|self[- ]distillation|distillation (loss|framework|method|approach|objective)",
    "Interpretability / explainability": r"interpretab|explainab(?!leml)|\bxai\b|saliency (maps?|methods?)|feature attributions?|attribution (methods?|maps?|scores?)|mechanistic|sparse auto-?encoders?|concept bottleneck|counterfactual explanations?|probing (classifiers?|tasks?|experiments?|analysis)",
    "Healthcare / clinical": r"clinical|healthcare|health care|\behrs?\b|electronic health|patients?\b|diseases?\b|mortality|survival analysis|epidemi(?!c (cascades?|learning))|biomedical|medical (records?|questions?|diagnosis|knowledge|qa|reports?|domain|applications?|data)",
    "Synthetic data / data augmentation": r"data augmentation|augmentation (strateg|polic|methods?|techniques?|pipelines?)|\bmixup\b|cutmix|(generat\w*|synthesi[sz]\w*|creat\w*|produc\w*|render\w*) (large-scale |high-quality |diverse |realistic )?synthetic (data|images?|datasets?|samples?|training)|synthetic (data|images?|samples?) (generation|for training|to train|pipelines?|engines?)|train(ed|ing)? (on|with) (synthetic|generated) (data|images?|samples?)|self[- ]generated (data|training)",
    "Human pose / body / hands": r"(human|body|hand|multi-person|person|2d human|3d human|whole-body|full-body|animal) pose|pose estimation of (humans?|people|persons?)|body (pose|shape|mesh|models?)|\bsmpl|hand (pose|reconstruction|mesh|tracking)|hands?[- ](and[- ])?objects? interactions?|(human|body|whole-body|animal|hand) keypoints?|human mesh",
    "World models / video prediction": r"\b(?:learn\w*|latent|deep|generative|neural|imagin\w*|dream\w*|predict\w*|videos?|train\w*|model-based|reinforcement|\brl\b|driving|dynamics?|simulat\w*|embodied|visual|pixels?|polic(?:y|ies)|interactive|action|autoregressive|diffusion|foundation models?)[^.;]{0,150}?(?<!real-)(?<!real )(?<!open-)(?<!open )(?<!semantic )(?<!spatial )world[- ]?models?|(?<!real-)(?<!real )(?<!open-)(?<!open )(?<!semantic )(?<!spatial )world[- ]?model(?:s|l?ing)?[^.;]{0,150}?\b(?:learn\w*|latent|deep|generative|neural|imagin\w*|dream\w*|predict\w*|videos?|train\w*|model-based|reinforcement|\brl\b|driving|dynamics?|simulat\w*|embodied|visual|pixels?|polic(?:y|ies)|interactive|action|autoregressive|diffusion|foundation models?)|video prediction",
    "Multimodal learning / cross-modal": r"multi-?modal(?! (data )?(distributions?|posteriors?|densit\w*|landscapes?|functions?|objectives?|optimi[sz]ation|behaviou?rs?|action distributions?|targets?|trajector\w*|motion (prediction|forecasting)|futures?|predictions?|outputs?|solutions?|modes?|prototypes?)\b)|multi-modality|cross-?modal|audio-?visual|omni-?modal|(sensor|modality|multi-sensor|rgb-?d|rgb-?t) fusion|(two|multiple|different|diverse|heterogeneous|missing|other|each|new|unseen|additional|auxiliary|visual|textual|text|image|audio|sensory|input) modalit(y|ies)\b",
    "Face analysis": r"\bface (recognition|detection|reconstruction|generation|editing|swapping|anti-spoofing|verification|images?|parsing|alignment|attributes?|landmarks?|forgery|super-resolution|restoration|animation|reenactment|models?)|\bfacial|talking[- ](heads?|faces?)|lip[- ]sync|human faces?|portrait",
    "Graph learning / knowledge graphs": r"(?!x)x",
    "Image classification / recognition": r"image (classification|recognition)|visual recognition|fine[- ]grained (visual )?(classification|recognition|categori[sz]ation)|long[- ]tail(ed)? (recognition|classification|learning|distributions?)|(?<!conditional )imagenet(?![- ]?(32|64|128|256|512)\b)(?! ?\(?\d{2,3} ?[×x*] ?\d{2,3})",
    "Speech / audio / music": r"(?<!part-of-)(?<!part of )(?<!hate )speech|\baudio\b|\bsounds? (events?|separation|generation|source)|music|speaker (recognition|verification|diarization|identification)|\bvoices?\b (conversion|cloning|synthesis)|acoustic (models?|features?|scene|signals?)|text[- ]to[- ]speech|spoken",
    "Few-shot / zero-shot / open-vocabulary": r"few[- ]shot|zero[- ]shot (learning|classification|generalization|transfer|recognition|segmentation|detection|setting|manner|performance)|open[- ]vocabulary|open[- ]set|novel (classes|categories)|unseen (classes|categories)",
    "Flow matching / normalizing flows": r"rectified flows?|normali[sz]ing flows?|continuous normali[sz]ing|stochastic interpolants?",
    "VAEs / vector-quantized models": r"variational auto-?encoders?|\bvaes?\b|vq-?vae|vq-?gan",
    "State-space models / Mamba / linear attention": r"\b(?:structured|selective|diagonal|linear) state[- ]space|\bmamba|state[- ]space (?:models?|layers?) \((?:ssms?|s4|s5)\)|(?<!bi)(?<!multi-)(?<!multi)(?<!x-)(?<!non)linear attention|\brwkv\b|(?<!non)(?<!non-)linear recurren|sub-?quadratic(?:ally)?[^.]{0,80}?(?:attention|transformers?|sequence|architectures?|alternatives?|language models?|recurren\w*|state[- ]space|ssms?|hyena)|(?:attention|transformers?|sequence|architectures?|alternatives?|language models?|recurren\w*|state[- ]space|ssms?|hyena)[^.]{0,80}?sub-?quadratic",
    "Spiking / neuromorphic networks": r"(?<!neural )(?<!neuronal )spiking(?! (?:activity|data|rates?|statistics|recordings?|responses?|trains?|patterns?|variability|counts?)\b)|neuromorphic(?!-paris)(?![- ](?:event[- ]based |vision |visual |image |imaging |polarization )?(?:cameras?|sensors?|events?|data(?:sets?)?|imaging|vision|x-ray|signals?|streams?)\b)",
    "3D Gaussian splatting": r"gaussian[- ]splatting|\b3dgs\b|\b4dgs\b|gaussian splats?",
    "Continual / incremental learning": r"continual learning|lifelong learning|incremental learning|class[- ]incremental|catastrophic forgetting|task[- ]incremental",
    "Multi-agent learning / game theory": r"multi[- ]agent (reinforcement|rl|learning|cooperation|coordination)|\bmarl\b|game[- ]theor|nash equilibri|equilibri(um|a) (computation|learning|finding)|mean[- ]field games?|mechanism design|auctions?\b|(markov|zero-sum|general-sum|two-player|multi-player|extensive[- ]form|normal[- ]form|potential|stochastic|cooperative|non-cooperative|bayesian|stackelberg|matrix|congestion|repeated|differential|imperfect[- ]information|sequential) games?\b",
    "Bandits / online learning": r"bandits?\b|regret (bounds?|minimi[sz]ation)|online convex|thompson sampling|\bucb\b|best[- ]arm",
    "Adversarial robustness / attacks": r"adversarial (examples?|attacks?|robustness|perturbations?|defen[cs]es?|vulnerabilit)|certified robust|robustness certificat|randomi[sz]ed smoothing|evasion attacks?|adversarially robust",
    "Fairness / bias": r"fairness|\bfair (classif|represent|learning|ranking|machine|ml|allocation|clustering|regression)|algorithmic fairness|demographic (parity|groups?|bias)|social bias|gender bias|racial bias|stereotyp|protected (attributes?|groups?)|equal(i[sz]ed)? (odds|opportunity)|group robustness|spurious correlation",
    "Uncertainty / calibration / conformal": r"uncertainty (quantification|estimation|calibration|aware)|uncertainty-aware|conformal|predictive uncertainty|epistemic|aleatoric|(confidence|model|probability) calibration|well[- ]calibrated|miscalibrat",
    "OOD / anomaly detection": r"(out[- ]of[- ]distribution|\bood\b)(\W+\w+){0,3}?\W+(detect\w*|identif\w*|reject\w*|scor\w*|discriminat\w*|flag\w*|filter\w*|separat\w*|segment\w*|locali[sz]\w*|abstain\w*|abstention|guard\w*|safeguard\w*|monitor\w*)|(detect\w*|identif\w*|reject\w*|scor\w*|discriminat\w*|flag\w*|filter\w*|separat\w*|segment\w*|locali[sz]\w*|abstain\w*|abstention|guard\w*|safeguard\w*|monitor\w*)(\W+\w+){0,3}?\W+(out[- ]of[- ]distribution|\bood\b)|anomal(y|ies|ous)(\W+\w+){0,2}?\W+(detect\w*|segment\w*|locali[sz]\w*|scor\w*|identif\w*|discover\w*|benchmarks?|datasets?)|(detect\w*|segment\w*|locali[sz]\w*|identif\w*|discover\w*)(\W+\w+){0,2}?\W+anomal\w*|novelty detection|outlier detection|open[- ]set recognition",
    "LLM safety / jailbreak": r"jailbreak|red[- ]teaming|harmful (content|outputs?|requests?|prompts?)|prompt injection|refusal|guardrails?|(llm|ai|model) safety(?![- ]gym)|safety (alignment|of (llms?|language|large))|unsafe (content|outputs?|prompts?)",
    "Efficient inference / sampling": r"efficient (serving|decoding|attention|generation|sampling)|speculative decoding|kv[- ]cache|token (pruning|merging|reduction|compression|dropping)|early[- ]exit|accelerat(e|ing|ion) (inference|sampling|diffusion|generation|decoding)(?![\w -]{0,40}\btraining\b(?!-free))|inference (acceleration|efficiency|latency|cost)|few[- ]step (generation|sampling|diffusion)|one[- ]step (generation|diffusion|sampling)",
    "Neural architecture search / AutoML": r"architecture search|automl|hyper-?parameter (optimi[sz]ation|search)",
    "Causal inference / discovery": r"causal (inference|discovery|effects?|structure|representation|graphs?)|treatment effects?|instrumental variables?|interventional|counterfactual (inference|reasoning|prediction|estimation|outcomes?)",
    "Object 6D pose estimation": r"(object|category-level|instance-level) pose (estimation|tracking)|(?<!non-rigid )(?<!geocentric )object pose|6-?dof object pose",
    "Video understanding / action recognition": r"video (understanding|recognition|classification|question answering|grounding|captioning|retrieval|representation|summari[sz]ation|llms?|language models?|reasoning)|\baction (recognition|detection|localization|segmentation|anticipation)|temporal (action|grounding|localization|sentence grounding)|activity recognition|long[- ]form video|video[- ]language|egocentric(?! (motions?|observers?|perspectives?|viewpoints?|frames?|coordinates?|maps?|representations?|depth|reference|navigation|agents?))",
    "Medical imaging": r"medical (imag|image|vision|segmentation|visual)|\bmri\b|(?-i:\bCT\b)(?=[- ]?(scans?|images?|imaging|volumes?|slices?|reconstruction|data(sets?)?|modalit\w*|segmentation|projections?|sinograms?|perfusion|angiograph\w*))|(low-dose|chest|cone-beam|\bpet|3d|abdominal|head|thoracic|lung|cardiac|brain|spectral|dental|sparse-view|limited-angle|4d|micro|medical) (?-i:CT\b)|computed tomography|x-?rays?\b|radiolog|histopatholog|(computational|digital|clinical) patholog|pathologists?\b|pathology (images?|slides?|imaging|foundation models?|reports?|datasets?)\b|pathological (images?|regions?|tissues?|slides?|lesions?|features?|areas?)|whole[- ]slide|ultrasound|\bfmri\b|retinal(?! (ganglion|mosaics?|periphery|representation manifold|circuits?|neurons?|cells?|coding|responses?|computations?))|fundus|dermatolog|endoscop|brain (imaging|tumou?r|mri)|mammogra|clinical imag",
    "Autonomous driving": r"au-?tonomous driving|self[- ]driving|\b(urban|automated|automatic|assisted|safe|intelligent|on-road|end-to-end|human|real-world|autonomous safe|naturalistic|closed-loop|indian|vehicle) driving\b|\bauto-?driving\b|\bdriving[- ](scenes?|scenarios?|datasets?|environments?|simulators?|simulation|policy|policies|behaviou?rs?|agents?|tasks?|applications?|assistance|safety|systems?|benchmarks?|perception|planning|world models?|domains?|situations?|conditions?|logs?|maneuvers?|decisions?|actions?|space|data(sets?)?|vehicles?|specific|videos? datasets?)\b|autonomous vehicles?|bird.?s[- ]eye[- ]view|\bbev\b|(?<!human )(?<!3d human )(?<!body )motion forecasting|trajectory (prediction|forecasting)|traffic (scenes?|scenarios?|participants?|simulation|agents?)|nuscenes|waymo|\bcarla\b|lane detection|occupancy (prediction|forecasting)",
    "Depth / stereo / optical flow / matching": r"depth (estimation|prediction|completion|maps?)|monocular depth|stereo (matching|depth|vision)|optical flow|scene flow|feature matching|image matching|(?<!human )keypoint (matching|description|detection)|dense correspondences?",
    "Point clouds / 3D understanding": r"point clouds?|lidar|3d (object detection|semantic|scene understanding|shape analysis|understanding|perception|vision[- ]language|grounding|visual grounding|question answering|point[- ](sets?|pairs?|convolutions?|language))|occupancy (grids?|networks?)|mesh (segmentation|analysis)|shape (correspondence|matching|analysis)",
    "Scientific ML / PDEs / physics": r"\bpdes?[- ]solving|solv\w* (\w+ ){0,2}(pdes?|partial differential equations?)\b|partial differential equations? solving|neural operators?|physics[- ]informed|\bpinns?\b|fluid (dynamics|simulation|flows?)|navier|climate[- ](models?|modell?ing|data|datasets?|forecast\w*|predictions?|projections?|emulat\w*|simulat\w*|science|dynamics|variables?|analysis|downscal\w*|extremes?|systems?|fields?|reanalysis)|\bclimax\b|sub-?seasonal|weather and climate|climate and weather|(?<!adverse )(?<!bad )(?<!diverse )(?<!extreme )(?<!all-)(?<!all )weather (forecast|prediction|models?|data)|numerical weather|turbulen|scientific (machine learning|discovery|computing)|(?<!cyber-)(?<!cyber )physical (systems?|simulation)|computational fluid",
    "NLP tasks": r"machine translation|(?<!video )(?<!data )summari[sz]ation|natural language (understanding|inference|generation|processing)|\bnlp\b|text classification|named entity|sentiment|(?<!talking )dialog(ue)? (systems?|generation|models?)|information extraction|translation (quality|models?)",
    "Robotics / embodied AI": r"\brobot|embodied|robotic|vision[- ]language[- ]action|\bvlas?\b|locomotion|(?<!image )(?<!photo )(?<!face )(?<!text )manipulation (tasks?|policies|skills?)|(visual|vision[- ]language|object[- ]goal) navigation|dexterous (hand|grasp|manipulat)\w*|\bgrippers?\b|bin[- ]picking|6-?dof grasp\w*",
    "Combinatorial optimization / scheduling": r"combinatorial (optimi[sz]ation|problems?)|traveling sales(man|person)|(?<!\()\btsp\b(?!\))(?!-[a-z])|vehicle routing|(?<!\bnoise )(?<!\bcovariance )(?<!\btoken )(?<!\breward )(?<!\bcurriculum )(?<!\bdata )(?<!\brate )(?<!\blr )(?<!\bdecay )(?<!\bdropout )(?<!\bentropy )(?<!\btimestep )(?<!\btime-step )(?<!\bstep )(?<!\bguidance )(?<!\bquantization )(?<!\bpatch )(?<!\btile )(?<!\bblock )(?<!\bdilated )(?<!\bvisual )(?<!\bsample )(?<!\bsize )(?<!\bwindow )(?<!\bcyclical )(?<!\bdiscount )(?<!\btemperature )(?<!\bflow )(?<!\bproximity )(?<!\bweight )(?<!\brehearsal )(?<!\bcognitive )(?<!\bgradient-aware )(?<!\brelaxed )(?<!\bupdate )(?<!\bfusion )(?<!\bview )(?<!\bthreshold )(?<!\bparameter )(?<!\boptimization )(?<!\blinear )(?<!\bgranularity )(?<!\bfast-slow )(?<!\bdual-timestep )(?<!\bexploration )(?<!\bregularization )(?<!\bkeyframe )(?<!\bmask )(?<!\bmasking )(?<!\bsampling )(?<!\bstability )(?<!\bloss )(?<!\baugmentation )(?<!\bmulti-objective )(?<!\btraining )(?<!\bdifficulty )(?<!\bramp )(?<!\btime-)(?<!\bepsilon-)(?<!\bmanual )scheduling(?! (of (the )?(memory|data|causal|learning|noise|guidance|classifier|weights?|thoughts)|weight|thoughts|your llm|strateg))|mixed[- ]integer|\bmilp\b|boolean satisfiability|\bsat solv|integer programming",
    "Recommendation / information retrieval": r"recommend(ation|er|ers)\b|(?<!\bpractical )(?<!\bprovide )(?<!\bprovides )(?<!\bproviding )(?<!\bconcrete )(?<!\bactionable )(?<!\bmake )(?<!\bmakes )(?<!\bmaking )(?<!\bour )(?<!\bpolicy )(?<!\bclear )(?<!\bkey )(?<!\bseveral )(?<!\boffer )(?<!\boffers )(?<!\bai )(?<!\brecourse )(?<!\bdesign )(?<!\bsimple )(?<!\bgeneral )(?<!\bspecific )(?<!\buseful )(?<!\bimportant )(?<!\bexplicit )(?<!\bsome )recommendations\b(?! (for|on|to) (improving|practitioners|future|researchers|how|the design|designing|practice|building|using|policy|policymakers|model (developers|selection|design)|evaluation|best|data|training|mitigating|addressing))|recommender|collaborative filtering|information retrieval|search engines?|dense retrieval|learning to rank|query (expansion|rewriting)|click[- ]through|retrievers?\b",
    "Vision-language / multimodal LLMs": r"(?:vision|visual)[- ]language(?![- ](?:navigation|correspondences?|features?|references?|pairs|fusion|interactions?|disciplin\w*)\b)(?!s\b)|vision[- ]and[- ]language (models?|pre-?training|pretrained|representations?|transformers?)|multi-?modal (large )?(language models?|llms?|reasoning models?|foundation models?)|omni-?modal (large )?(language models?|llms?|models?)|audio-visual (large )?(language models?|llms?)|\bmllms?\b|\blvlms?\b|\bvlms?\b|large multi-?modal models?|\blmms?\b|(?-i:\bCLIP\b)|image[- ]text (models?|pre-?training|alignment|pairs)|language[- ]image pre-?training",
    "Mathematical reasoning / theorem proving": r"(?<!\bwe )(?<!provide )(?<!give )(?<!present )formal proofs?|formal mathematics|\blean ?4|(?<=-)lean\b|\blean-\w|(?:(?<=\bin )|(?<=\bvia )|(?<=\bfor )|(?<=\bof )|(?<=\bthe )|(?<=\band )|(?<=\busing )|(?<=\bfrom )|(?<=\binto ))lean\b|\blean (?:theorems?|proofs?|provers?|code|verifier|compiler|repositor\w*|projects?|workbook|dojo|copilot|hammer|environment|kernel|tactics?|formali\w*|problems?|statements?|version)\b|\bmathlib|(?<!by strict )(?<!by rigorous )(?<!\bby )math(?:ematical)? (?:reasoning|word problems?|benchmarks?)|\bmath problems?|(?<!\ba )(?<!\bas )(?<!underlying )(?<!\bthat )(?<!\bthis )mathematical problems?|\bgsm8k\b|olympiad|auto-?formali[sz]|math(?:ematical)? problem[- ]solving",
    "Retrieval-augmented generation": r"retrieval[- ]augmented|retrieval[- ]enhanced|knowledge[- ]augmented generation",
    "Alignment / RLHF / preference optimization": r"\brlhf\b|preference (?:optimi[sz]ation|alignment|tuning)|\bdpo\b|(?:llm|language model|value|human|ai) alignment|aligning (?:llms?|language models?|large|ai)|constitutional ai|superalignment",
    "Hallucination / factuality": r"factuality|factual (?:consistency|errors?|accuracy)|hallucinat\w*(?:\W+\w+){0,5}?\W+(?:spurious|non-existent|nonexistent|implausible|incorrect|unfaithful|faithfulness|erroneous\w*|fabricat\w*|inconsisten\w*|absent\b|undesir\w*|unwanted|unreliab\w*)|(?:spurious|non-existent|nonexistent|implausible|incorrect|unfaithful|faithfulness|erroneous\w*|fabricat\w*|inconsisten\w*|absent\b|undesir\w*|unwanted|unreliab\w*)(?:\W+\w+){0,4}?\W+hallucinat|(?:risk|prone|susceptible|suffers?|suffering|severe)\W+(?:\w+\W+){0,3}?hallucinat|(?<!face )(?<!feature )(?<!texture )(?<!style )(?<!depth )hallucinations\b|object hallucination|non-hallucinat|hallucination[- ](?:free|resistant|resilient|aware|detection|mitigation|evaluation|benchmark\w*|rates?|control|reduction|problems?|issues?)|(?:mitigat|reduc|alleviat|suppress|prevent|eliminat|avoid|curb|combat|address|tackl|minimi[sz])\w*\s+(?:\S+\s+){0,4}hallucinat|(?:video|contextual|model|object|cross-modal|multi-?modal|knowledge|factual|audio|textual|relation\w*|reasoning|spatial|temporal|action|stroke) hallucination|hallucinated (?:artifacts?|structures?|content|objects?|information|details?|regions?|responses?|answers?|facts?|text|tokens?|actions?)|collapse to hallucination|suffers? (?:from )?(?:\w+ )?hallucination",
    "Model merging / editing / unlearning": r"model merging|\bmerging (models|weights|llms?)|task arithmetic|knowledge editing|model editing|(?<!bias )(?<!noise-)(?<!noise )(?<!alleviating )unlearning(?! (?:the |of )?(?:biases|bias\b|spurious))|machine unlearn",
    "Long context": r"long[- ]context|context (length|window)s?|length (generali[sz]ation|extrapolation)|long[- ]range arena",
    "Scaling laws / pretraining": r"scaling (laws?|behaviou?r)|compute[- ]optimal|pre-?training (data|corpus|corpora|recipe)|(?<!streaming-)data mixtures?(?! models?)|emergent abilit\w* (?:of|in) (?:large |llms?|language models?)|(?:predict\w*|mirage|sharp\w*|scale|scaling|size)\W+(?:\w+\W+){0,4}?emergent abilit|emergent abilit\w*\W+(?:\w+\W+){0,6}?(?:scale|scaling|size|threshold)",
    "Convolutional networks": r"(?<!graph )(?<!graph-)(?<!edge )(?<!mesh )convolutional (neural )?networks?|\bcnns?\b|convnets?",
    "Self-supervised / contrastive learning": r"self[- ]supervised|contrastive (learning|loss|pre-?training|representation\w*|objective)|representation learning|masked (image|autoencoders?|modell?ing|language modell?ing)|\bmae\b",
    "Active learning / data selection": r"\bactive learning|coresets?\b|data pruning|data curation|data valuation",
    "Quantization / pruning / compression": r"sparsif|low[- ]bit|binar(y|ized) neural|model compression|compress(ing|ion of) (the )?(models?|networks?|llms?|transformers?)|lottery ticket",
    "Statistical estimation / high-dimensional statistics": r"high[- ]dimensional (statistics|regression|estimation|inference)|sparse (regression|recovery|estimation|linear(?! (systems?|solvers?|layers?|attention|algebra|encoding|transforms?|surrogates?|rnns?|concept)))|\blasso\b|matrix (completion|factori[sz]ation|sensing|estimation)|tensor (completion|factori[sz]ation)|robust (statistics|estimation|mean estimation|regression)|hypothesis test|two[- ]sample test|(?<!crowd )(?<!object )density estimation|covariance estimation|mean estimation",
    "Segmentation": r"segment anything|(?<!cmu )(?<!cmu-)panoptic(?![- ]studio)|matting|segmenter|(semantic|instance) masks?",
    "OCR / documents / charts": r"\bocr\b|optical character(?!istic)|(?<!ai )(?<!ai-generated )(?<!machine-generated )(?<!llm-generated )(?<!generated )(?<!erroneous )text (recognition|detection|spotting)|scene text(?!ur)|document (understanding|parsing|images?|layout|visual|ai|vqa)|\bchart(s)? (understanding|question answering|qa|reasoning|parsing|summari[sz]ation|generation|captioning|recognition|images?|data|figures?)|\bchartqa\b|(bar|line|pie|scatter) (charts?|plots?)|charts?,? (and|or) (tables?|plots?|diagrams?|figures?|documents?)|(tables?|plots?|diagrams?|figures?|documents?),? (and|or) charts?|chart-to-\w+|table (understanding|recognition|parsing)|handwrit|layout analysis",
    "Tabular data": r"(?!x)x",
}
# Phrases whose sense in an abstract depends on other words of the same abstract (precision review, round 1):
# topic -> [(phrase, context must also match, context must not match, context = "abstract" or "sentence")]; None = no condition.
ABSTRACT_CONDITIONAL = {
    "Flow matching / normalizing flows": [
        (r"flow[- ]matching",
         None,
         r"gflownet|generative flow network", "abstract"),
    ],
    "VAEs / vector-quantized models": [
        (r"vector[- ]quantiz",
         None,
         r"weights?\b|kv[- ]cache|low-bit\b|\b\d-bit\b|sub-1-bit|bit-?widths?\b|post-training|model compression|quantization-aware|quantiz\w* (?:for|of) (?:llms?|large language models?)", "sentence"),
    ],
    "State-space models / Mamba / linear attention": [
        (r"\bdeep state[- ]space",
         r"long[- ]range|\bs4\b|\bs5\b|mamba|linear recurren|linear rnn|structured state|selective|hippo|\blru\b|sequence model",
         None, "abstract"),
        (r"\b(?:s4|s5)\b",
         r"state[- ]space|\bssms?\b|long[- ]range|long sequence|sequence model",
         None, "abstract"),
    ],
    "Few-shot / zero-shot / open-vocabulary": [
        (r"open[- ]world",
         None,
         r"\b(minecraft|embodied|robots?|robotic|navigation|video games?|games?|manipulation|autonomous driving|agents?)\b", "abstract"),
    ],
    "Continual / incremental learning": [
        (r"continual pre-?training",
         r"\bforget",
         None, "abstract"),
    ],
    "Multi-agent learning / game theory": [
        (r"multi[- ]agent systems?",
         None,
         r"\b(llms?|large language models?|language models?|gpt[- ]?\w*|mllms?|vlms?|chatgpt|agentic)\b", "abstract"),
    ],
    "Bandits / online learning": [
        (r"online learning",
         r"\b(regrets?|no[- ]regret|mistake[- ]bounds?|littlestone|expert advice|online convex|bandits?|adversarial (losses|online|environments?|sequences?)|online-to-(batch|non-?convex)|optimistic online|follow[- ]the[- ](regularized|perturbed)[- ]leader|ftrl|online gradient descent|multiplicative weights|hedge|competitive ratio|learnab\w*|sample complexity|minimax|equilibri\w*|sequential prediction|cumulative loss|games?\b(?! (engine|play)))\b",
         None, "abstract"),
    ],
    "Adversarial robustness / attacks": [
        (r"(?<!domain[- ])(?<!non-)adversarial training",
         r"robust|attack|perturb|defen[cs]e|certif",
         None, "abstract"),
    ],
    "Fairness / bias": [
        (r"debias",
         r"fair|gender|racial|\brace\b|social|demographic|sensitive attributes?|protected|stereotyp|spurious|shortcut|bias-conflicting|bias-aligned|minority group|worst-group|group robust|ethnic|skin tone|dataset'?s? bias|bias(ed)? (features?|attributes?|information)|biased datasets?|bias removal",
         None, "abstract"),
    ],
    "Uncertainty / calibration / conformal": [
        (r"(?<!camera )(?<!sensor )(?<!extrinsic )calibration errors?",
         r"uncertaint|confiden|probabilit|overconfiden|miscalibrat|\bece\b",
         r"camera|lidar|extrinsic|multi-view", "abstract"),
        (r"deep ensembles?",
         r"uncertaint|calibrat|out-of-distribution|\bood\b|predictive (distribution|variance)|bayesian|posterior|confidence",
         None, "abstract"),
    ],
    "LLM safety / jailbreak": [
        (r"toxic",
         None,
         r"drug|dose|molecul|compound|clinical|chemical|protein|peptide|tox21|toxcast|admet|herg", "abstract"),
    ],
    "Efficient inference / sampling": [
        (r"efficient inference",
         None,
         r"variational|posterior|\belbo\b|evidence lower bound|bayesian|latent variables?|gaussian process|mcmc|markov chain|amortized inference|approximate inference|probabilistic model", "abstract"),
    ],
    "Neural architecture search / AutoML": [
        (r"\bnas\b",
         r"architecture|search",
         None, "abstract"),
    ],
    "Bayesian & probabilistic inference": [
        (r"importance sampling",
         r"posterior|bayesian|monte carlo|marginal likelihood|partition function|normaliz(ing|ation) constant|proposal|target distribution|unnormali[sz]ed|rare event|boltzmann",
         r"off-policy|policy gradient|\brays?\b|nerf|render|timestep|noise schedule|log-snr|sketch|subsampl|mini-batch", "sentence"),
    ],
    "Causal inference / discovery": [
        (r"(?<!non-)causality|causal (models?|reasoning|learning)",
         None,
         r"causal (attention|mask\w*|transformers?|decoders?|tokeni\w*|convolution\w*|vae)|autoregressive|streaming|temporal causality|frame-by-frame|uni-?directional|next-token", "abstract"),
        (r"confound",
         r"causal|treatment|counterfactual|interven|observational|propensity|back-?door|adjust",
         None, "abstract"),
    ],
    "Object 6D pose estimation": [
        (r"objects?\W+(?:\w+\W+){0,5}?(6d|6-?dof) poses?|(6d|6-?dof) poses?\W+(?:\w+\W+){0,5}?objects?\b",
         None,
         r"\b(camera[- ](poses?|control\w*|trajector\w*|motions?|orientations?|information|locali[sz]ation)|device poses?|head-mounted|relocali[sz]ation|visual locali[sz]ation|odometry|slam|ego-?motion|extrinsics?|registration|assembl(y|ing))\b", "abstract"),
    ],
    "Depth / stereo / optical flow / matching": [
        (r"(?<!non-)\blocal features?\b",
         r"\b(match(ing|es|er)?\b|correspondences?|registration|(?-i:SIFT)\b|descriptors?|keypoints?|geometric verification|relocali|visual locali[sz]ation|ransac|homograph)",
         None, "sentence"),
    ],
    "Point clouds / 3D understanding": [
        (r"voxel\w*",
         r"\b(detect\w*|occupancy|scene completion|point[- ]based|points?\b|lidar|semantic|instances?|panoptic|3d segmentation|sparse conv\w*)",
         None, "sentence"),
    ],
    "Scientific ML / PDEs / physics": [
        (r"partial differential|\bpdes?\b",
         r"solver|surrogate|simulat|numerical|operator learning|neural operator|physic|fluid|scientific|engineering|mesh|boundary condition|initial condition|navier|burgers|darcy|discreti[sz]|turbulen|climate|weather|spatio-?temporal|time-dependent|parametric pde",
         None, "abstract"),
    ],
    "NLP tasks": [
        (r"(?<!visual )(?<!video )(?<!3d )(?<!d )question answering",
         None,
         r"\b(visual|video|videos|image|images|vlms?|lvlms?|mllms?|lmms?|vision|multimodal|multi-modal|3d|egocentric|chart|scene)\b", "sentence"),
    ],
    "Combinatorial optimization / scheduling": [
        (r"neural solvers?",
         None,
         r"\b(pdes?|partial differential|physics|navier|fluid|poisson|multigrid)\b", "abstract"),
    ],
    "Graph learning / knowledge graphs": [
        (r"knowledge graphs?|link prediction|node classification|(?<!scene )graph (generation|matching|clustering|representation learning|structure learning|classification|learning)|heterogeneous graphs?|hypergraphs?",
         None,
         r"\bscene graph (generation|prediction|parsing)\b", "abstract"),
    ],
    "Agents / tool use": [
        (r"\bmobile agents?\b",
         r"\b(gui|smartphones?|android|apps?|screens?|mllms?|vlms?|llms?|(large )?(vision[- ])?language models?)\b",
         None, "abstract"),
    ],
    "Reasoning / chain-of-thought": [
        (r"\b(?:3d|causal|counterfactual) reasoning",
         r"(?:(?:llms?|vlms?|mllms?|lvlms?|lmms?)\b|\b(?:chatgpt|gpt-?\d\w*|chatbots?|(?:large )?(?:vision[- ])?language models?|multi-?modal (?:large )?language|large multi-?modal|multi-?modal models?|unified models?|foundation models?|captioning|question[- ]answering|vqa|summar\w+|generated (?:text|answers?|responses?)|report generation|text generation|knowledge editing|factual\w*|dpo|rlhf|preference (?:optimi[sz]ation|alignment|data)|trustworth\w*|text-to-video|video-to-text|retrieval[- ]augmented|rag|refusal|jailbreak\w*)\b)",
         None, "abstract"),
    ],
    "Mathematical reasoning / theorem proving": [
        (r"theorem prov(?:ing|ers?)\b",
         None,
         r"soft[- ]unification|knowledge (?:base|graph)s?|link prediction", "abstract"),
        (r"formal verification",
         r"\b(?:llms?|language models?|theorems?|proofs?|provers?|lean|coq|isabelle|dafny|verus|code generation|code synthesis|programs?|verilog|software|math\w*)\b",
         None, "abstract"),
    ],
    "Retrieval-augmented generation": [
        (r"\brag\b",
         None,
         r"(?<!augmented generation )(?<!augmented-generation )(?<!augmentation generation )(?<!augmentation-generation )\(rag\)", "abstract"),
    ],
    "Alignment / RLHF / preference optimization": [
        (r"human feedback",
         r"\b(?:llms?|language models?|rlhf|human preferences?|alignment|aligned|aligning|diffusion|text-to-\w+|generative|chatbots?|assistants?|instruction\w*|mllms?|vlms?|lvlms?|lmms?|dpo|grpo|verifiers?|preference-based|pbrl|prompts?|responses?|chain-of-thought|multimodal|generation|gui|group relative|contrastive preference)\b",
         None, "abstract"),
        (r"reward model(?:s|l?ing)?\b|preference (?:learning|data)",
         r"\b(?:llms?|language models?|rlhf|human feedback|human preferences?|alignment|aligned|aligning|diffusion|text-to-\w+|generative|chatbots?|assistants?|instruction\w*|mllms?|vlms?|lvlms?|lmms?|dpo|grpo|verifiers?|preference-based|pbrl|prompts?|responses?|chain-of-thought|multimodal|generation|gui|group relative|contrastive preference)\b",
         None, "abstract"),
        (r"reward model(?:s|l?ing)?\b|preference (?:learning|data)",
         None,
         r"\b(?:bandits?|regret|mdps?|pomdps?|model-based (?:rl|reinforcement)|imitation|world models?|offline (?:rl|reinforcement)|inverse reinforcement|q-learning|value iteration|elections?|social choice|mallows|rankings?|robot\w*|manipulation|locomotion)\b", "abstract"),
    ],
    "Fine-tuning / PEFT / prompting": [
        (r"visual prompt",
         r"\b(?:learnable|learned|tuning|tuned|optimi[sz]\w*|adapt\w*|frozen|freez\w*|parameter[- ]efficient|reprogram\w*|prompt learning|trainable)\b",
         None, "abstract"),
    ],
    "Hallucination / factuality": [
        (r"(?<!face )(?<!feature )hallucinat",
         r"(?:(?:llms?|vlms?|mllms?|lvlms?|lmms?)\b|\b(?:chatgpt|gpt-?\d\w*|chatbots?|(?:large )?(?:vision[- ])?language models?|multi-?modal (?:large )?language|large multi-?modal|multi-?modal models?|unified models?|foundation models?|captioning|question[- ]answering|vqa|summar\w+|generated (?:text|answers?|responses?)|report generation|text generation|knowledge editing|factual\w*|dpo|rlhf|preference (?:optimi[sz]ation|alignment|data)|trustworth\w*|text-to-video|video-to-text|retrieval[- ]augmented|rag|refusal|jailbreak\w*)\b)",
         None, "abstract"),
    ],
    "Long context": [
        (r"long[- ]sequence",
         r"\b(?:quadratic\w*|(?:linear|sub-?quadratic) (?:complexity|cost)|self-attention|attention mechanism|(?:efficient|sparse|linear) attention|state[- ]space|ssms?|mamba|kv[- ]cache|context window|context length|sequence length|long[- ]range)\b",
         None, "abstract"),
    ],
    "Convolutional networks": [
        (r"(?<!graph )(?<!graph-)convolution(al)? (layers?|filters?|kernels?)",
         None,
         r"graph", "sentence"),
    ],
    "3D Gaussian splatting": [
        (r"(?<!lift-)(?<!lift, )(?<!lift )(?<!softmax )(?<!soft)(?<!many-to-many )(?<!m2m )(?<!average )(?<!surface )(?<!point )(?<!point-based )(?<!forward )(?<!differential )(?<!-aware )(?<!improved )splat(?!planner)(?!ting-based vt)(?! operation)",
         r"gaussian",
         None, "abstract"),
    ],
    "Self-supervised / contrastive learning": [
        (r"contrastive(?![- ](decoding|divergence|explanations?|preference|activation|examples?|analysis|reasoning|policy[- ]optimi[sz]ation))",
         None,
         r"preference optimi[sz]ation|\bdpo\b|\brlhf\b", "abstract"),
    ],
    "Active learning / data selection": [
        (r"sample selection",
         r"annotat|label(l)?ing (budget|cost)|budget|quer(y|ies)\b|acquisition|coresets?\b|k-?cent(er|re)|representative|\bactive\b",
         None, "abstract"),
        (r"data selection",
         None,
         r"\b(pseudo[- ]labels?|noisy[- ]labels?|label noise|test[- ]time)\b", "abstract"),
    ],
    "Quantization / pruning / compression": [
        (r"(?<!vector[- ])(?<!vector-)(?<!de)(?<!scalar )(?<!product )(?<!residual )(?<!color )(?<!semantic )(?<!lookup-free )quantiz",
         r"weights?|activations?|post-training|low-bit|\d+-bit|bit-?widths?|llms?|large language|models?|networks?|kv[- ]cache|integer|mixed[- ]precision|quantization-aware|\bptq\b|\bqat\b|inference|deploy|compress",
         r"codebook|tokeni[sz]|discreti[sz]|measurements?|compressed sensing|quantization (noise|errors?)|dequantiz", "sentence"),
        (r"(?<!branch-and-bound )(?<!tree )(?<!node )(?<!search )(?<!beam )(?<!candidate )(?<!path )(?<!data )(?<!dataset )(?<!ensemble )(?<!batch )(?<!sample )(?<!sentence-level )pruning",
         r"weights?|channels?|filters?|heads?|layers?|neurons?|parameters?|structured|unstructured|networks?|models?|llms?|sparsity|lottery|tokens?",
         r"\b(data|dataset|samples?|batch|sentences?|search|tree|beam|candidates?)\b(?! (models?|networks?))", "sentence"),
    ],
    "Optimization": [
        (r"convergence (rates?|analysis|guarantees?|to (a |the )?(stationary|global|local|optimal))",
         None,
         r"sampler|mcmc|anneal|importance sampling|posterior|marginal likelihood|normali[sz]ing flows?", "sentence"),
    ],
    "Statistical estimation / high-dimensional statistics": [
        (r"tensor decomposition",
         None,
         r"compress|\bcnns?\b|splatting|avatar|render|nerf|radiance", "abstract"),
    ],
    "Segmentation": [
        (r"(?<!action )(?<!temporal )(?<!word )(?<!speaker )(?<!model )(?<!syllabic )(?<!syllable )segmentation",
         r"\b(image|images|video|videos|pixel|pixels|mask|masks|point clouds?|scene|scenes|lidar|medical|mri|visual|vision|3d|object|objects)\b|(?-i:\bCT\b)",
         None, "abstract"),
        (r"(?<!action )(?<!temporal )(?<!word )(?<!speaker )(?<!model )(?<!syllabic )(?<!syllable )segmentation",
         None,
         r"\b(time[- ]series|syllab\w*|spoken|speech)\b", "abstract"),
    ],
    "Tabular data": [
        (r"(?<!horizon )(?<!episodic )(?<!finite )tabular(?![- ](setting|settings|case|cases|mdps?\b|markov|rl\b|reinforcement|environments?|q-learning|q-functions?|nas\b|polic(y|ies)|value functions?|games?\b|pomdps?|bandits?|regime|search|partially observable|domains?|atari))(?!,? (and|or)? ?(linear|function approximation|low-rank|continuous|atari|mujoco)\b)",
         r"tabular (data|datasets?|learning|benchmarks?|classification|regression|features?|prediction|foundation|models?)",
         None, "abstract"),
        (r"(?<!horizon )(?<!episodic )(?<!finite )tabular(?![- ](setting|settings|case|cases|mdps?\b|markov|rl\b|reinforcement|environments?|q-learning|q-functions?|nas\b|polic(y|ies)|value functions?|games?\b|pomdps?|bandits?|regime|search|partially observable|domains?|atari))(?!,? (and|or)? ?(linear|function approximation|low-rank|continuous|atari|mujoco)\b)",
         None,
         r"\b(reinforcement|mdps?|pomdps?|q-learning|regret|bellman|markov decision)\b", "abstract"),
    ],
    "Speech / audio / music": [
        (r"\btts\b",
         None,
         r"test[- ]time scaling", "abstract"),
    ],
}


URL = re.compile(r"(https?://|www\.)\S+|\b(github|gitlab|huggingface|bitbucket)\.(com|co|org|io)/\S*|\b\S+\.github\.io\S*", re.I)


def strip_urls(text):
    """Code and project links are not content (e.g. github.com/MAC-AutoML matched NAS / AutoML): removed before matching."""
    return URL.sub(" ", text)


def _sentence(text, i):
    """The sentence of text around position i (split at . ! ? ;)."""
    s = max(text.rfind(c, 0, i) for c in ".!?;") + 1
    ends = [j for j in (text.find(c, i) for c in ".!?;") if j >= 0]
    return text[s:min(ends) + 1 if ends else len(text)]


class AbstractRule:
    """Abstract rule of one topic: the regex (ABSTRACT_OVERRIDES, else the title rule) plus the conditional phrases of
    ABSTRACT_CONDITIONAL, which count only when the abstract (scope "abstract") or the phrase's own sentence (scope
    "sentence") also matches / does not match a context regex."""

    def __init__(self, base, conditional=()):
        self.base = re.compile(base, re.I)
        self.cond = [(re.compile(q, re.I), re.compile(r, re.I) if r else None, re.compile(x, re.I) if x else None, sc)
                     for q, r, x, sc in conditional]
        self.pattern = base + "".join(f" || {q} [{sc} also: {r or '-'}; not: {x or '-'}]" for q, r, x, sc in conditional)

    @staticmethod
    def _ok(r, x, ctx):
        return (r is None or r.search(ctx)) and (x is None or not x.search(ctx))

    def finditer(self, text, context=None):
        ctx = text if context is None else context            # context: the whole abstract when text is one sentence
        ms = list(self.base.finditer(text))
        for q, r, x, sc in self.cond:
            if sc == "sentence":
                ms += [m for m in q.finditer(text) if self._ok(r, x, _sentence(text, m.start()))]
            elif self._ok(r, x, ctx):
                ms += list(q.finditer(text))
        return iter(sorted(ms, key=lambda m: (m.start(), -m.end())))

    def search(self, text, context=None):
        return next(self.finditer(text, context), None)

    def mark(self, text, context=None):
        out, last = [], 0
        for m in self.finditer(text, context):
            if m.start() >= last:
                out += [text[last:m.start()], "[[", m.group(0), "]]"]; last = m.end()
        return "".join(out + [text[last:]])


ABSTRACT_COMPILED = {n: AbstractRule(ABSTRACT_OVERRIDES.get(n, p), ABSTRACT_CONDITIONAL.get(n, ())) for n, _, p in TOPICS}


def main():
    with open(os.path.join(ROOT, "data", "papers.csv")) as f:
        papers = list(csv.DictReader(f))
    names = [n for n, _, _ in COMPILED]
    with open(os.path.join(ROOT, "data", "topic_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "venue", "year"] + names)
        for p in papers:
            w.writerow([p["uid"], p["venue"], p["year"]] + [int(bool(rx.search(p["title"]))) for _, _, rx in COMPILED])
    print(f"{len(papers)} papers x {len(names)} topics written")


if __name__ == "__main__":
    main()
