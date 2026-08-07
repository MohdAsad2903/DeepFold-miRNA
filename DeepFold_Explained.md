# DeepFold — Complete Project Explanation

---

## PART 1 — THE PROBLEM

### What are miRNAs and why do SNPs cause disease?

MicroRNAs (miRNAs) are tiny RNA molecules — typically 18–25 nucleotides long — that do not code for proteins. Instead they act as regulators. They bind to messenger RNAs and either block translation or trigger degradation of that mRNA. Roughly 60% of all human protein-coding genes are regulated this way. A single miRNA can suppress dozens of genes simultaneously, which makes it a master control switch.

Before a miRNA becomes functional, it starts as a longer precursor called a pre-miRNA. This pre-miRNA folds back on itself into a hairpin structure — a stem with a loop at the top. This specific shape is not decorative. The shape is what the cellular machinery (Drosha, then Dicer) recognises. Drosha cuts the base of the stem. Dicer cuts t    he loop off the top. If the shape is wrong, the cuts happen in the wrong place or not at all.

A SNP — a Single Nucleotide Polymorphism — is a mutation where one nucleotide is swapped for another. The key insight of this project is that even a single base change can collapse the hairpin structure. A G that used to pair with a C on the opposite strand now becomes an A. That breaks a hydrogen bond. The entire local region refolds. The Drosha recognition site is disrupted. The miRNA is never produced. The 60 genes it was supposed to regulate are now dysregulated. This is the direct route from one mutation to cancer or cardiovascular disease.

### Why is classification hard?

The same position in a hairpin can be benign in one miRNA family and pathogenic in another because the surrounding sequence context determines whether the local fold is robust or fragile. The signal is subtle — you are looking at the difference between two probability distributions over RNA folds, not a clear structural break. And the dataset ceiling is fundamental: there are only a few thousand independently validated miRNA SNPs in existence across all of human genetics. You cannot simply collect more data, because validation requires wet-lab confirmation.

Existing tools (FATHMM-MKL, CADD) were built for protein-coding variants and apply generic sequence conservation scores. They do not model RNA secondary structure or hairpin-specific thermodynamics. DeepFold was built to fill this gap.

**DeepFold in one paragraph:** DeepFold is an ensemble classifier that takes two RNA sequences — a healthy pre-miRNA and a mutant version differing by exactly one nucleotide — and predicts whether that mutation is likely to cause disease. It does this by combining four different representations of the mutation's impact: the visual change in the predicted RNA contact map (CNN), the topological change in the RNA graph structure (GCN), the compositional change in sequence motifs (k-mer XGBoost), the thermodynamic stability change (MFE XGBoost), and the evolutionary context from a large pre-trained RNA language model (RNA-FM XGBoost). A meta-learner combines all five signals into a single pathogenicity probability.

### Diagram 1 — Healthy vs Mutant Hairpin

```
HEALTHY HAIRPIN                    MUTANT HAIRPIN
                                   (G → A at position 17)

      5'                                 5'
      |                                  |
  ....UGAG....                       ....UGAG....
  ....||||....  ← stem pairs         ....||||....
  ....ACUC....                       ....ACUC....
      |                                  |
    [LOOP]                             [LOOP]
      |                                  |
  ....GC......  ← G:C pair here     ....AC......  ← A:C = NO PAIR
  ....CG......                       ....CG......     ↑
      |                                  |          SNP here
   [SEED: pos 2–8]               [SEED disrupted]
      |                                  |
      3'                                 3'

  → Drosha recognises               → Fold collapses locally
    clean stem base                   Drosha site misaligned
  → Dicer cuts correctly             → miRNA never produced
  → 60 target genes regulated       → Target genes dysregulated
```

The most important thing: one base change is enough to break the entire processing pathway because the molecular machinery reads shape, not just sequence.

---

## PART 2 — THE DATASET

### Where did the data come from?

**Disease class — COSMIC v98:** COSMIC is the Catalogue of Somatic Mutations in Cancer. It catalogues mutations found in tumour sequencing studies. Somatic mutations in cancer patients that fall within pre-miRNA regions are used as the disease class. The logic is direct: if a mutation in a miRNA hairpin appears repeatedly across cancer cohorts, it is likely disrupting that miRNA's tumour-suppressive or oncogenic function. The limitation is noise: some COSMIC mutations may be passenger mutations that happened to be in a cancer cell but did not cause the cancer. This label noise is real, and it is one reason the AUC ceiling exists.

**Benign class — gnomAD v3.1.2:** gnomAD is the Genome Aggregation Database, which catalogues naturally occurring human genetic variants across ~76,000 whole-genome sequences from healthy individuals. The filter applied was AF_popmax > 0.005. AF_popmax is the allele frequency in the most-represented population. Setting this above 0.5% means the variant must be common enough that it has been present in the human population for many generations. If a variant were truly pathogenic, natural selection would have removed it before it reached 0.5% frequency. So any variant that common is almost certainly tolerated — i.e., benign. A lower threshold (say 0.001) would include rare variants that might actually be mildly pathogenic, which would contaminate the benign class.

**Why 1000 Genomes was rejected:** The 1000 Genomes dataset has shallower coverage per sample and its variant calls in non-coding regions are less reliable than gnomAD. gnomAD was specifically designed for variant interpretation and has stricter quality filters.

**Mapping pipeline:** A variant in COSMIC or gnomAD is specified by chromosome, genomic position, reference allele, and alternate allele (VCF format). To use it as training data, the project needs to know where that genomic position falls inside the miRNA hairpin sequence from miRBase. The mapping pipeline:
1. Takes the genomic coordinate and finds which miRNA hairpin it overlaps
2. For minus-strand miRNAs, reverse-complements the coordinate arithmetic
3. Extracts the full pre-miRNA sequence from miRBase
4. Validates that the reference allele at that position matches the miRBase sequence (catches genome assembly mismatches)
5. Applies the alternate allele to produce the mutant sequence
6. Records the SNP position index within the hairpin

**Balance:** Raw COSMIC gives more disease variants than qualified gnomAD benign variants in pre-miRNA regions. After filtering, deduplication, overlap removal (any variant appearing in both COSMIC and gnomAD is removed), and downsampling, the dataset is balanced to 1,186 per class (2,372 total). Balance matters because an imbalanced dataset causes classifiers to learn a threshold bias — they implicitly predict the majority class. With balanced classes you get AUC and accuracy that reflect true discrimination.

### Diagram 2 — Dataset Construction Pipeline

```
COSMIC v98 CSV                    gnomAD v3.1.2 VCF
(somatic mutations)               (population variants)
        |                                  |
        ▼                                  ▼
 Filter: pre-miRNA               Filter: AF_popmax > 0.005
 region overlap                  (common = tolerated)
        |                                  |
   ~8,400 variants                   ~14,200 variants
        |                                  |
        ▼                                  ▼
 Map to miRBase hairpin           Map to miRBase hairpin
 (VCF coord → hairpin pos)        (VCF coord → hairpin pos)
 Validate reference allele        Validate reference allele
        |                                  |
   ~4,100 mapped                     ~6,800 mapped
        |                                  |
        ▼                                  ▼
 Deduplicate within class         Deduplicate within class
        |                                  |
   ~3,800 unique                     ~5,900 unique
        |                                  |
        └──────────┬───────────────────────┘
                   ▼
         Remove overlapping variants
         (variants appearing in BOTH sources)
                   |
          ~3,600 D | ~5,700 B
                   |
                   ▼
         Downsample majority class
         Balance to 1,186 per class
                   |
                   ▼
         ┌─────────────────────┐
         │  final_dataset.csv  │
         │  2,372 total rows   │
         │  1,186 disease (1)  │
         │  1,186 benign  (0)  │
         └─────────────────────┘
```

The most important thing: the AF_popmax > 0.005 filter is the scientific backbone of the benign class — without it, the training labels would be unreliable.

---

## PART 3 — UFold AND CONTACT MAPS

### What is a contact map?

An RNA contact map is a 2D matrix where entry [i, j] contains the probability that nucleotide i base-pairs with nucleotide j. For a hairpin of length L, the contact map is an L×L matrix. High values appear along a diagonal band that represents the stem base-pairs. The loop appears as a gap in that band. Bulges and mismatches appear as disruptions in the diagonal pattern. The contact map is a complete representation of RNA secondary structure as a continuous probability field.

### What is UFold?

UFold is a deep learning model based on U-Net architecture that predicts RNA contact maps from sequence alone. Unlike traditional RNA folding tools (RNAfold, mfold) that use thermodynamic energy minimisation, UFold was trained on known RNA structures and learns statistical patterns of base pairing directly. It takes a 17-channel tensor as input and produces a contact probability map as output.

### The 17-channel input tensor

To encode a sequence of length L for UFold, each possible pair (i, j) is represented. The first step is one-hot encoding — each nucleotide A, U, C, G becomes a 4-dimensional vector (1,0,0,0), (0,1,0,0), etc. The outer product of the one-hot vector at position i with the one-hot vector at position j gives a 4×4 = 16-dimensional encoding for each pair. This 16-channel representation captures which specific nucleotide combination exists at each position pair. The 17th channel is the canonical pairing mask — a binary map showing where Watson-Crick pairs (A-U, G-C) and wobble pairs (G-U) are geometrically possible based on sequence alone. This provides a hard constraint that UFold uses to bias its predictions toward chemically possible pairings.

### The 4-channel output tensor

DeepFold constructs a 4-channel tensor for each training sample, sized (4, 128, 128) — all maps are padded or cropped to 128×128:

- **C1 — Healthy contact map:** UFold's predicted contact probabilities for the wild-type sequence. Values typically range 0.3–0.8 in paired regions.
- **C2 — Mutant contact map:** UFold's predicted contact probabilities for the mutant sequence. Differs from C1 wherever the SNP disrupts pairing.
- **C3 — Absolute difference:** |C1 − C2|. This is the structural perturbation map. It shows exactly where the folding has changed. Values are small — typically 0.01–0.05 — because a single SNP changes the fold subtly, not catastrophically.
- **C4 — Canonical pairing mask:** Where base pairing is chemically possible. Acts as a structural prior for the CNN.

### Why the magnitude difference matters critically

C1 and C2 have values in the 0.3–0.8 range. C3 has values in the 0.01–0.05 range — about 10x smaller. In early CNN versions (v1, v2), all four channels were fed through the same convolutional layers. Because C1 and C2 dominated the gradient signal at 10x larger magnitude, C3 was effectively ignored — the gradient from the perturbation map was swamped. The model converged on classifying based on the absolute contact map rather than the change in contact map. This was the root cause of underperformance in v1 and v2.

### Diagram 3 — Contact Map Construction

```
Input Sequence (healthy, L=66 nt)
UGAGGUAGUAGGUUGUAUAGUU...
          |
          ▼
    UFold (U-Net)
    17-channel input tensor
          |
          ▼
  ┌─────────────────────┐
  │   HEALTHY MAP (C1)  │   ← values 0.3–0.8 in stem
  │   66×66 → 128×128   │
  │  ░░▓▓▓▓▓░░░░░░░░░░  │
  │  ░▓▓▓▓▓▓▓░░░░░░░░░  │
  │  ░▓▓░░░░▓▓░░░░░░░░  │   stem pairs appear
  │  ░░░░░░░░░░░░░░░░░  │   as diagonal bands
  └─────────────────────┘

Apply SNP: position 17, G → A
          |
          ▼
  ┌─────────────────────┐
  │   MUTANT MAP (C2)   │   ← values 0.3–0.8 in stem
  │   66×66 → 128×128   │     but break at pos 17
  │  ░░▓▓▓▓▓░░░░░░░░░░  │
  │  ░▓▓▓░▓▓▓░░░░░░░░░  │   ← gap at SNP site
  │  ░▓▓░░░░▓▓░░░░░░░░  │
  │  ░░░░░░░░░░░░░░░░░  │
  └─────────────────────┘

          |
          ▼ |C1 - C2|

  ┌─────────────────────────────────────────┐
  │         DIFFERENCE MAP (C3)             │
  │   values 0.01–0.05 (tiny but real)      │
  │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │
  │  ░░░░░▒░░░░░░░░░░░░░░░░░░░░░▒░░░░░░   │
  │  ░░░░░▒░░░░░░░░░░░░░░░░░░░░░▒░░░░░░   │  ← hot spots
  │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │    at SNP row/col
  └─────────────────────────────────────────┘
  (▒ = measurable change, ░ = near zero)
```

The most important thing: C3 is the diagnostic signal — the difference map is what reveals pathogenicity — but its 10x smaller magnitude means it must be handled separately from C1/C2.

---

## PART 4 — CNN v4

### Why CNN on contact maps?

A contact map is a 2D image-like structure. Convolutional neural networks are designed exactly for this — they detect local spatial patterns regardless of where in the image those patterns appear. A stem disruption at rows 10–20 and one at rows 50–60 should both activate the same pathogenicity detector. CNNs achieve this via weight sharing across spatial positions.

### The split-path design — what went wrong in v1/v2

In CNN v1 and v2, all four channels (C1, C2, C3, C4) were concatenated and processed through a single shared encoder. The problem: C3 has values ~10x smaller than C1 and C2. When the convolutional filters are applied, the gradient flowing back through C1 and C2 is ~10x larger than through C3. Weight updates centre on learning C1 and C2 patterns. C3 — the actual structural perturbation signal — contributes essentially nothing to the learned representation. The model was effectively classifying based on what the healthy hairpin looks like, not how it changed.

The fix in v4: **split-path input design.** C3 and C4 (the perturbation map and pairing mask) go through a dedicated path with 32 convolutional filters. C1 and C2 (the absolute maps) go through a separate path with 16 filters. Each path has its own normalisation statistics computed separately. The outputs are concatenated (giving 48 channels) before the shared encoder. This forces the network to learn a meaningful representation of the perturbation independently before merging.

### CoordAttention — why position matters

Standard convolutions are translation-invariant — they respond the same way regardless of where in the image a pattern appears. But RNA contact maps have biologically meaningful position. A structural change at position 5 of a hairpin (near the Drosha cut site at the base of the stem) is biologically different from a change at position 64 (deep in the loop). CoordAttention adds position awareness by performing horizontal pooling (averaging across each row) and vertical pooling (averaging across each column), then using these as spatial attention weights. The model learns which row/column positions are more informative for pathogenicity prediction.

### MultiScalePool — why one pooling scale is insufficient

After the convolutional encoder produces a (256, 8, 8) feature map, it needs to be collapsed into a vector for classification. A single AdaptiveAvgPool(4) (as in v3) produces a (256, 4, 4) = 4,096-dimensional vector but discards information at different spatial scales. RNA structural changes are spatially distributed — the perturbation at the SNP site propagates across the map. MultiScalePool runs three operations in parallel: global average pool → 256 dims, global max pool → 256 dims, and 2×2 spatial pool → 1,024 dims. These are concatenated into a 1,536-dimensional vector. The global average captures the overall disruption magnitude. The global max captures the peak disruption. The 2×2 spatial captures which quadrant of the hairpin is disrupted.

### Label smoothing and Mixup

**Label smoothing (ε=0.10):** Instead of training the model to output probability 1.0 for disease and 0.0 for benign, it trains toward 0.90 and 0.10 respectively. This is appropriate because COSMIC labels contain noise — some disease-labelled mutations may be passenger mutations. Hard 0/1 targets cause the model to overfit to noisy labels. Smoothed targets regularise the output distribution.

**Mixup (α=0.2):** Two training samples are interpolated: the contact map tensor and the label are both mixed at a random λ drawn from Beta(0.2, 0.2). The model trains on (λ·map1 + (1−λ)·map2) with label (λ·y1 + (1−λ)·y2). This works on contact maps because they are continuous probability tensors — interpolating two contact maps produces something that is biologically meaningful (a structure somewhere between the two). Mixup smooths the decision boundary and reduces overconfidence.

### Diagram 4 — CNN v4 Architecture

```
Input: (4, 128, 128)
Channels: [C1=healthy | C2=mutant | C3=diff | C4=mask]
                    |
         ┌──────────┴────────────┐
         │                       │
   C3+C4 (2ch)             C1+C2 (2ch)
   MAIN PATH                CTX PATH
   Conv2d → 32ch            Conv2d → 16ch
   BN + ReLU                BN + ReLU
         │                       │
         └──────────┬────────────┘
                    │ cat → 48ch
                    ▼
            CoordAttention(48)
            [H-pool × V-pool spatial weights]
                    │
            SEBlock(48)
            [channel reweighting]
                    │
            Dropout2d + MaxPool2d(2)
            → (48, 64, 64)
                    │
            ResidualBlock(48→64) + SE
            → (64, 64, 64)
                    │
            MaxPool2d(2) → (64, 32, 32)
                    │
            ResidualBlock(64→128) + SE
            → (128, 32, 32)
                    │
            MaxPool2d(2) → (128, 16, 16)
                    │
            ResidualBlock(128→256) + SE
            → (256, 16, 16)
                    │
            MaxPool2d(2) → (256, 8, 8)
                    │
         ┌──────────┼──────────┐
    AvgPool(1)  MaxPool(1)  AvgPool(2)
      → 256      → 256      → 1024
         └──────────┼──────────┘
                    │ cat → 1536
                    ▼
             Linear(1536→512) + ReLU + Drop(0.4)
             Linear(512→64)   + ReLU + Drop(0.2)
             Linear(64→2)
                    │
              Softmax → P(pathogenic)
```

The most important thing: the split-path design exists specifically because C3 has 10x smaller magnitude than C1/C2 and would be ignored by a shared encoder.

---

## PART 5 — SIAMESE GCN

### Why represent RNA as a graph?

A contact map captures base-pairing probabilities but treats the structure as a 2D image. A graph represents the structure more naturally — each nucleotide is a node, and edges represent actual relationships. This representation is explicitly structural and does not depend on positional indexing the way a 2D image does. Graph Neural Networks operate directly on this topology and are invariant to graph isomorphisms, which is appropriate for RNA structure.

### What the nodes and edges are

**Nodes:** Each nucleotide is a node with a 6-dimensional feature vector: a 4-dimensional one-hot encoding of the base (A, U, C, G), one feature for normalised position within the hairpin (position / length), and one binary feature indicating whether it is in a confident base-pair (mask). 6 features total per node.

**Edges:** Three types of edges are added:
1. **UFold contact edges:** If UFold assigns contact probability > 0.5 to the pair (i, j), an edge is drawn. These are the predicted structural base pairs.
2. **Backbone edges:** Every adjacent pair (i, i+1) is connected automatically. This captures the RNA backbone connectivity.
3. **Self-loops:** Every node connects to itself. This is standard in graph neural networks to allow a node to retain its own features during message passing.

The adjacency matrix is normalised using the symmetric normalisation D^{-1/2} A D^{-1/2} where D is the degree matrix. This prevents high-degree nodes from dominating.

### Why Siamese with shared weights

A Siamese architecture runs two copies of the same GCN encoder, but crucially, both copies share the same weights. The healthy graph and the mutant graph are processed in parallel by one network.

If two separate encoders with independent weights were used, each would learn its own representation system for healthy and mutant sequences. When you subtract the embeddings, the difference would be meaningless because the two networks learned in different coordinate spaces. Shared weights force both healthy and mutant graphs to be encoded in the same representation space. Now the difference between embeddings is semantically meaningful — it measures the structural change in a common coordinate system.

### The 4-way interaction

After the shared GCN encodes both graphs to embedding vectors h (healthy) and m (mutant), the classifier receives their concatenation:

**[h, m, |h−m|, h×m]**

Each term captures something different:
- **h** — the absolute representation of the healthy structure
- **m** — the absolute representation of the mutant structure  
- **|h−m|** — the magnitude of the change at each dimension; this is the primary pathogenicity signal
- **h×m** — the elementwise product; this captures which dimensions are co-active (large in both), which detects structural features that are preserved versus those that are altered

Together these four views give the classifier access to the full relationship between the two structures. Using only |h−m| would miss information about what the structures are in absolute terms.

### Diagram 5 — Siamese GCN

```
Healthy Sequence              Mutant Sequence
UGAGGUAGUAGG...               UGAGGUAUUAGG...
     |                               |
     ▼                               ▼
Build Graph                    Build Graph
Nodes: nucleotides (6-dim)     Nodes: nucleotides (6-dim)
Edges: UFold >0.5 + backbone   Edges: UFold >0.5 + backbone

┌─────────────────┐           ┌─────────────────┐
│  Healthy Graph  │           │  Mutant Graph   │
│  N nodes, E1 e  │           │  N nodes, E2 e  │
└────────┬────────┘           └────────┬────────┘
         │                            │
         ▼                            ▼
  ┌─────────────────────────────────────────┐
  │      SHARED GCN ENCODER (same weights) │
  │   GCN Layer 1: 6 → 64                  │
  │   GCN Layer 2: 64 → 128                │
  │   Global Mean Pool → 128-dim vector     │
  └──────────┬──────────────────┬──────────┘
             │                  │
             ▼                  ▼
        h (128-dim)        m (128-dim)
             │                  │
             └────────┬─────────┘
                      ▼
          Compose interaction vector:
          [h | m | |h−m| | h×m]
          = 128+128+128+128 = 512-dim
                      │
                      ▼
          Linear(512→128) + ReLU
          Linear(128→2)
                      │
                 P(pathogenic)
```

The most important thing: shared weights are not an optimisation — they are a scientific requirement for the embedding difference to be semantically meaningful.

---

## PART 6 — THREE XGBOOST MODELS

### Why three separate XGBoost models?

Each model captures a fundamentally different aspect of the mutation's impact. They are designed to be as orthogonal as possible so that the meta-learner gains new information from each one.

### k-mer XGBoost — what the sequence composition reveals

k-mers are all possible subsequences of length k. For each sequence, the frequency of every possible k-mer is computed and normalised. The feature vector contains:

- k=3 healthy frequencies: 4³ = 64 dimensions
- k=3 mutant frequencies: 64 dimensions  
- k=3 difference (healthy − mutant): 64 dimensions
- k=4 healthy frequencies: 4⁴ = 256 dimensions
- k=4 mutant frequencies: 256 dimensions
- k=4 difference: 256 dimensions
- k=5 difference only: 4⁵ = 1,024 dimensions
- Positional features (GC content, length, SNP position, distance from centre): 4 dimensions
- Seed region features (is SNP in seed, seed GC, seed disruption): 4 dimensions
- Conservation proxy (local GC, AU/GU ratio, entropy): 3 dimensions

**Why k=5 uses only the difference and not healthy + mutant separately:** A single SNP changes at most 5 five-mers (a k-mer is affected if the SNP falls within any of the k positions). The absolute frequency distributions for healthy and mutant differ in at most 10 of the 1,024 dimensions. Encoding healthy and mutant separately adds 1,024 dimensions of near-identical information that confuses the model. The difference vector directly encodes what changed.

Total: approximately 2,000 dimensions. AUC standalone: **0.704**. This is the strongest standalone model.

### MFE XGBoost — what thermodynamics reveals

MFE (Minimum Free Energy) is the most stable secondary structure predicted by energy minimisation (RNAfold). A more negative MFE means a more stable structure. The key features include:
- MFE of healthy vs mutant sequences
- delta_mfe = MFE(mutant) − MFE(healthy): a positive value means the mutation destabilised the structure
- Ensemble diversity (how many competing structures exist)
- Base pair probabilities for specific positions
- Stem length, loop length, GC content of stem

The key finding: disease SNPs cause an average delta_mfe of +1.42 kcal/mol (more destabilising) versus benign SNPs at +1.14 kcal/mol. The difference is real but small relative to the variance, which is why MFE standalone AUC is only **0.633**.

### RNA-FM XGBoost — what evolution reveals

RNA-FM is a large transformer model pre-trained on 23 million non-coding RNA sequences from databases like RNAcentral. It learned the statistical language of RNA — which sequences are evolutionarily plausible, which motifs are conserved, which positions co-evolve. It produces a 640-dimensional embedding for each sequence.

**Why frozen embeddings rather than fine-tuning:** Fine-tuning RNA-FM on 2,372 samples would cause catastrophic forgetting — the pre-trained evolutionary knowledge would be overwritten by the noise in the small dataset. The 640-dimensional frozen embedding is already a rich representation of evolutionary plausibility. Using it as-is preserves all 23M sequences worth of information.

**Why the difference embedding is more useful than raw embeddings:** A healthy pre-miRNA embedding captures the general properties of that miRNA family — its GC content, its evolutionary context, its typical structure. This information is the same for both healthy and mutant. The difference embedding (healthy embedding − mutant embedding) captures specifically how the embedding space changes when the SNP is applied — this is the mutation's evolutionary footprint.

### Optuna tuning — why it must only see the training fold

Optuna is a Bayesian hyperparameter optimisation framework. For each outer fold, it runs 50 trials searching for the best XGBoost hyperparameters. Each trial is evaluated using an inner 3-fold cross-validation on the training data only. The test fold is never seen by Optuna. If Optuna saw the test fold, it would search through hyperparameter space until it found settings that happen to perform well on that test fold — this is data leakage disguised as cross-validation. The reported AUC would be optimistically biased.

### Diagram 6 — Three Parallel Feature Extractors

```
       Healthy Sequence              Mutant Sequence
       UGAGGUAGUAGG...               UGAGGUAUUAGG...
              │                             │
     ┌────────┼─────────────────────────────┤
     │        │                             │
     ▼        ▼                             ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│   k-mer          │  │   MFE / RNAfold  │  │   RNA-FM Transformer │
│   Feature        │  │   Thermodynamics │  │   (frozen, 23M pre   │
│   Extraction     │  │                  │  │    training seqs)    │
│                  │  │                  │  │                      │
│ k3 h, m, diff    │  │ MFE healthy      │  │ Embed healthy → 640d │
│ k4 h, m, diff    │  │ MFE mutant       │  │ Embed mutant  → 640d │
│ k5 diff only     │  │ delta_mfe        │  │ diff → 640d          │
│ pos features     │  │ stem length      │  │                      │
│ seed features    │  │ loop size        │  │                      │
│ conservation     │  │ ...12 total      │  │                      │
└────────┬─────────┘  └────────┬─────────┘  └──────────┬───────────┘
         │                     │                        │
         ▼                     ▼                        ▼
    ~2,000 dims            12 dims                  640 dims
         │                     │                        │
         ▼                     ▼                        ▼
    XGBoost              XGBoost                  XGBoost
    (Optuna tuned)      (Optuna tuned)            (Optuna tuned)
         │                     │                        │
         ▼                     ▼                        ▼
    P(path)=0.72         P(path)=0.58             P(path)=0.65
    AUC 0.704            AUC 0.633                AUC ~0.660
```

The most important thing: the three models are intentionally designed to capture orthogonal signals so that combining them adds genuine new information.

---

## PART 7 — ENSEMBLE STACKING

### What stacking is and why it beats averaging

Simple averaging of 5 model probabilities assumes all models are equally good and make independent errors. Stacking learns how to combine the models optimally. It trains a meta-learner to predict the true label given the predictions of all five base models. If model A is always wrong when model B is confident, the meta-learner learns to downweight A when B is high. Averaging cannot capture this.

### Out-of-fold predictions — why they are essential

If you train the CNN, GCN, and XGBoost models on the full dataset, then collect their predictions on the full dataset, and use those predictions to train the meta-learner — you have data leakage. Each base model already saw every sample during training, so its predictions on those samples are artificially good. The meta-learner trains on these overfitted predictions and learns a combination rule that is useless on new data.

Out-of-fold (OOF) predictions fix this. For each fold, the base model is trained on 80% of the data and its predictions are collected on the remaining 20% (which it has never seen). After 5 folds, every sample has exactly one OOF prediction — generated by a model that never trained on that sample. The meta-learner trains on these honest predictions.

### Rank normalisation — why raw probabilities cannot be stacked directly

The CNN outputs probabilities calibrated to its own learned decision boundary. The MFE XGBoost outputs probabilities calibrated to a completely different learned boundary. A CNN score of 0.65 and an MFE score of 0.65 do not mean the same thing. Directly stacking them teaches the meta-learner that these numbers are on the same scale, which is wrong.

Rank normalisation maps each model's predictions to a uniform distribution using `scipy.stats.rankdata`. A score of 0.999 (raw probability) becomes 0.95 (rank percentile if it's in the top 5%). Now all five inputs to the meta-learner are on the same [0,1] uniform scale. The meta-learner learns from relative ranks, not absolute probability values.

### Why XGBoost as meta-learner over logistic regression

Logistic regression can only model linear combinations of the base model predictions. It can learn "weight CNN by 0.3 and kmer-XGB by 0.4". But the optimal combination is non-linear: "trust kmer-XGB when RNA-FM is also high, but downweight kmer-XGB when MFE is strongly negative." This is an interaction that requires a non-linear model. XGBoost captures these interactions through its tree structure.

### Weight distribution insight

The meta-learner implicitly assigns influence to each base model. From feature importance analysis:
- **k-mer XGB:** dominant — sequence composition is the strongest single signal
- **MFE XGB:** second highest weight despite its low standalone AUC — it provides information the sequence models do not have
- **Siamese GCN:** moderate — graph topology adds independent structural signal
- **CNN v4:** lower weight — partially redundant with GCN (both model structure) but still contributes
- **RNA-FM XGB:** contributes orthogonal evolutionary context

### Diagram 7 — Ensemble Stacking

```
Input: healthy + mutant sequences
              │
    ┌─────────┼─────────────┬─────────────┬─────────────┐
    │         │             │             │             │
    ▼         ▼             ▼             ▼             ▼
  CNN v4    Siamese      k-mer          MFE           RNA-FM
  (5 fold   GCN          XGBoost        XGBoost       XGBoost
   avg)     (5 fold)     (5 fold)       (5 fold)      (5 fold)
    │         │             │             │             │
    ▼         ▼             ▼             ▼             ▼
  P=0.62    P=0.58        P=0.79         P=0.51        P=0.68
  (raw)     (raw)         (raw)          (raw)         (raw)
    │         │             │             │             │
    └─────────┴─────────────┴─────────────┴─────────────┘
                            │
                     Rank Normalise
                   (scipy.stats.rankdata)
                     → [0.55, 0.48, 0.92, 0.41, 0.71]
                            │
                            ▼
                  XGBoost Meta-Learner
                  (trained on OOF preds)
                  learns non-linear combination
                            │
                            ▼
                    Final P(pathogenic) = 0.73
                            │
              ┌─────────────┼──────────────────┐
              ▼             ▼                  ▼
         Label:        Confidence:        Ensemble AUC:
       Pathogenic         Medium             0.7338
```

The most important thing: out-of-fold predictions are not an optimisation trick — they are the only way to get an honest meta-learner.

---

## PART 8 — EVALUATION

### Why StratifiedGroupKFold grouped by miRNA family is essential

miRNA families share sequence similarity. hsa-mir-21 and hsa-mir-21b share long stretches of identical sequence. If one appears in the training fold and the other in the test fold, the model has effectively already seen the test sequence during training. It will memorise rather than generalise.

StratifiedGroupKFold groups all members of the same miRNA family together and ensures the entire group goes either into training or into testing — never split across both. The "Stratified" part ensures each fold maintains the 50/50 class ratio despite the family constraint.

If standard StratifiedKFold (without grouping) were used, similar sequences would appear in both train and test. The reported AUC would likely inflate by 0.05–0.10 — appearing as 0.78–0.82 instead of 0.7338. This would be a scientifically dishonest result.

### What AUC 0.7338 means in plain language

AUC-ROC (Area Under the Receiver Operating Characteristic Curve) measures how well the model ranks disease variants above benign variants. An AUC of 0.7338 means: if you take one random disease variant and one random benign variant from the test set, the model correctly ranks the disease variant as more pathogenic 73.38% of the time. Random chance is 50%. Perfect discrimination is 100%. 73.38% is meaningful — it captures real biological signal — but it is not sufficient for clinical deployment without expert review.

### Why the AUC ceiling exists

Four factors create the ceiling at 0.720–0.734:

1. **Dataset size:** 2,372 samples across 5 folds gives ~475 test samples per fold. Statistical variance at this size is large (typically ±0.02).
2. **COSMIC label noise:** Some somatic mutations in COSMIC are passengers — they happened in cancer cells but did not cause cancer. The model cannot distinguish these from truly pathogenic variants because nothing in the input signal separates them.
3. **UFold accuracy:** The contact maps themselves are predictions, not ground truth. UFold has its own error rate, and wrong contact maps produce wrong structural signals.
4. **Fundamental difficulty:** Some SNPs are genuinely borderline — their functional effect depends on cellular context, tissue type, and interacting proteins, none of which are captured by sequence features alone.

To push past 0.75 would require: 10,000+ independently validated variants, PhyloP/GERP conservation scores, fine-tuned RNA-FM, or multi-tissue experimental validation data.

### Diagram 8 — StratifiedGroupKFold by miRNA Family

```
All 2,372 samples grouped by miRNA family:

Family A: mir-21  (47 samples) ──────► FOLD 1 TEST
Family B: mir-155 (31 samples) ──────► FOLD 1 TEST
Family C: mir-17  (28 samples) ──────► FOLD 2 TEST
Family D: mir-34a (22 samples) ──────► FOLD 3 TEST
...

FOLD 1:
  TRAIN: Families C, D, E, F, G... (≈1,900 samples)
   TEST: Families A, B            (≈475 samples)
   → NO overlap in family membership

FOLD 2:
  TRAIN: Families A, B, D, E...   (≈1,900 samples)
   TEST: Family C                 (≈475 samples)

... and so on for 5 folds.

KEY CONSTRAINT:
  mir-21 appears in TRAIN or TEST — never both
  mir-21b (same family) always goes with mir-21
  ✓ Prevents sequence similarity leakage
  ✗ Standard KFold would split families freely → inflated AUC
```

The most important thing: grouping by miRNA family is not optional — without it, the reported AUC would be dishonestly inflated by similarity-based memorisation.

---

## PART 9 — KEY FINDINGS

### Finding 1 — Sequence composition beats structural disruption

The k-mer XGBoost (AUC 0.704) outperforms the CNN on contact maps (AUC 0.625) as a standalone model. This is counterintuitive — we built the CNN specifically to detect structural disruption, which is the biological mechanism. But what this tells us is that the compositional change introduced by an SNP — which sequence motifs are created or destroyed — is more statistically discriminative than the predicted structural change. Biologically, this makes sense: certain k-mer patterns are specifically associated with Drosha/Dicer recognition sequences, and introducing or removing those patterns is a stronger pathogenicity signal than the subtle difference in predicted contact maps.

### Finding 2 — The seed region paradox

The seed region of a miRNA (positions 2–8) is its most functionally critical part — it is what determines which mRNAs are targeted. Intuition says mutations here are most dangerous. But analysis shows that gnomAD benign variants appear MORE frequently in seed positions (11.8% of benign SNPs) than COSMIC disease variants (9.6% of disease SNPs). The population genetics explanation: if a mutation in the seed region were truly pathogenic, it would be under strong negative selection and would never accumulate to AF_popmax > 0.5% in human populations. The gnomAD variants in seed positions are there precisely because they were tolerated by natural selection. Truly pathogenic mutations in the seed were eliminated before they could reach high population frequency. Disease SNPs cluster instead in stem positions where structural disruption occurs.

### Finding 3 — Thermodynamics adds orthogonal value

MFE XGBoost has the weakest standalone AUC (0.633) but receives a disproportionately high weight in the ensemble. This reveals that thermodynamic features capture real information that composition features do not. A variant can have a neutrally-looking k-mer profile but still measurably destabilise the hairpin. The average delta_mfe for disease variants (+1.42 kcal/mol) versus benign (+1.14 kcal/mol) is a small but statistically real difference. Including this signal in the ensemble improves overall performance beyond what composition and structure models achieve alone.

### Finding 4 — The AUC ceiling is a statement about the problem's difficulty

The ceiling at 0.720–0.734 is scientifically honest. It tells us that with the current data and features, there is a limit to how much of the signal can be captured. The ceiling is not a failure of architecture — it is a reflection of the problem's fundamental difficulty. Some variants are genuinely uncertain, and no model can classify them correctly without additional data types (conservation scores, experimental measurements, tissue expression context). Pushing past this ceiling requires more data and more signals, not a better neural network.

---

## PART 10 — SYSTEM ARCHITECTURE

### End-to-end flow

**User input:** The researcher opens the frontend, selects a miRNA ID (or pastes a FASTA sequence), selects an SNP position, and clicks predict. The frontend sends a POST request to the backend with JSON containing the miRNA ID, the healthy sequence, the mutant sequence, and the SNP index.

**Backend validation:** FastAPI validates the request. It checks sequence length (15–300 nt), nucleotide charset (AUCTG only), that sequences are the same length, that they differ by exactly 1 nucleotide, and applies rate limiting (15 requests/min per IP).

**Contact map generation:** UFold generates contact maps for both sequences. The 4-channel tensor (C1, C2, C3, C4) is constructed at (4, 128, 128).

**Parallel inference:** All 5 model groups run simultaneously. The CNN processes the 4-channel tensor through the split-path architecture. The Siamese GCN builds two graphs and runs them through the shared encoder. k-mer, MFE, and RNA-FM extractors compute their feature vectors. Each model group averages over its 5 trained fold models.

**Rank normalisation + meta-learner:** The 5 raw probabilities are rank-normalised, then the XGBoost meta-learner produces a final pathogenicity probability.

**SHAP explanation:** The k-mer XGBoost fold 1 model is used as a proxy for SHAP computation (full SHAP on all models is too slow for real-time). TreeExplainer computes Shapley values for the top features, which are mapped to biological labels.

**Uncertainty quantification:** The standard deviation across the 5 sub-model probabilities is computed as the disagreement score. The miRNA family is checked against the training set index for OOD detection.

**Response:** JSON with probability, label, confidence, base model breakdown, processing time, SHAP explanation, disagreement score, and OOD flag. Frontend renders this into the diagnostic dashboard with the 3D structural visualiser, SHAP bar chart, and model agreement indicator.

### Diagram 9 — Full System

```
┌────────────────────────────────────────────────────────┐
│                   BROWSER (Next.js)                    │
│  User selects hsa-mir-21, SNP pos 17                   │
│  → POST /predict {mirna_id, seq_h, seq_m, snp_pos}     │
└─────────────────────────┬──────────────────────────────┘
                          │ HTTP JSON
                          ▼
┌─────────────────────────────────────────────────────────┐
│               FastAPI Backend (port 8088)                │
│                                                         │
│  1. Input Validation                                    │
│     ✓ length 15–300 nt                                  │
│     ✓ charset AUCTG                                     │
│     ✓ exactly 1 mutation                                │
│     ✓ rate limit check                                  │
│                      │                                  │
│  2. Contact Map Generation (~300ms)                     │
│     UFold → C1, C2, C3, C4 → (4, 128, 128) tensor      │
│                      │                                  │
│  3. Parallel Inference                                  │
│     ┌────────┬────────┬────────┬────────┬────────┐     │
│     │CNN v4  │GCN     │k-mer   │MFE     │RNA-FM  │     │
│     │5 folds │5 folds │XGB×5   │XGB×5   │XGB×5   │     │
│     │~400ms  │~200ms  │~50ms   │~100ms  │~800ms  │     │
│     │P=0.62  │P=0.58  │P=0.79  │P=0.51  │P=0.68  │     │
│     └────────┴────────┴────────┴────────┴────────┘     │
│                      │                                  │
│  4. Rank Normalise → [0.55, 0.48, 0.92, 0.41, 0.71]   │
│                      │                                  │
│  5. XGBoost Meta-Learner → P(pathogenic) = 0.73         │
│                      │                                  │
│  6. SHAP Explanation (proxy on kmer fold1, ~100ms)      │
│                      │                                  │
│  7. Uncertainty: std([0.62,0.58,0.79,0.51,0.68])=0.10  │
│     OOD: check mirna_id in training set index           │
│                      │                                  │
│  Response JSON: prob, label, SHAP, UQ, OOD              │
└─────────────────────────┬───────────────────────────────┘
                          │ HTTP JSON
                          ▼
┌────────────────────────────────────────────────────────┐
│               FRONTEND DISPLAY                         │
│                                                        │
│  ┌──────────────┐  ┌────────────────┐  ┌───────────┐  │
│  │ 3D RNA       │  │ Result Panel   │  │ SHAP      │  │
│  │ Hairpin      │  │ P=0.73         │  │ Bar Chart │  │
│  │ (Three.js)   │  │ PATHOGENIC     │  │ Top feats │  │
│  │ SNP site     │  │ Medium conf.   │  │           │  │
│  │ glows red    │  │ Agree: LOW     │  │           │  │
│  └──────────────┘  └────────────────┘  └───────────┘  │
│                                                        │
│  + Research page: ClinVar truth table                  │
│  + Dashboard: Calibration, PR curves, model AUCs       │
│  + Batch: CSV upload → 500 variants                    │
└────────────────────────────────────────────────────────┘
```

The most important thing: the ensemble architecture is not just about accuracy — it is about scientific credibility, because each model interrogates the mutation from a different biological perspective.

---

## SUMMARY TABLE

| Component | Input | Output | Standalone AUC | Role in Ensemble |
|---|---|---|---|---|
| CNN v4 | (4,128,128) contact tensor | P(path) | ~0.625 | Structural perturbation image |
| Siamese GCN | Two RNA graphs | P(path) | ~0.640 | Topological structure change |
| k-mer XGBoost | 2,000-dim freq vector | P(path) | 0.704 | Dominant sequence motif signal |
| MFE XGBoost | 12 thermo features | P(path) | 0.633 | Orthogonal stability signal |
| RNA-FM XGBoost | 640-dim diff embedding | P(path) | ~0.660 | Evolutionary context |
| **Meta-Learner** | **5 rank-normalised scores** | **P(path)** | **0.7338** | **Final ensemble** |

---

*DeepFold is not a clinical diagnostic tool — it is a research-grade variant interpretation aid designed to prioritise variants for experimental validation.*
