# DeepFold: Classifying Disease-Associated miRNA SNPs Using UFold Contact Maps — Full Explanation

## PROJECT OVERVIEW

DeepFold is a bioinformatics + deep learning pipeline that classifies whether a single nucleotide polymorphism (SNP) located inside a pre-miRNA hairpin region is disease-associated or benign. It uses RNA secondary structure predictions from UFold (a deep learning RNA folding model) as input features for a convolutional neural network and several other classifiers.

The pipeline runs on Kaggle with a Tesla T4 GPU. The reference genome is GRCh38 (hg38).

---

## DATASET EXPLANATION

### What are miRNAs and pre-miRNAs?

MicroRNAs (miRNAs) are short (~22 nucleotide) non-coding RNA molecules that regulate gene expression by binding to target mRNAs and silencing them. Before becoming mature miRNAs, they exist as precursor hairpin structures called pre-miRNAs (~60-120 nucleotides long). These hairpins fold into a characteristic stem-loop secondary structure. SNPs (single base changes) in these hairpin regions can disrupt the RNA folding and are associated with diseases including cancer.

### Data Sources

**Disease SNPs — COSMIC (Catalogue of Somatic Mutations in Cancer):**
- 6,600 raw SNPs located within human pre-miRNA hairpin regions
- These are somatic mutations observed in cancer patients
- After deduplication (removing duplicate variant IDs where multiple cancers map to the same SNP): 6,512 unique disease SNPs
- Each record contains: chromosome, position, ref allele, alt allele, variation_id, pre_mirna name, source, region, mature_mirna name

**Benign SNPs — gnomAD v3.1.2 (Genome Aggregation Database):**
- gnomAD v3.1.2 contains variants from 76,156 whole genomes across diverse global populations
- Why gnomAD instead of 1000 Genomes: 1000 Genomes (2,504 genomes) only yielded ~537 benign SNPs in pre-miRNA windows after MAF filtering — too few for training
- Extraction method: bcftools streams directly from Google Cloud Storage HTTPS URLs using tabix indices, querying only miRNA regions per chromosome (avoids downloading 10-30 GB per chromosome VCF)
- Filter: AF_popmax > 0.005 — keeps SNPs where the maximum allele frequency across ANY single population is ≥ 0.5%. This is the scientifically correct definition of a "benign common variant" — a variant tolerated at population level even if globally rare
- Why not use a lower threshold: Testing AF_popmax > 0.001 and 0.0005 degraded AUC from 0.673 to 0.632 because rarer variants have noisier benign labels (some may actually be pathogenic)

**miRBase v22.1:**
- Provides hairpin FASTA sequences for 1,917 human pre-miRNAs (the actual RNA sequences of the hairpin structures)
- Provides GFF3 annotation with genomic coordinates, strand information, and mature miRNA positions within each hairpin
- Used to map genomic SNP positions to positions within the hairpin sequence

### Dataset Construction

1. Disease SNPs are loaded from COSMIC CSV, chromosome names normalised (strip "chr" prefix), alleles uppercased, deduplicated → 6,512 unique disease SNPs
2. Benign SNPs are extracted from gnomAD by streaming VCF files for each chromosome, filtering for biallelic SNPs with AF_popmax > 0.005 in miRNA regions
3. Any gnomAD variant that also appears in COSMIC is removed (exact match on chrom+position+ref+alt) to ensure clean label separation
4. Each SNP is mapped to a relative position within its hairpin using GFF3 coordinates:
   - Plus strand: relative_pos = genomic_pos − hairpin_start + 1
   - Minus strand: relative_pos = hairpin_end − genomic_pos + 1 (reverse complement)
5. Reference allele validation: every mapped SNP's reference allele is checked against the miRBase FASTA — zero mismatches required
6. For each SNP, a healthy sequence (original hairpin) and mutant sequence (with the SNP applied) are generated. Strand-aware allele conversion is applied for minus-strand miRNAs
7. Benign samples get Label=0, disease samples get Label=1
8. Dataset is balanced by undersampling the majority class → 1,186 per class = 2,372 total samples
9. Integrity checks: no duplicate (miRNA_ID, rsID) pairs, perfect class balance, no sequence leakage (healthy ≠ mutant), exactly 1 nucleotide difference per sample

### Final Dataset Schema

Each row in final_dataset.csv contains:
- Sample_ID: unique identifier (e.g., "hsa-mir-608_0" for benign, "hsa-mir-608_disease_42" for disease)
- miRNA_ID: the pre-miRNA name (e.g., "hsa-mir-608")
- rsID: variant identifier (COSMIC variation_id for disease, "chrom:position" for benign)
- Seq_Healthy: the original hairpin RNA sequence (using U not T)
- Seq_Mutant: the hairpin with the SNP applied (exactly 1 base different)
- Label: 0 = benign, 1 = disease

Total: 2,372 samples (1,186 benign + 1,186 disease)

---

## FULL CODE EXPLANATION — STEP BY STEP

### STEP 1: ENVIRONMENT SETUP

**Cell 1 — Create directory structure:**
Creates folders: DeepFold_Dataset/raw_data, sequences/healthy, sequences/mutated, processed_maps/npy. These store intermediate and final outputs.

**Cell 2 — GPU verification:**
Checks torch.cuda.is_available(), prints device name (Tesla T4) and VRAM (15.6 GB). GPU is required for UFold inference and CNN training.

**Cell 3 — Build htslib + bcftools from source:**
Kaggle's default bcftools lacks libcurl support and cannot stream remote HTTPS VCF files. This cell:
- Runs apt-get update with retry logic (3 attempts, 10s backoff)
- Installs build dependencies (libcurl4-openssl-dev, libssl-dev, zlib, bz2, lzma, autoconf, make, gcc)
- Downloads htslib 1.18 source, configures with --enable-libcurl, compiles and installs
- Downloads bcftools 1.18 source, configures with --with-htslib=/usr/local, compiles and installs
- Skips rebuild if bcftools 1.18 is already installed (idempotent)
- Build time: ~3-5 minutes

---

### STEP 2: LOAD DISEASE SNPs (COSMIC)

**Cell 4 — Load raw disease SNPs:**
```python
disease_raw = pd.read_csv(disease_path)  # 6,600 rows
disease_raw.columns = disease_raw.columns.str.lower().str.strip()
```
Loads the COSMIC pre-miRNA disease SNP CSV. Standardises column names to lowercase.

**Cell 5 — Normalise and deduplicate:**
- Strips "chr" prefix from chromosome column for consistency with gnomAD naming
- Converts position to int, ref/alt to uppercase stripped strings
- Deduplicates on (chr, position, ref, alt) — multiple cancer types can map to the same genomic variant
- Result: 6,512 unique disease SNPs saved to DeepFold_Dataset/raw_data/disease_snps.csv

---

### STEP 3: EXTRACT BENIGN SNPs (gnomAD v3.1.2)

**Cell 6 — Install bedtools:**
Simple apt-get install for downstream region operations.

**Cell 7 — Build miRNA BED file from GFF3:**
```python
gff = pd.read_csv("hsa.gff3", sep="\t", comment="#", header=None)
mirna_regions = gff[gff["type"] == "miRNA_primary_transcript"].copy()
mirna_regions["start"] = mirna_regions["start"] - 1  # GFF3 1-based → BED 0-based
```
Parses miRBase v22.1 GFF3 to extract 1,918 pre-miRNA genomic coordinates. Converts from 1-based (GFF3) to 0-based (BED) start positions. Strips "chr" prefix. Saves as miRNA.bed — this tells bcftools exactly which genomic regions to query.

**Cell 8 — Stream benign SNPs from gnomAD:**
For each of 23 chromosomes (1-22 + X):
1. Builds a comma-separated region string from the BED file (e.g., "chr1:17300-17450,chr1:187800-187960,...")
2. Runs bcftools view with filters:
   - `-r region_str` — query only miRNA regions (not the whole chromosome)
   - `-v snps` — SNPs only, no indels
   - `-m2 -M2` — biallelic variants only
   - `-i 'AF_popmax>0.005'` — MAF ≥ 0.5% in at least one population
3. Streams directly from the GCS HTTPS URL using the tabix index
4. Appends non-header lines to miRNA_all_snps_body.txt
5. Cleans up per-chromosome VCF files

**Cell 9 — Remove disease overlap from benign set:**
```python
merged = gnomad_raw.merge(disease_df[["chr","position","ref","alt"]], how="left", indicator=True)
benign_clean = merged[merged["_merge"] == "left_only"]
```
Left-joins gnomAD variants with disease SNPs; keeps only gnomAD-only variants (no disease overlap). Deduplicates on (chrom, position, ref, alt). Saves to DeepFold_Dataset/raw_data/benign_snps.csv.

---

### STEP 4: MAP SNPs TO HAIRPIN POSITIONS

**Cell 10 — Install biopython:**
pip install biopython for FASTA parsing.

**Cell 11 — Load hairpin FASTA:**
```python
for record in SeqIO.parse("hairpin.fa", "fasta"):
    if record.id.startswith("hsa-"):
        hairpin_dict[record.id] = str(record.seq)
```
Loads 1,917 human pre-miRNA hairpin sequences into a dictionary keyed by miRNA ID (e.g., "hsa-mir-21").

**Cell 12 — Map SNP genomic coordinates to hairpin-relative positions:**
For each benign SNP:
1. Find which miRNA region(s) overlap the SNP's genomic position using the GFF3 coordinates
2. Calculate relative position within the hairpin:
   - Plus strand (+): relative_pos = genomic_pos − hairpin_start + 1
   - Minus strand (−): relative_pos = hairpin_end − genomic_pos + 1
3. Store: mirna_id, chrom, genomic_position, relative_position, strand, ref, alt
Saves to mapped_snps.csv.

**Cell 13 — Reference allele validation:**
```python
complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
for each mapped SNP:
    seq_base = hairpin_sequence[relative_pos - 1]
    if strand == "-": ref = complement[ref]  # reverse complement for minus strand
    ref_rna = ref.replace("T", "U")  # compare in RNA space
    assert seq_base == ref_rna
```
Critical quality check: verifies that the reference allele from the VCF matches the nucleotide at the mapped position in the miRBase FASTA. All comparisons in RNA space (T→U since hairpin sequences use uracil). Zero mismatches required — any mismatch indicates a coordinate system bug.

---

### STEP 5: GENERATE MUTANT SEQUENCES AND BUILD DATASET

**Cell 14 — Generate healthy/mutant sequence pairs (benign SNPs):**
For each mapped benign SNP:
1. Look up the hairpin sequence from hairpin_dict
2. Validate bounds (relative_pos within sequence length)
3. Apply strand-aware allele conversion: for minus-strand miRNAs, both ref and alt are reverse-complemented
4. Verify the reference allele matches the FASTA at that position
5. Substitute the alternate allele at the SNP position → mutant sequence
6. Convert both sequences to RNA (T→U)
7. Save as separate FASTA files (sequences/healthy/ID.fasta, sequences/mutated/ID.fasta)
8. Log any skipped samples with reasons (hairpin_not_found, position_out_of_bounds, ref_mismatch)

**Cell 15 — Strict positional validation:**
Spot-checks 20 generated pairs:
- Asserts exactly 1 nucleotide difference between healthy and mutant
- Asserts the difference is at precisely the expected relative position
Catches any off-by-one errors.

**Cell 16 — Build benign sample records (Label=0):**
Reads each healthy/mutant FASTA pair, assembles into a DataFrame with columns: Sample_ID, miRNA_ID, rsID (chrom:position format), Seq_Healthy, Seq_Mutant, Label=0.

**Cell 17 — Add disease samples (Label=1):**
Similar process for COSMIC disease SNPs, but:
- Uses the `pre_mirna` column directly from COSMIC for hairpin lookup (more reliable than GFF3 Name= attribute for multi-locus names like "hsa-mir-3158-1")
- Gets relative position from GFF3 overlap
- Same strand-aware allele conversion, ref validation, and mutation application
- Label=1

**Cell 18 — Merge, balance, and save:**
```python
full_df = pd.concat([benign_master, disease_master])
full_df = full_df.drop_duplicates(subset=["miRNA_ID", "rsID"])
n_min = full_df["Label"].value_counts().min()
balanced_df = full_df.groupby("Label").apply(lambda x: x.sample(n=n_min, random_state=42))
```
Concatenates benign + disease, deduplicates, undersamples majority class to achieve exact 1:1 balance. Shuffled with random_state=42. Saves to DeepFold_Dataset/final_dataset.csv (2,372 samples).

**Cell 19 — Dataset integrity checks:**
Four assertions that must all pass:
1. No duplicate (miRNA_ID, rsID) pairs
2. Perfect class balance (Label 0 count == Label 1 count)
3. No sequence leakage (Seq_Healthy ≠ Seq_Mutant for every sample)
4. Exactly 1 nucleotide difference per sample (spot-check on 20 random samples)

---

### STEP 6: UFold CONTACT MAP INFERENCE

**Cell 20 — Clone UFold repository:**
Clones https://github.com/uci-cbcl/UFold.git and installs dependencies.

**Cell 21 — Load UFold model and run inference:**

**Sequence encoding function (seq_to_input):**
Converts an RNA sequence into UFold's 17-channel input tensor:
- Channels 1-16: One-hot outer product maps. Each base is one-hot encoded (A=0, U/T=1, C=2, G=3), then the 16 possible outer products (4×4) are computed as L×L matrices. This encodes which base-pair combinations exist at each (i,j) position.
- Channel 17: Canonical pairing mask — set to 1.0 where Watson-Crick (A-U, C-G) or wobble (G-U) pairs are sterically possible (minimum gap ≥ 4 nucleotides between positions i and j).
- Padded/cropped to 128×128.
- Returns tensor of shape (1, 17, 128, 128).

**UFold model loading:**
```python
contact_net = FCNNet(img_ch=17)  # U-Net architecture from UFold repo
# Note: the class is actually U_Net, imported as FCNNet (naming fix from earlier repo versions)
state_dict = torch.load("ufold_train_alldata.pt")
# Strip "module." prefix if model was saved with DataParallel
contact_net.load_state_dict(state_dict)
contact_net.eval()
```

**Contact map inference (get_contact_map):**
```python
pred = contact_net(inp)           # raw logits
score_map = torch.sigmoid(pred)   # convert to probabilities [0, 1]
```
Returns the raw sigmoid output (continuous probabilities) NOT the thresholded binary output. This preserves the probabilistic nature of the predictions.

**4-channel tensor construction:**
For each of the 2,372 samples:
- C1 = get_contact_map(Seq_Healthy) — UFold prediction for healthy sequence
- C2 = get_contact_map(Seq_Mutant) — UFold prediction for mutant sequence
- C3 = |C1 − C2| — absolute difference = structural perturbation map (WHERE the SNP changed the structure)
- C4 = canonical pairing mask from the healthy sequence's 17th input channel
- Stack into (128, 128, 4) tensor, save as .npy file

Important: C3 (the diff channel) typically has values in the range 0.01-0.05, while C1/C2 have values in the range 0.3-0.8. This ~10x magnitude difference is critical for the CNN architecture design.

**Cell 22 — Sanity check:**
Verifies all 2,372 .npy files exist, have shape (128, 128, 4), and C3 is non-trivial (not all zeros) in ~100% of samples.

---

### STEP 7: DeepFoldCNN v3 TRAINING

**Cell 23 — Imports, seeds, dataset class:**

**SNPDataset class:**
```python
class SNPDataset(Dataset):
    def __getitem__(self, idx):
        tensor = np.load(f"{NPY_DIR}/{row['Sample_ID']}.npy")  # (128,128,4)
        tensor = torch.tensor(tensor).permute(2, 0, 1)          # → (4,128,128)
        # Augmentation (train only): random horizontal/vertical flips
        # Biologically valid because contact maps are symmetric
        return tensor, label, miRNA_ID
```

**Cell 24 — Model architecture:**

**SEBlock (Squeeze-and-Excitation):**
Channel attention mechanism: AdaptiveAvgPool → FC(channels→channels/4) → ReLU → FC(channels/4→channels) → Sigmoid → multiply with input. Learns to reweight channel importance.

**ResidualBlock:**
Two Conv2d(3×3) layers with BatchNorm and ReLU, followed by SE attention, residual skip connection (with 1×1 projection if channels change), and Dropout2d.

**DeepFoldCNN v3 — the main model:**
ROOT CAUSE FIX from v1/v2: In v1/v2, all 4 channels went through a shared first conv layer. Because C1/C2 have ~10x larger magnitude than C3, the learned filters were dominated by C1/C2, effectively ignoring the structural disruption signal in C3 (the most biologically relevant channel).

v3 SOLUTION — split-path input:
```
Input: (B, 4, 128, 128)

Main path (C3+C4 — diff + canonical mask):
  Conv2d(2→32, 3×3) + BN + ReLU → 32 channels
  These channels carry the structural disruption signal
  Dedicated conv filters learn diff-specific patterns

Context path (C1+C2 — healthy + mutant):
  Conv2d(2→16, 3×3) + BN + ReLU → 16 channels
  Compressed summary of absolute fold geometry

Merge:
  Concatenate [main(32), context(16)] → 48 channels
  SE attention (re-weights main vs context contribution)
  Dropout2d(0.10) + MaxPool(2) → (48, 64, 64)

Shared encoder:
  ResidualBlock(48→64) + MaxPool  → (64, 32, 32)
  ResidualBlock(64→128) + MaxPool → (128, 16, 16)
  ResidualBlock(128→256) + MaxPool → (256, 8, 8)

Classifier head:
  AdaptiveAvgPool2d(4) → (256, 4, 4)
  Flatten → 4096
  Linear(4096→512) + ReLU + Dropout(0.4)
  Linear(512→64) + ReLU + Dropout(0.2)
  Linear(64→2) → logits for [benign, disease]

~3.4M trainable parameters
```

**Cell 25 — Training utilities:**

**cosine_warmup_schedule:** Linear warmup for first 5 epochs (LR ramps from 0 to base LR), then cosine decay to 1e-6 floor for the remaining epochs.

**make_weighted_ce:** Creates CrossEntropyLoss with inverse-frequency class weights. Since the dataset is balanced 1:1, both weights = 1.0. Replaced FocalLoss — gamma=2.0 was suppressing disease-class gradients and causing recall to collapse in hard folds (fold 1: 0.371, fold 5: 0.330 in v1/v2).

**train_epoch / eval_epoch:** Standard PyTorch training loop with per-group channel normalisation applied inline: `tensors = (tensors - ch_mean) / ch_std`. Gradient clipping at max_norm=1.0.

**compute_metrics:** Calculates accuracy, AUC, precision, recall, F1, and confusion matrix from labels and predicted probabilities.

**Cell 26 — 5-fold cross-validation training loop:**

**Evaluation protocol: StratifiedGroupKFold**
```python
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
groups = df["miRNA_ID"].values
```
This ensures each test fold contains ENTIRELY UNSEEN miRNA families. No miRNA appearing in training can appear in the test set. This is critical because miRNAs within the same family share high sequence similarity — standard StratifiedKFold would leak information and inflate results.

**For each of the 5 folds:**
1. Split train indices into 85% training / 15% validation (random permutation)
2. Create DataLoaders (batch_size=32, shuffle=True for train, num_workers=2)
3. Initialize fresh DeepFoldCNN model on GPU
4. Initialize AdamW optimizer (lr=3e-4, weight_decay=1e-4)
5. Initialize cosine warmup scheduler

6. Compute per-group channel normalisation statistics from training data:
   - C1/C2 (context channels) get their own mean/std
   - C3/C4 (main channels) get their own separate mean/std
   - CRITICAL: uniform 4-channel z-score would destroy the magnitude difference between C3 (~0.01-0.05) and C1/C2 (~0.3-0.8) that the split-path conv filters rely on

7. Training loop (up to 120 epochs):
   - Train: forward pass through model, compute weighted CE loss, backward pass, clip gradients (max_norm=1.0), optimizer step
   - Validate: compute val loss, val accuracy, val AUC
   - Save checkpoint if val loss improves (saves model state_dict + channel normalisation stats)
   - Early stopping: if val loss hasn't improved for 25 epochs, stop
   - Print metrics every 10 epochs

8. Load best checkpoint, evaluate on test fold
9. Collect test predictions for aggregate metrics

After all 5 folds: prints mean ± std for accuracy, AUC, precision, recall, F1.

---

### STEP 8: ALTERNATIVE METHODS AND COMPARISON

**Cell 27 — Classical ML on handcrafted structural features:**

Extracts 28 features from each 4-channel contact map tensor:
- C1/C2 contact statistics (8): mean, std, max, fraction of high-confidence pairs (>0.5) for both healthy and mutant maps
- C3 (structural diff) statistics (6): mean, std, max, sum, fraction changed (>0.1 and >0.3)
- C4 canonical mask (2): mean, fraction
- Cross-channel (1): Pearson correlation between C1 and C2 flattened vectors
- SNP site features (1): mean of C3 in a 5×5 window centred on the SNP position
- Sequence properties (4): GC content, normalised length, SNP relative position, distance from centre
- Structural change (5): mean of C1/C2/C3 rows at SNP position, pairs gained, pairs lost

Two classifiers evaluated:
- SVM with RBF kernel (C=10, gamma="scale", with StandardScaler)
- XGBoost / GradientBoostingClassifier (200 trees, max_depth=4, lr=0.05)

Both use the same StratifiedGroupKFold protocol. Feature importance from XGBoost is saved.

**Cell 28 — Siamese GCN (Graph Convolutional Network):**

**Graph construction (build_graph_tensors):**
- Nodes: nucleotides, with 6-dim features: one-hot base (4) + normalised position (1) + mask (1)
- Edges: UFold contact score > 0.5 threshold + backbone edges (i↔i+1) + self-loops
- Adjacency normalisation: D^(-1/2) A D^(-1/2) for stable message passing
- Padded to fixed size MAX_L=128

**GCNLayer:** h' = ReLU(LayerNorm(A × W(h))) — standard spectral GCN with layer normalisation.

**SiameseGCN architecture:**
- Shared GCN encoder (3 layers: 6→64→64→128) processes BOTH healthy and mutant graphs with identical weights
- Graph-level embeddings via masked mean pooling: (h × mask).sum / mask.sum
- Classifier input: concatenation of [h, m, |h−m|, h×m] — 4-way interaction (512 dim)
- Classifier: FC(512→256) → FC(256→64) → FC(64→2)
- Training: 80 epochs, batch_size=32, AdamW lr=1e-3, CosineAnnealingLR, early stopping patience=20

**Cell 29 — k-mer XGBoost with seed region and conservation features:**

**971-dimensional feature vector:**

k-mer frequencies (960 dims):
- For k=3: 64 possible trimers (4^3), frequency vectors for healthy, mutant, and difference = 192
- For k=4: 256 possible 4-mers (4^4), frequency vectors for healthy, mutant, and difference = 768
- Total k-mer: 960 dimensions

Positional properties (4 dims):
- GC content of healthy sequence
- Normalised sequence length (length / 128)
- SNP relative position (snp_pos / length)
- SNP distance from centre (|snp_pos - L/2| / (L/2))

Seed region features (4 dims) — computed from miRBase GFF3 mature miRNA coordinates:
- snp_in_seed: 1 if SNP falls in mature miRNA positions 2-8 (the seed region that binds target mRNAs)
- snp_in_mature: 1 if SNP is anywhere within a mature miRNA
- seed_disruption: 1 if SNP changes a seed-region base
- mature_offset: normalised SNP position within the mature sequence

Conservation proxy features (3 dims):
- gc_5mer_ctx: GC fraction in ±2 nucleotide window around SNP (GC in stems tends to be more conserved)
- is_purine: 1 if reference base is a purine (transversions are more disruptive than transitions)
- snp_entropy: Shannon entropy of the 3-mer at the SNP site (low entropy = conserved context)

Classifier: GradientBoostingClassifier (300 trees, max_depth=4, lr=0.05, subsample=0.8).

Key finding: Benign SNPs appear SLIGHTLY MORE in seed positions (11.8%) than disease SNPs (9.6%). Interpretation: common variants in seed positions have passed population-level natural selection and are tolerated. Disease SNPs cluster more in structural stem regions where FOLD DISRUPTION (not seed sequence change) is the primary mechanism of pathogenicity.

Result: AUC 0.704 ± 0.021 — the strongest single model.

**Cell 30 — ViennaRNA MFE XGBoost:**

12 thermodynamic features per sample:
- mfe_healthy: minimum free energy of healthy sequence (kcal/mol) — predicted by ViennaRNA's RNA.fold()
- mfe_mutant: MFE of mutant sequence
- delta_mfe: mfe_mutant − mfe_healthy (positive = mutant is less stable / more destabilised)
- abs_delta_mfe: absolute value of delta_mfe
- mfe_ratio: mfe_mutant / mfe_healthy
- bp_healthy: number of base pairs in the MFE structure (counted from dot-bracket: number of "(" characters)
- bp_mutant: base pairs in mutant MFE structure
- delta_bp: change in base-pair count (bp_mutant − bp_healthy)
- snp_in_stem: 1 if the SNP position is in a paired region (the character at the SNP position in the dot-bracket structure is "(" or ")")
- snp_in_loop: 1 if the SNP position is in an unpaired region (character is ".")
- ensemble_diversity_h: structural ensemble diversity of healthy sequence (from RNA.pf_fold + RNA.mean_bp_distance)
- ensemble_diversity_m: ensemble diversity of mutant sequence

Key stats from this dataset:
- delta_mfe: disease mean=+1.42 kcal/mol, benign mean=+1.14 kcal/mol (disease SNPs destabilise hairpins more)
- snp_in_stem rate: disease=0.695, benign=0.687 (disease SNPs slightly more likely to hit paired positions)

Classifier: GradientBoosting with StandardScaler pipeline (200 trees, depth=3).
Result: AUC 0.633 ± 0.013 standalone.

**Cell 31 — Ensemble stacking:**

```python
meta_X4 = np.column_stack([cnn_probs, gcn_probs, kmer_probs, mfe_probs])
meta_model = LogisticRegression(C=1.0, max_iter=1000)
```

Combines all 4 base models via a logistic regression meta-learner:
- Input: 4 columns of out-of-fold (OOF) probabilities — these are predictions on held-out data from each base model's 5-fold CV, so there's no label leakage
- The meta-learner trains on these OOF probabilities using the same StratifiedGroupKFold splits
- It learns optimal linear weights for combining the 4 models

Learned weight distribution (mean across 5 folds):
- k-mer XGB: weight 2.49 (46.3% share) — strongest base model dominates
- MFE XGB: weight 1.49 (27.7% share) — thermodynamic features are the most complementary
- Siamese GCN: weight 0.87 (16.2% share)
- CNN v3: weight 0.53 (9.8% share) — weakest standalone but still contributes

Result: Enhanced 4-model ensemble achieves AUC 0.720 ± 0.020 — the best result.

Final comparison table (also saved to all_methods_comparison.csv):
- Enhanced Ensemble (CNN+GCN+kXGB+MFE): AUC 0.720 ± 0.020, Acc 0.658, F1 0.650 ← BEST
- k-mer+seed XGBoost: AUC 0.704 ± 0.021, Acc 0.645, F1 0.633 ← Best single model
- Siamese GCN: AUC 0.642 ± 0.027, Acc 0.600, F1 0.605
- ViennaRNA MFE XGBoost: AUC 0.633 ± 0.013, Acc 0.603, F1 0.609
- CNN v3 (DeepFoldCNN): AUC 0.625 ± 0.018, Acc 0.585, F1 0.535
- XGBoost (structural features): AUC 0.619 ± 0.030, Acc 0.587, F1 0.577
- SVM (structural features): AUC 0.612 ± 0.019, Acc 0.571, F1 0.578

---

## KEY SCIENTIFIC FINDINGS

1. **Sequence composition is the dominant signal** — k-mer XGB (AUC 0.704) outperforms all structural models. The compositional change introduced by a SNP (captured by k-mer difference vectors) is more discriminative than the structural change (captured by UFold contact map differences).

2. **Thermodynamic features are orthogonal and complementary** — ViennaRNA MFE XGB achieves only 0.633 standalone but receives 27.7% weight in the ensemble. Disease SNPs destabilise hairpins more on average (+1.42 vs +1.14 kcal/mol delta_mfe).

3. **The seed region finding is mechanistically interesting** — Benign gnomAD SNPs appear slightly MORE in seed positions (11.8%) than disease COSMIC SNPs (9.6%). Common variants in seed positions have passed population-level selection. Disease SNPs preferentially cluster in structural stem regions where fold disruption — not seed sequence change — drives pathogenicity.

4. **Evaluation rigour matters** — StratifiedGroupKFold grouped by miRNA family prevents information leakage from related sequences. Standard StratifiedKFold would produce inflated results.

5. **The AUC 0.720 ceiling** reflects genuine difficulty at this dataset size (2,372 samples, ~475 test per fold). Future directions include RNA language model features (RNA-FM, SpliceBERT), expanded datasets, PhyloP/GERP conservation scores, and multi-task learning.
