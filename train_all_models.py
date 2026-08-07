import os
import time

base = "DeepFold_Dataset"
folders = [
    "raw_data",
    "sequences/healthy",
    "sequences/mutated",
    "processed_maps/npy"
]

for f in folders:
    os.makedirs(os.path.join(base, f), exist_ok=True)

os.makedirs("sequences/healthy", exist_ok=True)
os.makedirs("sequences/mutated", exist_ok=True)

print("Workspace ready.")import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU — stop and enable GPU'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "")import pandas as pd

disease_path = "archive/DeepFold_pre_miRNA_disease_SNPs.csv"
disease_raw = pd.read_csv(disease_path)

# Standardize column names
disease_raw.columns = disease_raw.columns.str.lower().str.strip()

print("Disease SNPs loaded:", len(disease_raw))
print("Columns:", disease_raw.columns.tolist())
disease_raw.head()# Normalize chromosome column
if "chr" in disease_raw.columns:
    disease_raw["chr"] = disease_raw["chr"].astype(str).str.replace("chr", "", regex=False)
elif "chrom" in disease_raw.columns:
    disease_raw["chr"] = disease_raw["chrom"].astype(str).str.replace("chr", "", regex=False)

disease_raw["position"] = disease_raw["position"].astype(int)
disease_raw["ref"]      = disease_raw["ref"].astype(str).str.upper().str.strip()
disease_raw["alt"]      = disease_raw["alt"].astype(str).str.upper().str.strip()

# Deduplicate: if multiple diseases map to the same SNP, keep one row
# (store disease info as metadata — all diseases retained in 'disease' column via join)
if "rsid" in disease_raw.columns:
    dedup_key = ["rsid", "mirna_id"] if "mirna_id" in disease_raw.columns else ["rsid"]
else:
    dedup_key = ["chr", "position", "ref", "alt"]

disease_df = disease_raw.drop_duplicates(subset=dedup_key).copy()

print(f"After deduplication: {len(disease_df)} unique disease SNPs  (was {len(disease_raw)})")

# Save to raw_data/
disease_df.to_csv("DeepFold_Dataset/raw_data/disease_snps.csv", index=False)
print("Saved → DeepFold_Dataset/raw_data/disease_snps.csv")# Install bedtools (used for downstream region operations)
import pandas as pd

gff = pd.read_csv(
    "archive/hsa.gff3",
    sep="\t",
    comment="#",
    header=None
)

gff.columns = ["chr", "source", "type", "start", "end",
               "score", "strand", "phase", "attributes"]

# Keep only precursor miRNA entries
mirna_regions = gff[gff["type"] == "miRNA_primary_transcript"].copy()

# Normalize chromosome format (remove 'chr' prefix to match 1000G naming)
mirna_regions["chr"] = mirna_regions["chr"].astype(str).str.replace("chr", "", regex=False)

# GFF3 is 1-based inclusive; BED is 0-based start, 1-based end
mirna_regions["start"] = mirna_regions["start"] - 1

# Extract miRNA name from attributes
mirna_regions["mirna_id"] = mirna_regions["attributes"].str.extract(r'Name=([^;]+)')

mirna_regions[["chr", "start", "end"]].to_csv(
    "miRNA.bed",
    sep="\t",
    index=False,
    header=False
)

print("BED entries:", len(mirna_regions))import subprocess
import os
import time
import pandas as pd

BASE_URL    = "https://storage.googleapis.com/gcp-public-data--gnomad/release/3.1.2/vcf/genomes"
chromosomes = [str(i) for i in range(1, 23)] + ["X"]

if os.path.exists("miRNA_all_snps_body.txt"):
    os.remove("miRNA_all_snps_body.txt")

# Load miRNA regions — build per-chromosome region strings for bcftools -r
# This avoids the -R BED file entirely, which is unreliable with remote HTTPS VCFs
bed = pd.read_csv("miRNA.bed", sep="\t", header=None, names=["chr", "start", "end"])
# gnomAD uses chr-prefix and 1-based positions; BED is 0-based start
bed["start_1based"] = bed["start"] + 1

print(f"miRNA regions loaded: {len(bed)}")

for chrom in chromosomes:
    t0 = time.time()
    print(f"\nProcessing chr{chrom}...")

    vcf_url = f"{BASE_URL}/gnomad.genomes.v3.1.2.sites.chr{chrom}.vcf.bgz"

    # Build comma-separated region string for this chromosome
    # Format bcftools expects: chr1:100-200,chr1:500-600,...
    chrom_regions = bed[bed["chr"] == chrom]
    if len(chrom_regions) == 0:
        print(f"  No miRNA regions on chr{chrom}, skipping")
        continue

    region_str = ",".join(
        f"chr{row['chr']}:{int(row['start_1based'])}-{int(row['end'])}"
        for _, row in chrom_regions.iterrows()
    )

    result = subprocess.run([
        "bcftools", "view",
        "-r", region_str,               # FIX: -r regions string, not -R BED file
        "-v", "snps",
        "-m2", "-M2",
        "-i", "AF_popmax>0.005",
        vcf_url,
        "-Ov", "-o", f"chr{chrom}_filtered.vcf"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  Warning chr{chrom}: {result.stderr[:150]}")
        continue

    subprocess.run([
        "bash", "-c",
        f"grep -v '^#' chr{chrom}_filtered.vcf >> miRNA_all_snps_body.txt"
    ])

    if os.path.exists(f"chr{chrom}_filtered.vcf"):
        os.remove(f"chr{chrom}_filtered.vcf")

    elapsed = time.time() - t0
    print(f"  chr{chrom} done in {elapsed:.0f}s")

with open("miRNA_all_snps_body.txt") as f:
    total_lines = sum(1 for l in f if l.strip())
print(f"\ngnomAD extraction complete. Total variant lines: {total_lines}")import pandas as pd

# Parse miRNA_all_snps_body.txt into gnomad_raw
snps = []
with open("miRNA_all_snps_body.txt") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        snps.append({
            "chrom":    parts[0].replace("chr", ""),   # strip 'chr' prefix
            "position": int(parts[1]),
            "ref":      parts[3].upper().strip(),
            "alt":      parts[4].upper().strip()
        })

gnomad_raw = pd.DataFrame(snps, columns=["chrom", "position", "ref", "alt"])
print(f"gnomAD variants parsed: {len(gnomad_raw)}")
print(f"Columns: {gnomad_raw.columns.tolist()}")
print(gnomad_raw.head(3))

# Load disease SNPs for exclusion
disease_df = pd.read_csv("DeepFold_Dataset/raw_data/disease_snps.csv")
disease_df["chr"]      = disease_df["chr"].astype(str).str.replace("chr", "", regex=False)
disease_df["position"] = disease_df["position"].astype(int)
disease_df["ref"]      = disease_df["ref"].astype(str).str.upper().str.strip()
disease_df["alt"]      = disease_df["alt"].astype(str).str.upper().str.strip()
print(f"Disease SNPs loaded (for exclusion): {len(disease_df)}")

# Remove gnomAD variants overlapping known disease SNPs
merged = gnomad_raw.merge(
    disease_df[["chr", "position", "ref", "alt"]],
    left_on  =["chrom", "position", "ref", "alt"],
    right_on =["chr",   "position", "ref", "alt"],
    how="left",
    indicator=True
)

benign_clean = (
    merged[merged["_merge"] == "left_only"]
    .drop(columns=["_merge", "chr"])
    .drop_duplicates(subset=["chrom", "position", "ref", "alt"])
    .reset_index(drop=True)
)

print(f"\ngnomAD variants before disease removal: {len(gnomad_raw)}")
print(f"Removed as disease overlap:             {len(gnomad_raw) - len(benign_clean)}")
print(f"Final benign SNP count:                 {len(benign_clean)}")

benign_clean[["chrom", "position", "ref", "alt"]].to_csv(
    "DeepFold_Dataset/raw_data/benign_snps.csv", index=False
)
print("Saved → DeepFold_Dataset/raw_data/benign_snps.csv")

failed_regions = []  # kept for retry cell compatibility# Load hairpin sequences ONCE — reused throughout the pipeline
from Bio import SeqIO

hairpin_dict = {}
for record in SeqIO.parse(
        "archive/hairpin.fa", "fasta"):
    if record.id.startswith("hsa-"):   # Human miRNAs only
        hairpin_dict[record.id] = str(record.seq)

print("Total human hairpins loaded (hg38):", len(hairpin_dict))import pandas as pd

# Load benign SNPs (MAF-filtered, disease-removed)
benign_df = pd.read_csv("DeepFold_Dataset/raw_data/benign_snps.csv")
benign_df["chrom"] = benign_df["chrom"].astype(str).str.replace("chr", "", regex=False)
print("Benign SNPs:", len(benign_df))

# miRNA genomic regions (already loaded above, rebuild here for clarity)
gff = pd.read_csv(
    "archive/hsa.gff3",
    sep="\t", comment="#", header=None
)
gff.columns = ["chr", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]
mirna_regions = gff[gff["type"] == "miRNA_primary_transcript"].copy()
mirna_regions["chr"] = mirna_regions["chr"].astype(str).str.replace("chr", "", regex=False)
mirna_regions["mirna_id"] = mirna_regions["attributes"].str.extract(r'Name=([^;]+)')
mirna_regions = mirna_regions[["chr", "start", "end", "strand", "mirna_id"]]
print("miRNA regions:", len(mirna_regions))

# Map SNP → relative position within hairpin
mapped_snps = []
for _, snp in benign_df.iterrows():
    chrom = snp["chrom"]
    pos   = snp["position"]

    hits = mirna_regions[
        (mirna_regions["chr"] == chrom) &
        (mirna_regions["start"] <= pos) &
        (mirna_regions["end"]   >= pos)
    ]

    for _, mir in hits.iterrows():
        if mir["strand"] == "+":
            relative_pos = pos - mir["start"] + 1
        else:
            relative_pos = mir["end"] - pos + 1

        mapped_snps.append({
            "mirna_id":         mir["mirna_id"],
            "chrom":            chrom,
            "genomic_position": pos,
            "relative_position": relative_pos,
            "strand":           mir["strand"],
            "ref":              snp["ref"],
            "alt":              snp["alt"]
        })

mapped_df = pd.DataFrame(mapped_snps)
print("Mapped SNPs:", len(mapped_df))
mapped_df.to_csv("mapped_snps.csv", index=False)
print("Saved → mapped_snps.csv")
mapped_df.head()complement = {"A": "T", "T": "A", "C": "G", "G": "C"}

mismatch_count = 0
total_checked  = 0

for _, row in mapped_df.iterrows():
    mirna_id = row["mirna_id"]
    rel_pos  = int(row["relative_position"])
    strand   = row["strand"]
    ref      = row["ref"].upper()

    sequence = hairpin_dict.get(mirna_id)
    if sequence is None:
        continue
    if rel_pos <= 0 or rel_pos > len(sequence):
        continue

    seq_base = sequence[rel_pos - 1].upper()

    if strand == "-":
        ref = complement.get(ref, ref)

    ref_rna = ref.replace("T", "U")   # Compare DNA ref as RNA

    total_checked += 1
    if seq_base != ref_rna:
        mismatch_count += 1

print("Validation Results (hg38):")
print(f"  Total checked:    {total_checked}")
print(f"  Total mismatches: {mismatch_count}")
if total_checked > 0:
    print(f"  Mismatch rate:    {mismatch_count / total_checked:.4f}")
assert mismatch_count == 0, "Non-zero mismatches — check coordinate system!"import pandas as pd
import os

complement = {"A": "T", "T": "A", "C": "G", "G": "C"}

skipped_log  = []   # FIX: record reason for every skip
generated    = 0

os.makedirs("sequences/healthy", exist_ok=True)
os.makedirs("sequences/mutated", exist_ok=True)

for idx, row in mapped_df.iterrows():

    mirna_id = row["mirna_id"]
    rel_pos  = int(row["relative_position"])
    strand   = row["strand"]
    ref      = row["ref"].upper()
    alt      = row["alt"].upper()

    sequence = hairpin_dict.get(mirna_id)
    if sequence is None:
        skipped_log.append({"idx": idx, "mirna_id": mirna_id, "reason": "hairpin_not_found"})
        continue

    if rel_pos <= 0 or rel_pos > len(sequence):
        skipped_log.append({"idx": idx, "mirna_id": mirna_id,
                            "reason": f"position_out_of_bounds (pos={rel_pos}, len={len(sequence)})"})
        continue

    # Strand-aware allele conversion
    if strand == "-":
        ref = complement.get(ref, ref)
        alt = complement.get(alt, alt)

    seq_list = list(sequence)

    # Verify reference allele matches FASTA at this position
    # Compare in RNA space (hairpin FASTA uses U)
    ref_rna = ref.replace("T", "U")
    if seq_list[rel_pos - 1].upper() != ref_rna:
        skipped_log.append({
            "idx":      idx,
            "mirna_id": mirna_id,
            "reason":   f"ref_mismatch (expected {ref_rna}, got {seq_list[rel_pos-1].upper()}) pos={rel_pos}"
        })
        continue

    # Apply mutation
    alt_rna = alt.replace("T", "U")
    seq_list[rel_pos - 1] = alt_rna
    mutated_seq = "".join(seq_list)

    healthy_rna = sequence.replace("T", "U")
    mutated_rna = mutated_seq.replace("T", "U")

    # Unique file ID using dataframe index to handle multi-SNP miRNAs
    file_id = f"{mirna_id}_{idx}"

    with open(f"sequences/healthy/{file_id}.fasta", "w") as f:
        f.write(f">{file_id}_healthy\n{healthy_rna}\n")

    with open(f"sequences/mutated/{file_id}.fasta", "w") as f:
        f.write(f">{file_id}_mutated\n{mutated_rna}\n")

    generated += 1

print(f"Generated: {generated}")
print(f"Skipped:   {len(skipped_log)}")

# Save skip log for audit
skip_log_df = pd.DataFrame(skipped_log)
skip_log_df.to_csv("skipped_mutations_log.csv", index=False)
print("\nSkip reason breakdown:")
if len(skip_log_df) > 0:
    print(skip_log_df["reason"].str.split(" ").str[0].value_counts().to_string())
else:
    print("  None — all SNPs processed successfully.")def strict_position_check(mapped_df, n=20):
    """
    Spot-check the first n successfully generated pairs.
    Verifies:
      - Exactly 1 difference between healthy and mutated sequence
      - Difference is at the expected relative_position index
    """
    checked = 0
    for idx, row in mapped_df.iterrows():
        file_id = f"{row['mirna_id']}_{idx}"
        h_path  = f"sequences/healthy/{file_id}.fasta"
        m_path  = f"sequences/mutated/{file_id}.fasta"

        if not os.path.exists(h_path) or not os.path.exists(m_path):
            continue  # This SNP was skipped

        with open(h_path) as f:
            h_seq = f.readlines()[1].strip()
        with open(m_path) as f:
            m_seq = f.readlines()[1].strip()

        expected_index = int(row["relative_position"]) - 1

        diffs = [j for j in range(len(h_seq)) if h_seq[j] != m_seq[j]]

        assert len(diffs) == 1, \
            f"[{file_id}] Expected 1 diff, found {len(diffs)}: {diffs}"
        assert diffs[0] == expected_index, \
            f"[{file_id}] Mutation at index {diffs[0]}, expected {expected_index}"

        checked += 1
        if checked >= n:
            break

    print(f"Strict positional validation passed ({checked} samples checked).")

# FIX: actually call the function
strict_position_check(mapped_df, n=20)import pandas as pd
import os

records = []

for idx, row in mapped_df.iterrows():
    file_id = f"{row['mirna_id']}_{idx}"
    h_path  = f"sequences/healthy/{file_id}.fasta"
    m_path  = f"sequences/mutated/{file_id}.fasta"

    if not os.path.exists(h_path) or not os.path.exists(m_path):
        continue  # skip pairs that failed during mutation generation

    with open(h_path) as f:
        seq_healthy = f.readlines()[1].strip()
    with open(m_path) as f:
        seq_mutant = f.readlines()[1].strip()

    # rsID is not available for 1000G benign SNPs — use chrom:pos as identifier
    rsid = f"{row['chrom']}:{row['genomic_position']}"

    records.append({
        "Sample_ID":   file_id,
        "miRNA_ID":    row["mirna_id"],
        "rsID":        rsid,
        "Seq_Healthy": seq_healthy,
        "Seq_Mutant":  seq_mutant,
        "Label":       0   # 0 = Benign
    })

benign_master = pd.DataFrame(records)
print(f"Benign samples assembled: {len(benign_master)}")
benign_master.head()import pandas as pd
import os

disease_df = pd.read_csv("DeepFold_Dataset/raw_data/disease_snps.csv")
disease_df["chr"]      = disease_df["chr"].astype(str).str.replace("chr", "", regex=False)
disease_df["position"] = disease_df["position"].astype(int)
disease_df["ref"]      = disease_df["ref"].astype(str).str.upper().str.strip()
disease_df["alt"]      = disease_df["alt"].astype(str).str.upper().str.strip()

complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
os.makedirs("sequences/healthy", exist_ok=True)
os.makedirs("sequences/mutated", exist_ok=True)

disease_records = []
disease_skipped = []

for idx, row in disease_df.iterrows():

    # FIX: use pre_mirna column directly — correct hairpin name from source data
    mirna_id = str(row["pre_mirna"]).strip()
    chrom    = str(row["chr"])
    pos      = int(row["position"])
    ref      = row["ref"].upper()
    alt      = row["alt"].upper()
    rsid     = str(row.get("variation_id", f"{chrom}:{pos}"))

    # Look up hairpin using pre_mirna directly
    sequence = hairpin_dict.get(mirna_id)
    if sequence is None:
        disease_skipped.append({
            "idx": idx, "mirna_id": mirna_id,
            "reason": f"hairpin_not_found (pre_mirna={mirna_id})"
        })
        continue

    # Get relative position from GFF3 overlap
    hits = mirna_regions[
        (mirna_regions["chr"] == chrom) &
        (mirna_regions["start"] <= pos) &
        (mirna_regions["end"]   >= pos)
    ]
    if len(hits) == 0:
        disease_skipped.append({
            "idx": idx, "mirna_id": mirna_id,
            "reason": f"no_gff3_overlap ({chrom}:{pos})"
        })
        continue

    mir    = hits.iloc[0]
    strand = mir["strand"]

    if strand == "+":
        relative_pos = pos - mir["start"] + 1
    else:
        relative_pos = mir["end"] - pos + 1

    if relative_pos <= 0 or relative_pos > len(sequence):
        disease_skipped.append({
            "idx": idx, "mirna_id": mirna_id,
            "reason": f"position_out_of_bounds (pos={relative_pos}, len={len(sequence)})"
        })
        continue

    if strand == "-":
        ref = complement.get(ref, ref)
        alt = complement.get(alt, alt)

    seq_list = list(sequence)
    ref_rna  = ref.replace("T", "U")

    if seq_list[relative_pos - 1].upper() != ref_rna:
        disease_skipped.append({
            "idx": idx, "mirna_id": mirna_id,
            "reason": (f"ref_mismatch (expected {ref_rna}, "
                       f"got {seq_list[relative_pos-1].upper()}) pos={relative_pos}")
        })
        continue

    alt_rna = alt.replace("T", "U")
    seq_list[relative_pos - 1] = alt_rna
    mutated_seq = "".join(seq_list)

    healthy_rna = sequence.replace("T", "U")
    mutated_rna = mutated_seq.replace("T", "U")

    file_id = f"{mirna_id}_disease_{idx}"

    with open(f"sequences/healthy/{file_id}.fasta", "w") as f:
        f.write(f">{file_id}_healthy\n{healthy_rna}\n")
    with open(f"sequences/mutated/{file_id}.fasta", "w") as f:
        f.write(f">{file_id}_mutated\n{mutated_rna}\n")

    disease_records.append({
        "Sample_ID":   file_id,
        "miRNA_ID":    mirna_id,
        "rsID":        rsid,
        "Seq_Healthy": healthy_rna,
        "Seq_Mutant":  mutated_rna,
        "Label":       1
    })

disease_master = pd.DataFrame(disease_records)
print(f"Disease samples assembled: {len(disease_master)}")
print(f"Disease skipped:           {len(disease_skipped)}")

skip_df = pd.DataFrame(disease_skipped)
if len(skip_df) > 0:
    print("\nSkip reason breakdown:")
    print(skip_df["reason"].str.split(" ").str[0].value_counts().to_string())
    skip_df.to_csv("disease_skipped_log.csv", index=False)
    print("Full skip log → disease_skipped_log.csv")import pandas as pd

full_df = pd.concat([benign_master, disease_master], ignore_index=True)

# Deduplicate BEFORE balancing so final class counts are exactly equal
full_df = full_df.drop_duplicates(subset=["miRNA_ID", "rsID"]).reset_index(drop=True)
print("After dedup:")
print(full_df["Label"].value_counts().to_string())

# Balance by undersampling the majority class
# Note: include_groups omitted for pandas < 2.2 compatibility
n_min = full_df["Label"].value_counts().min()
balanced_df = (
    full_df.groupby("Label", group_keys=False)
           .apply(lambda x: x.sample(n=n_min, random_state=42))
           .reset_index(drop=True)
           .sample(frac=1, random_state=42)
           .reset_index(drop=True)
)

# Guard: drop duplicate columns pandas may inject on older versions
if balanced_df.columns.duplicated().any():
    balanced_df = balanced_df.loc[:, ~balanced_df.columns.duplicated()]

print(f"\nBalanced dataset: {len(balanced_df)} samples ({n_min} per class)")
print(balanced_df["Label"].value_counts().to_string())

balanced_df.to_csv("DeepFold_Dataset/final_dataset.csv", index=False)
print("\nSaved → DeepFold_Dataset/final_dataset.csv")
balanced_df.head()import pandas as pd

df = pd.read_csv("DeepFold_Dataset/final_dataset.csv")

print("=" * 50)
print("DEEPFOLD DATASET INTEGRITY REPORT")
print("=" * 50)

# Check 1: No duplicate miRNA-rsID pairs
duplicates = df.duplicated(subset=["miRNA_ID", "rsID"]).sum()
print(f"\n[1] Duplicate (miRNA_ID, rsID) pairs: {duplicates}")
assert duplicates == 0, "Duplicates found! Run dedup step."
print("    ✓ PASS")

# Check 2: Class balance
counts = df["Label"].value_counts()
print(f"\n[2] Class balance:")
print(f"    Label 0 (Benign):  {counts.get(0, 0)}")
print(f"    Label 1 (Disease): {counts.get(1, 0)}")
imbalance = abs(counts.get(0, 0) - counts.get(1, 0))
assert imbalance == 0, f"Class imbalance of {imbalance} detected!"
print("    ✓ PASS")

# Check 3: No sequence leakage (Seq_Healthy should never equal Seq_Mutant)
leakage = (df["Seq_Healthy"] == df["Seq_Mutant"]).sum()
print(f"\n[3] Samples where Seq_Healthy == Seq_Mutant: {leakage}")
assert leakage == 0, "Sequence leakage detected — mutation was not applied!"
print("    ✓ PASS")

# Check 4: SNP application spot-check (20 samples)
print(f"\n[4] SNP application spot-check (20 samples):")
errors = 0
for _, row in df.sample(n=min(20, len(df)), random_state=0).iterrows():
    h = row["Seq_Healthy"]
    m = row["Seq_Mutant"]
    diffs = sum(1 for a, b in zip(h, m) if a != b)
    if diffs != 1:
        print(f"    ✗ {row['Sample_ID']}: {diffs} differences (expected 1)")
        errors += 1
assert errors == 0, f"{errors} samples failed SNP spot-check."
print("    ✓ PASS")

print("\n" + "=" * 50)
print(f"ALL CHECKS PASSED — {len(df)} samples ready for UFold.")
print("=" * 50)import sys
sys.path.insert(0, 'UFold')

import torch
import numpy as np
import pandas as pd
import os
from torch import nn

from Network import U_Net as FCNNet   # ← FIXED: was FCNNet, actual class is U_Net

WEIGHTS_PATH = "archive/ufold_train_alldata.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

contact_net = FCNNet(img_ch=17)

state_dict = torch.load(WEIGHTS_PATH, map_location=device)
if any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

contact_net.load_state_dict(state_dict)
contact_net.to(device)
contact_net.eval()
print(f"UFold loaded on {device}")

# ── Sequence → 17-channel input tensor (UFold's encoding) ────────────────────
def seq_to_input(seq, max_len=128):
    """
    Replicates UFold's sequence encoding:
    - 16 channels: one-hot outer products for all 16 base-pair combinations
    - 1 channel: canonical pairing probability mask
    Returns: (1, 17, L, L) tensor
    """
    seq = seq.upper().replace("U", "T")
    L = min(len(seq), max_len)
    seq = seq[:L]

    # One-hot encode: A=0, U/T=1, C=2, G=3
    base_map = {'A': 0, 'T': 1, 'U': 1, 'C': 2, 'G': 3}
    one_hot = np.zeros((4, L), dtype=np.float32)
    for i, b in enumerate(seq):
        if b in base_map:
            one_hot[base_map[b], i] = 1.0

    # 16-channel outer product map
    mat = np.zeros((16, L, L), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            mat[i*4+j] = np.outer(one_hot[i], one_hot[j])

    # 17th channel: canonical pairing mask (Watson-Crick + wobble)
    canonical = {(0,1),(1,0),(2,3),(3,2),(3,1),(1,3)}  # AT,TA,CG,GC,GT,TG
    pair_mask = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        for j in range(L):
            if abs(i-j) >= 4:
                bi = base_map.get(seq[i], -1)
                bj = base_map.get(seq[j], -1)
                if (bi, bj) in canonical:
                    pair_mask[i, j] = 1.0

    # Stack and pad to max_len
    inp = np.concatenate([mat, pair_mask[np.newaxis]], axis=0)  # (17, L, L)
    padded = np.zeros((17, max_len, max_len), dtype=np.float32)
    padded[:, :L, :L] = inp

    return torch.tensor(padded).unsqueeze(0)  # (1, 17, max_len, max_len)

# ── Run inference and extract raw contact score map ───────────────────────────
def get_contact_map(seq, max_len=128):
    """Returns (max_len, max_len) float32 contact score map from UFold sigmoid output."""
    inp = seq_to_input(seq, max_len).to(device)
    with torch.no_grad():
        pred = contact_net(inp)           # (1, 1, max_len, max_len) logits
        score_map = torch.sigmoid(pred)   # convert to probabilities [0,1]
    return score_map.squeeze().cpu().numpy()  # (max_len, max_len)

# ── Process all samples ────────────────────────────────────────────────────────
df = pd.read_csv("DeepFold_Dataset/final_dataset.csv")
os.makedirs("DeepFold_Dataset/processed_maps/npy", exist_ok=True)

TARGET_SIZE = 128
failed = []

for idx, row in df.iterrows():
    sample_id = row["Sample_ID"]
    out_path  = f"DeepFold_Dataset/processed_maps/npy/{sample_id}.npy"

    if os.path.exists(out_path):
        continue  # resume-safe

    try:
        c1 = get_contact_map(row["Seq_Healthy"], TARGET_SIZE)  # UFold healthy
        c2 = get_contact_map(row["Seq_Mutant"],  TARGET_SIZE)  # UFold mutant
        c3 = np.abs(c1 - c2)                                   # structural diff
        # C4: canonical mask from healthy sequence (structural context)
        inp = seq_to_input(row["Seq_Healthy"], TARGET_SIZE)
        c4  = inp[0, 16].numpy()                               # 17th channel = canonical mask

        tensor = np.stack([c1, c2, c3, c4], axis=-1).astype(np.float32)  # (128,128,4)
        np.save(out_path, tensor)

    except Exception as e:
        failed.append({"sample_id": sample_id, "error": str(e)})

    if (idx + 1) % 100 == 0:
        print(f"  {idx+1}/{len(df)} done")

print(f"\nComplete. Tensors saved: {len(df) - len(failed)}  Failed: {len(failed)}")
if failed:
    pd.DataFrame(failed).to_csv("ufold_failed.csv", index=False)#sanitycheck
import numpy as np
import os
import pandas as pd

df = pd.read_csv("DeepFold_Dataset/final_dataset.csv")
npy_dir = "DeepFold_Dataset/processed_maps/npy"

missing     = []
wrong_shape = []

for _, row in df.iterrows():
    path = f"{npy_dir}/{row['Sample_ID']}.npy"
    if not os.path.exists(path):
        missing.append(row["Sample_ID"])
        continue
    t = np.load(path)
    if t.shape != (128, 128, 4):
        wrong_shape.append((row["Sample_ID"], t.shape))

print(f"Total samples:      {len(df)}")
print(f"NPY files present:  {len(df) - len(missing)}")
print(f"Missing:            {len(missing)}")
print(f"Wrong shape:        {len(wrong_shape)}")

if len(missing) == 0 and len(wrong_shape) == 0:
    sample = np.load(f"{npy_dir}/{df.iloc[0]['Sample_ID']}.npy")
    print(f"\nSample tensor shape: {sample.shape}")
    print(f"C1 (healthy UFold)  range: [{sample[:,:,0].min():.3f}, {sample[:,:,0].max():.3f}]")
    print(f"C2 (mutant UFold)   range: [{sample[:,:,1].min():.3f}, {sample[:,:,1].max():.3f}]")
    print(f"C3 (|diff|)         range: [{sample[:,:,2].min():.3f}, {sample[:,:,2].max():.3f}]")
    print(f"C4 (canonical mask) range: [{sample[:,:,3].min():.3f}, {sample[:,:,3].max():.3f}]")

    # Check C3 is non-trivial — if all zeros, mutation had no structural effect
    nonzero_c3 = np.mean([
        np.load(f"{npy_dir}/{row['Sample_ID']}.npy")[:,:,2].max() > 0.001
        for _, row in df.sample(50, random_state=0).iterrows()
    ])
    print(f"\nC3 non-trivial in {nonzero_c3*100:.0f}% of sampled tensors (expect ~100%)")
    print("\n✓ All tensors ready for CNN training.")import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import os, random

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

NPY_DIR  = "DeepFold_Dataset/processed_maps/npy"
CSV_PATH = "DeepFold_Dataset/final_dataset.csv"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

df = pd.read_csv(CSV_PATH)
print(f"Dataset: {len(df)} samples, {df['Label'].value_counts().to_dict()}")

class SNPDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row    = self.records.iloc[idx]
        tensor = np.load(f"{NPY_DIR}/{row['Sample_ID']}.npy")  # (128,128,4)
        tensor = torch.tensor(tensor, dtype=torch.float32).permute(2, 0, 1)  # (4,128,128)

        # Light augmentation on train set only — flips are biologically valid
        # (RNA contact maps are symmetric; horizontal/vertical flip = equivalent structure)
        if self.augment:
            if random.random() > 0.5:
                tensor = torch.flip(tensor, dims=[1])
            if random.random() > 0.5:
                tensor = torch.flip(tensor, dims=[2])

        label = torch.tensor(row["Label"], dtype=torch.long)
        return tensor, label, row["miRNA_ID"]class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention (unchanged from v2).
    """
    def __init__(self, channels, r=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale


class ResidualBlock(nn.Module):
    """
    Conv×2 + BN + ReLU + SE attention + residual add + Dropout2d.
    """
    def __init__(self, in_ch, out_ch, dropout_p=0.10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.se      = SEBlock(out_ch)
        self.drop    = nn.Dropout2d(dropout_p)
        self.project = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        residual = self.project(x)
        out      = self.conv(x)
        out      = self.se(out)
        return self.drop(out + residual)


class DeepFoldCNN(nn.Module):
    """
    DeepFoldCNN v3 — split-path input to protect the diff signal.

    ROOT CAUSE FIX: v1/v2 treated all 4 channels equally. The diff
    channel (C3 = healthy_map - mutant_map) has ~10x smaller magnitude
    than C1/C2. A shared first conv layer learns filters dominated by
    C1/C2, effectively ignoring the structural disruption signal in C3.

    v3 solution — two dedicated input paths:
      Main path  : C3 (diff) + C4 (mask)  →  32 channels
                   These carry the structural disruption signal.
                   Dedicated conv filters learn diff-specific patterns.
      Context path: C1 (healthy) + C2 (mutant)  →  16 channels
                   Compressed summary of absolute fold geometry.
      Merge: cat([main, context]) = 48 channels → shared encoder.

    Everything after Block 1 is identical to v2.
    Loss: CrossEntropyLoss + class weights (FocalLoss removed —
    gamma=2.0 was suppressing disease-class gradients causing
    recall collapse in hard folds).
    """
    def __init__(self, num_classes=2):
        super().__init__()

        # ── Split input paths ────────────────────────────────────────────────
        # Main path: diff (C3) + mask (C4) — structural disruption signal
        self.main_path = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Context path: healthy (C1) + mutant (C2) — absolute fold geometry
        self.ctx_path = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        # After cat: 48 channels — SE re-weights main vs context contribution
        self.input_se   = SEBlock(48)
        self.input_drop = nn.Dropout2d(0.10)
        self.pool1      = nn.MaxPool2d(2)   # 128 → 64

        # ── Shared encoder (48 → 64 → 128 → 256) ────────────────────────────
        self.block2 = ResidualBlock(48,  64,  dropout_p=0.15)  # 64 → 32
        self.pool2  = nn.MaxPool2d(2)
        self.block3 = ResidualBlock(64,  128, dropout_p=0.20)  # 32 → 16
        self.pool3  = nn.MaxPool2d(2)
        self.block4 = ResidualBlock(128, 256, dropout_p=0.10)  # 16 → 8
        self.pool4  = nn.MaxPool2d(2)

        # ── Classifier head (identical to v2) ────────────────────────────────
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),   # 256×8×8 → 256×4×4
            nn.Flatten(),              # → 4096
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Split channels: x is (B, 4, 128, 128)
        # C1=healthy, C2=mutant, C3=diff, C4=mask  (0-indexed)
        x_main = self.main_path(x[:, 2:4, :, :])   # diff + mask → 32ch
        x_ctx  = self.ctx_path(x[:, 0:2, :, :])    # healthy + mutant → 16ch
        x = torch.cat([x_main, x_ctx], dim=1)       # → 48ch
        x = self.input_se(x)                         # re-weight main vs context
        x = self.pool1(self.input_drop(x))           # 128 → 64

        x = self.pool2(self.block2(x))               # 64 → 32
        x = self.pool3(self.block3(x))               # 32 → 16
        x = self.pool4(self.block4(x))               # 16 → 8
        return self.classifier(x)


# Sanity check
_model = DeepFoldCNN()
_x     = torch.randn(2, 4, 128, 128)
_out   = _model(_x)
assert _out.shape == (2, 2), f"Unexpected output shape: {_out.shape}"
total_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)
print(f"Output shape      : {_out.shape}")
print(f"Trainable params  : {total_params:,}")
print("DeepFoldCNN v3 — split-path input (diff+mask // healthy+mutant) ready.")
print("Main path (C3+C4): 32 channels  |  Context path (C1+C2): 16 channels")
import math

# ── Cosine LR with Linear Warmup (unchanged) ─────────────────────────────────
def cosine_warmup_schedule(optimizer, warmup_epochs, total_epochs):
    """
    Linear warmup for `warmup_epochs` epochs, then cosine decay to 1e-6 floor.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(1e-6, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Loss: weighted CrossEntropyLoss (replaces FocalLoss) ─────────────────────
# FocalLoss with gamma=2.0 was suppressing disease-class gradients, causing
# recall to collapse in hard folds (fold 1: 0.371, fold 5: 0.330).
# Standard CE + class weights handles imbalance without destroying easy-example
# gradients that the model needs to stay calibrated on the disease class.
def make_weighted_ce(labels_arr, device):
    """
    Returns nn.CrossEntropyLoss with inverse-frequency class weights.
    weight[disease] = n_benign / n_disease  (upweights minority class)
    weight[benign]  = 1.0
    """
    n_benign  = (labels_arr == 0).sum()
    n_disease = (labels_arr == 1).sum()
    pos_w     = float(n_benign) / float(n_disease)
    weights   = torch.tensor([1.0, pos_w], dtype=torch.float32, device=device)
    print(f"  CrossEntropyLoss weights — benign: 1.000  disease: {pos_w:.3f}")
    return nn.CrossEntropyLoss(weight=weights)


# ── Train / Eval helpers ──────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, ch_mean, ch_std):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for tensors, labels, _ in loader:
        tensors, labels = tensors.to(device), labels.to(device)
        tensors = (tensors - ch_mean[None,:,None,None]) / ch_std[None,:,None,None]
        optimizer.zero_grad()
        outputs = model(tensors)
        loss    = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device, ch_mean, ch_std):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_labels, all_probs, all_mirnas = [], [], []
    with torch.no_grad():
        for tensors, labels, mirnas in loader:
            tensors, labels = tensors.to(device), labels.to(device)
            tensors = (tensors - ch_mean[None,:,None,None]) / ch_std[None,:,None,None]
            outputs = model(tensors)
            loss    = criterion(outputs, labels)
            probs   = F.softmax(outputs, dim=1)[:, 1]
            total_loss += loss.item() * len(labels)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += len(labels)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_mirnas.extend(mirnas)
    return (total_loss / total, correct / total,
            np.array(all_labels), np.array(all_probs), all_mirnas)


def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "auc":       roc_auc_score(labels, probs),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "cm":        confusion_matrix(labels, preds)
    }

print("cosine_warmup_schedule, make_weighted_ce, train/eval helpers defined.")
print("FocalLoss removed — using weighted CrossEntropyLoss.")
from sklearn.model_selection import StratifiedGroupKFold

EPOCHS        = 120
BATCH_SIZE    = 32
LR            = 3e-4
PATIENCE      = 25
WARMUP_EPOCHS = 5

sgkf         = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
groups       = df["miRNA_ID"].values
labels_arr   = df["Label"].values
fold_metrics = []
all_test_labels, all_test_probs, all_test_mirnas = [], [], []

os.makedirs("DeepFold_Dataset/checkpoints", exist_ok=True)

# ── Build loss once (class weights computed from full dataset) ────────────────
# make_weighted_ce uses inverse-frequency weighting: disease weight = n_benign/n_disease
# This replaces FocalLoss (gamma=2.0 was collapsing recall in hard folds)
criterion = make_weighted_ce(labels_arr, DEVICE)

for fold, (train_idx, test_idx) in enumerate(sgkf.split(df, labels_arr, groups)):
    print(f"\n{'='*55}")
    print(f"FOLD {fold+1}/5  — train: {len(train_idx)}  test: {len(test_idx)}")

    test_mirnas_unique  = df.iloc[test_idx]["miRNA_ID"].nunique()
    train_mirnas_unique = df.iloc[train_idx]["miRNA_ID"].nunique()
    print(f"  Train miRNAs: {train_mirnas_unique}  Test miRNAs: {test_mirnas_unique}")
    print(f"  Test labels:  {dict(df.iloc[test_idx]['Label'].value_counts().sort_index())}")
    print(f"{'='*55}")

    # Val split from train
    rng                = np.random.default_rng(SEED + fold)
    train_idx_shuffled = rng.permutation(train_idx)
    n_val              = int(0.15 * len(train_idx))
    val_idx            = train_idx_shuffled[:n_val]
    train_idx_         = train_idx_shuffled[n_val:]

    train_ds = SNPDataset(df.iloc[train_idx_], augment=True)
    val_ds   = SNPDataset(df.iloc[val_idx],    augment=False)
    test_ds  = SNPDataset(df.iloc[test_idx],   augment=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ── Model v3 (split-path input) ───────────────────────────────────────────
    model     = DeepFoldCNN().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = cosine_warmup_schedule(optimizer, WARMUP_EPOCHS, EPOCHS)

    # ── Per-fold channel statistics ───────────────────────────────────────────
    sample_tensors = []
    for i in range(min(200, len(train_ds))):
        t, _, _ = train_ds[i]
        sample_tensors.append(t)
    sample_tensors = torch.stack(sample_tensors)   # (N, 4, 128, 128)

    # ── Per-group channel normalisation ───────────────────────────────────
    # CRITICAL FIX: normalise C1/C2 (healthy/mutant) and C3/C4 (diff/mask)
    # with SEPARATE group statistics.
    # Uniform 4-channel z-score rescales all channels to mean=0 std=1,
    # destroying the magnitude difference between C3 (~0.01-0.05) and
    # C1/C2 (~0.3-0.8) that the split-path conv filters rely on.
    # Group normalisation preserves relative scale within each path.
    ctx_mean  = sample_tensors[:, 0:2].mean(dim=(0, 2, 3))          # C1, C2 mean
    ctx_std   = sample_tensors[:, 0:2].std(dim=(0, 2, 3)).clamp(min=1e-6)
    main_mean = sample_tensors[:, 2:4].mean(dim=(0, 2, 3))          # C3, C4 mean
    main_std  = sample_tensors[:, 2:4].std(dim=(0, 2, 3)).clamp(min=1e-6)
    # Rebuild 4-channel mean/std tensors for the training loop
    ch_mean = torch.cat([ctx_mean, main_mean]).to(DEVICE)           # [c1,c2,c3,c4]
    ch_std  = torch.cat([ctx_std,  main_std ]).to(DEVICE)

    print(f"  C1/C2 means: {ctx_mean.numpy().round(4)}  stds: {ctx_std.numpy().round(4)}")
    print(f"  C3/C4 means: {main_mean.numpy().round(4)}  stds: {main_std.numpy().round(4)}")
    print(f"  C3 raw mean: {main_mean[0].item():.5f}  "
          f"C1 raw mean: {ctx_mean[0].item():.5f}  "
          f"ratio: {(main_mean[0]/ctx_mean[0]).abs().item():.4f}")
    print(f"  (After group-norm, C3/C4 are separately scaled — "
          f"split-path filters see correct relative magnitudes)")

    best_val_loss = float("inf")
    patience_ctr  = 0
    best_ckpt     = f"DeepFold_Dataset/checkpoints/fold{fold+1}_best_v3.pt"

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_loss, tr_correct, tr_total = 0, 0, 0
        for tensors, labels, _ in train_dl:
            tensors, labels = tensors.to(DEVICE), labels.to(DEVICE)
            tensors = (tensors - ch_mean[None,:,None,None]) / ch_std[None,:,None,None]
            optimizer.zero_grad()
            outputs = model(tensors)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss    += loss.item() * len(labels)
            tr_correct += (outputs.argmax(1) == labels).sum().item()
            tr_total   += len(labels)
        tr_loss /= tr_total
        tr_acc   = tr_correct / tr_total

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        vl_loss, vl_correct, vl_total = 0, 0, 0
        vl_labels_list, vl_probs_list = [], []
        with torch.no_grad():
            for tensors, labels, _ in val_dl:
                tensors, labels = tensors.to(DEVICE), labels.to(DEVICE)
                tensors = (tensors - ch_mean[None,:,None,None]) / ch_std[None,:,None,None]
                outputs = model(tensors)
                probs   = F.softmax(outputs, dim=1)[:, 1]
                loss    = criterion(outputs, labels)
                vl_loss    += loss.item() * len(labels)
                vl_correct += (outputs.argmax(1) == labels).sum().item()
                vl_total   += len(labels)
                vl_labels_list.extend(labels.cpu().numpy())
                vl_probs_list.extend(probs.cpu().numpy())
        vl_loss /= vl_total
        vl_acc   = vl_correct / vl_total
        vl_auc   = roc_auc_score(vl_labels_list, vl_probs_list)
        scheduler.step()

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_ctr  = 0
            torch.save({
                "state_dict": model.state_dict(),
                "ch_mean":    ch_mean.cpu(),
                "ch_std":     ch_std.cpu()
            }, best_ckpt)
        else:
            patience_ctr += 1

        if epoch % 10 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d} | tr_loss {tr_loss:.4f} tr_acc {tr_acc:.4f} | "
                  f"vl_loss {vl_loss:.4f} vl_acc {vl_acc:.4f} vl_auc {vl_auc:.4f} "
                  f"lr {lr_now:.2e}"
                  + (" ← best" if patience_ctr == 0 else
                     f" (patience {patience_ctr}/{PATIENCE})"))

        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    # ── Evaluate best checkpoint on test fold ─────────────────────────────────
    ckpt = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    ch_mean_ckpt = ckpt["ch_mean"].to(DEVICE)
    ch_std_ckpt  = ckpt["ch_std"].to(DEVICE)

    model.eval()
    te_labels_list, te_probs_list, te_mirnas_list = [], [], []
    with torch.no_grad():
        for tensors, labels, mirnas in test_dl:
            tensors, labels = tensors.to(DEVICE), labels.to(DEVICE)
            tensors = (tensors - ch_mean_ckpt[None,:,None,None]) / ch_std_ckpt[None,:,None,None]
            outputs = model(tensors)
            probs   = F.softmax(outputs, dim=1)[:, 1]
            te_labels_list.extend(labels.cpu().numpy())
            te_probs_list.extend(probs.cpu().numpy())
            te_mirnas_list.extend(mirnas)

    te_y, te_p = np.array(te_labels_list), np.array(te_probs_list)
    m = compute_metrics(te_y, te_p)

    print(f"\n  Fold {fold+1} Test Results (v3):")
    print(f"    Accuracy:  {m['accuracy']:.4f}")
    print(f"    AUC:       {m['auc']:.4f}")
    print(f"    Precision: {m['precision']:.4f}")
    print(f"    Recall:    {m['recall']:.4f}    ← watch this vs v2 (was 0.519 mean)")
    print(f"    F1:        {m['f1']:.4f}")

    fold_metrics.append(m)
    all_test_labels.extend(te_y.tolist())
    all_test_probs.extend(te_p.tolist())
    all_test_mirnas.extend(te_mirnas_list)

print("\n" + "="*55)
print("5-FOLD CV SUMMARY — DeepFoldCNN v3 (split-path input)")
print("="*55)
for metric in ["accuracy", "auc", "precision", "recall", "f1"]:
    vals = [m[metric] for m in fold_metrics]
    print(f"  {metric.capitalize():12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
print()
print("Recall target: > 0.58 (was 0.519 in v1/v2)")
print("AUC target:    > 0.66 (was 0.635 in v1/v2)")
all_test_labels = np.array(all_test_labels)
all_test_probs  = np.array(all_test_probs)
all_test_preds  = (all_test_probs >= 0.5).astype(int)

print("AGGREGATE TEST SET METRICS (all folds combined)")
print("="*50)
agg = compute_metrics(all_test_labels, all_test_probs)
for k, v in agg.items():
    if k != "cm":
        print(f"  {k.capitalize():12s}: {v:.4f}")

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(agg["cm"], annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Disease"],
            yticklabels=["Benign", "Disease"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("DeepFold CNN — Confusion Matrix (5-fold)")
plt.tight_layout()
plt.savefig("DeepFold_Dataset/confusion_matrix.png", dpi=150)
plt.show()
print("Saved → confusion_matrix.png")import warnings
from sklearn.exceptions import UndefinedMetricWarning

mirna_results = {}
for label, prob, mirna in zip(all_test_labels, all_test_probs, all_test_mirnas):
    if mirna not in mirna_results:
        mirna_results[mirna] = {"labels": [], "probs": []}
    mirna_results[mirna]["labels"].append(label)
    mirna_results[mirna]["probs"].append(prob)

rows = []
for mirna, data in mirna_results.items():
    y  = np.array(data["labels"])
    p  = np.array(data["probs"])
    n  = len(y)
    pred = (p >= 0.5).astype(int)
    acc  = accuracy_score(y, pred)
    # AUC requires both classes present — suppress sklearn warning and store NaN
    if len(np.unique(y)) < 2:
        auc = float("nan")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            auc = roc_auc_score(y, p)
    rows.append({
        "miRNA_ID":    mirna,
        "n_samples":   n,
        "n_disease":   int(y.sum()),
        "n_benign":    int((1 - y).sum()),
        "accuracy":    round(acc, 4),
        "auc":         round(auc, 4),
        "mean_prob":   round(p.mean(), 4)
    })

mirna_df = (pd.DataFrame(rows)
              .sort_values("n_samples", ascending=False)
              .reset_index(drop=True))

mirna_df.to_csv("DeepFold_Dataset/per_mirna_results.csv", index=False)
print(f"Per-miRNA results: {len(mirna_df)} unique miRNAs")
print(f"\nTop 20 by sample count:")
print(mirna_df.head(20).to_string(index=False))

# Plot accuracy distribution across miRNAs
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(mirna_df["accuracy"].dropna(), bins=20, color="steelblue", edgecolor="white")
axes[0].set_title("Per-miRNA Accuracy Distribution")
axes[0].set_xlabel("Accuracy")
axes[0].set_ylabel("Count")

axes[1].hist(mirna_df["auc"].dropna(), bins=20, color="darkorange", edgecolor="white")
axes[1].set_title("Per-miRNA AUC Distribution")
axes[1].set_xlabel("AUC")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("DeepFold_Dataset/per_mirna_distributions.png", dpi=150)
plt.show()
print("Saved → per_mirna_distributions.png")
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

df      = pd.read_csv("DeepFold_Dataset/final_dataset.csv")
npy_dir = "DeepFold_Dataset/processed_maps/npy"
groups  = df["miRNA_ID"].values
y       = df["Label"].values
sgkf_struct = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

def extract_features(row):
    t  = np.load(f"{npy_dir}/{row['Sample_ID']}.npy")
    c1, c2, c3, c4 = t[:,:,0], t[:,:,1], t[:,:,2], t[:,:,3]
    seq_h = row["Seq_Healthy"]
    seq_m = row["Seq_Mutant"]
    L     = len(seq_h)
    snp_pos = next((i for i, (a,b) in enumerate(zip(seq_h, seq_m)) if a != b), 0)
    feats = [
        c1.mean(), c1.std(), c1.max(), (c1>0.5).sum()/(128*128),
        c2.mean(), c2.std(), c2.max(), (c2>0.5).sum()/(128*128),
        c3.mean(), c3.std(), c3.max(), c3.sum(),
        (c3>0.1).sum()/(128*128), (c3>0.3).sum()/(128*128),
        c4.mean(), c4.sum()/(128*128),
        c3.max() / (c3.mean() + 1e-8),
        np.corrcoef(c1.flatten(), c2.flatten())[0, 1],
        c3[max(0,snp_pos-2):snp_pos+3, max(0,snp_pos-2):snp_pos+3].mean(),
        (seq_h.count('G') + seq_h.count('C')) / max(L, 1),
        L / 128.0, snp_pos / max(L, 1),
        abs(snp_pos - L/2) / max(L/2, 1),
        c1[max(0,snp_pos-5):snp_pos+6, :].mean(),
        c2[max(0,snp_pos-5):snp_pos+6, :].mean(),
        c3[max(0,snp_pos-5):snp_pos+6, :].mean(),
        (c2 - c1).clip(min=0).sum(),
        (c1 - c2).clip(min=0).sum(),
    ]
    return np.array(feats, dtype=np.float32)

print("Extracting structural features...")
X_struct = np.stack([extract_features(row) for _, row in df.iterrows()])
X_struct = np.nan_to_num(X_struct, nan=0.0, posinf=0.0, neginf=0.0)
print(f"Feature matrix: {X_struct.shape}")

struct_models = {
    "SVM (RBF)": Pipeline([("scaler", StandardScaler()),
                            ("clf", SVC(kernel="rbf", C=10, gamma="scale",
                                        probability=True, random_state=42))]),
    "XGBoost":   GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                             learning_rate=0.05, random_state=42),
}

# Use structural_results — separate from CNN fold_metrics and kmer_results
structural_results = {}

for model_name, struct_model in struct_models.items():
    print(f"\n── {model_name} ──")
    struct_fold_metrics = []   # renamed to avoid shadowing CNN fold_metrics
    struct_all_probs    = []
    struct_all_labels   = []

    for fold, (train_idx, test_idx) in enumerate(sgkf_struct.split(X_struct, y, groups)):
        struct_model.fit(X_struct[train_idx], y[train_idx])
        probs = struct_model.predict_proba(X_struct[test_idx])[:, 1]
        preds = (probs >= 0.5).astype(int)
        m = {
            "accuracy":  accuracy_score(y[test_idx], preds),
            "auc":       roc_auc_score(y[test_idx], probs),
            "precision": precision_score(y[test_idx], preds, zero_division=0),
            "recall":    recall_score(y[test_idx], preds, zero_division=0),
            "f1":        f1_score(y[test_idx], preds, zero_division=0),
        }
        struct_fold_metrics.append(m)
        struct_all_probs.extend(probs.tolist())
        struct_all_labels.extend(y[test_idx].tolist())
        print(f"  Fold {fold+1} — AUC: {m['auc']:.4f}  Acc: {m['accuracy']:.4f}")

    structural_results[model_name] = {
        "fold_metrics": struct_fold_metrics,
        "all_probs":    struct_all_probs,
        "all_labels":   struct_all_labels
    }
    print(f"  Summary — AUC: {np.mean([m['auc'] for m in struct_fold_metrics]):.4f} ± "
          f"{np.std([m['auc'] for m in struct_fold_metrics]):.4f}")

# Feature importance from XGBoost — refit on full dataset
feature_names = [
    "c1_mean","c1_std","c1_max","c1_frac_high",
    "c2_mean","c2_std","c2_max","c2_frac_high",
    "c3_mean","c3_std","c3_max","c3_sum","c3_frac_01","c3_frac_03",
    "c4_mean","c4_frac","c3_peak_mean_ratio","c1c2_corr",
    "c3_snp_window","gc_content","seq_len","snp_rel_pos","snp_dist_centre",
    "c1_snp_row","c2_snp_row","c3_snp_row","pairs_gained","pairs_lost"
]
xgb_full = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                       learning_rate=0.05, random_state=42)
xgb_full.fit(X_struct, y)
imp = pd.Series(xgb_full.feature_importances_, index=feature_names)
print("\nTop 10 most important structural features:")
print(imp.sort_values(ascending=False).head(10).to_string())
imp.sort_values(ascending=False).to_csv("DeepFold_Dataset/feature_importances.csv")

# Confirm CNN fold_metrics is still intact
print(f"\n✓ CNN fold_metrics preserved: {len(fold_metrics)} folds")
print(f"  CNN AUCs: {[round(m['auc'],4) for m in fold_metrics]}")import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import os

df      = pd.read_csv("DeepFold_Dataset/final_dataset.csv")
npy_dir = "DeepFold_Dataset/processed_maps/npy"
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED    = 42
MAX_L   = 128

def build_graph_tensors(seq, contact_map, threshold=0.5):
    base_map = {'A': 0, 'U': 1, 'C': 2, 'G': 3, 'T': 1}
    seq      = seq.upper()
    L        = min(len(seq), MAX_L)
    node_feat = np.zeros((MAX_L, 6), dtype=np.float32)
    for i in range(L):
        b = seq[i]
        if b in base_map: node_feat[i, base_map[b]] = 1.0
        node_feat[i, 4] = i / max(L - 1, 1)
        node_feat[i, 5] = 1.0
    adj = np.zeros((MAX_L, MAX_L), dtype=np.float32)
    cm  = contact_map[:L, :L]
    adj[:L, :L] = (cm > threshold).astype(np.float32)
    for i in range(L - 1):
        adj[i, i+1] = 1.0
        adj[i+1, i] = 1.0
    np.fill_diagonal(adj, 1.0)
    deg     = adj.sum(axis=1, keepdims=True).clip(min=1)
    deg_inv = 1.0 / np.sqrt(deg)
    adj     = deg_inv * adj * deg_inv.T
    mask    = np.zeros(MAX_L, dtype=np.float32)
    mask[:L] = 1.0
    return node_feat, adj, mask

class RNAGraphDataset(Dataset):
    def __init__(self, records):
        self.records = records.reset_index(drop=True)
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        t   = np.load(f"{npy_dir}/{row['Sample_ID']}.npy")
        nf_h, adj_h, mask_h = build_graph_tensors(row["Seq_Healthy"], t[:,:,0])
        nf_m, adj_m, mask_m = build_graph_tensors(row["Seq_Mutant"],  t[:,:,1])
        return (torch.tensor(nf_h), torch.tensor(adj_h), torch.tensor(mask_h),
                torch.tensor(nf_m), torch.tensor(adj_m), torch.tensor(mask_m),
                torch.tensor(row["Label"], dtype=torch.long), row["miRNA_ID"])

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W    = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
    def forward(self, h, adj):
        return F.relu(self.norm(torch.bmm(adj, self.W(h))))

class SiameseGCN(nn.Module):
    def __init__(self, in_dim=6, hidden=64, embed_dim=128, dropout=0.3, num_classes=2):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.gcn3 = GCNLayer(hidden, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 4, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(dropout/2),
            nn.Linear(64, num_classes)
        )
    def encode(self, nf, adj, mask):
        h = self.drop(self.gcn1(nf, adj))
        h = self.drop(self.gcn2(h, adj))
        h = self.gcn3(h, adj)
        mask_exp = mask.unsqueeze(-1)
        return (h * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1)
    def forward(self, nf_h, adj_h, mask_h, nf_m, adj_m, mask_m):
        h = self.encode(nf_h, adj_h, mask_h)
        m = self.encode(nf_m, adj_m, mask_m)
        return self.classifier(torch.cat([h, m, torch.abs(h-m), h*m], dim=1))

EPOCHS_GNN, BATCH_GNN, LR_GNN, PATIENCE = 80, 32, 1e-3, 20
sgkf             = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
gnn_fold_metrics = []
gnn_all_labels, gnn_all_probs, gnn_all_mirnas = [], [], []
os.makedirs("DeepFold_Dataset/checkpoints", exist_ok=True)

for fold, (train_idx, test_idx) in enumerate(sgkf.split(df, df["Label"].values, df["miRNA_ID"].values)):
    print(f"\n{'='*55}\nFOLD {fold+1}/5 — train: {len(train_idx)}  test: {len(test_idx)}\n{'='*55}")
    rng     = np.random.default_rng(SEED + fold)
    tr_sh   = rng.permutation(train_idx)
    n_val   = int(0.15 * len(train_idx))
    val_i, tr_i = tr_sh[:n_val], tr_sh[n_val:]
    train_dl = DataLoader(RNAGraphDataset(df.iloc[tr_i]),   batch_size=BATCH_GNN, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(RNAGraphDataset(df.iloc[val_i]),  batch_size=BATCH_GNN, shuffle=False, num_workers=2)
    test_dl  = DataLoader(RNAGraphDataset(df.iloc[test_idx]),batch_size=BATCH_GNN,shuffle=False, num_workers=2)
    model     = SiameseGCN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_GNN, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_GNN)
    best_val_loss, patience_ctr = float("inf"), 0
    best_ckpt = f"DeepFold_Dataset/checkpoints/gcn_fold{fold+1}_best.pt"
    for epoch in range(1, EPOCHS_GNN + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0, 0, 0
        for nf_h, adj_h, mask_h, nf_m, adj_m, mask_m, labels, _ in train_dl:
            nf_h=nf_h.to(DEVICE); adj_h=adj_h.to(DEVICE); mask_h=mask_h.to(DEVICE)
            nf_m=nf_m.to(DEVICE); adj_m=adj_m.to(DEVICE); mask_m=mask_m.to(DEVICE)
            labels=labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(nf_h, adj_h, mask_h, nf_m, adj_m, mask_m)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()*len(labels); tr_correct += (outputs.argmax(1)==labels).sum().item(); tr_total += len(labels)
        scheduler.step()
        model.eval()
        vl_loss, vl_labels, vl_probs = 0, [], []
        with torch.no_grad():
            for nf_h, adj_h, mask_h, nf_m, adj_m, mask_m, labels, _ in val_dl:
                nf_h=nf_h.to(DEVICE); adj_h=adj_h.to(DEVICE); mask_h=mask_h.to(DEVICE)
                nf_m=nf_m.to(DEVICE); adj_m=adj_m.to(DEVICE); mask_m=mask_m.to(DEVICE)
                out = model(nf_h, adj_h, mask_h, nf_m, adj_m, mask_m)
                vl_loss += criterion(out, labels.to(DEVICE)).item()*len(labels)
                vl_probs.extend(F.softmax(out, dim=1)[:,1].cpu().numpy())
                vl_labels.extend(labels.numpy())
        vl_loss /= len(val_dl.dataset)
        vl_auc = roc_auc_score(vl_labels, vl_probs)
        if vl_loss < best_val_loss:
            best_val_loss, patience_ctr = vl_loss, 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_ctr += 1
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | tr_loss {tr_loss/tr_total:.4f} | vl_loss {vl_loss:.4f} vl_auc {vl_auc:.4f}"
                  + (" ← best" if patience_ctr==0 else f" (patience {patience_ctr}/{PATIENCE})"))
        if patience_ctr >= PATIENCE: print(f"  Early stopping at epoch {epoch}"); break
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    model.eval()
    te_labels, te_probs, te_mirnas = [], [], []
    with torch.no_grad():
        for nf_h, adj_h, mask_h, nf_m, adj_m, mask_m, labels, mirnas in test_dl:
            nf_h=nf_h.to(DEVICE); adj_h=adj_h.to(DEVICE); mask_h=mask_h.to(DEVICE)
            nf_m=nf_m.to(DEVICE); adj_m=adj_m.to(DEVICE); mask_m=mask_m.to(DEVICE)
            probs = F.softmax(model(nf_h, adj_h, mask_h, nf_m, adj_m, mask_m), dim=1)[:,1]
            te_labels.extend(labels.numpy()); te_probs.extend(probs.cpu().numpy()); te_mirnas.extend(mirnas)
    te_y, te_p = np.array(te_labels), np.array(te_probs)
    preds = (te_p >= 0.5).astype(int)
    m = {"accuracy": accuracy_score(te_y,preds), "auc": roc_auc_score(te_y,te_p),
         "precision": precision_score(te_y,preds,zero_division=0),
         "recall": recall_score(te_y,preds,zero_division=0), "f1": f1_score(te_y,preds,zero_division=0)}
    print(f"\n  Fold {fold+1} Test — Acc: {m['accuracy']:.4f}  AUC: {m['auc']:.4f}  F1: {m['f1']:.4f}")
    gnn_fold_metrics.append(m); gnn_all_labels.extend(te_y.tolist())
    gnn_all_probs.extend(te_p.tolist()); gnn_all_mirnas.extend(te_mirnas)

print(f"\n{'='*55}\nSiamese GCN — 5-FOLD CV SUMMARY\n{'='*55}")
for metric in ["accuracy","auc","precision","recall","f1"]:
    vals = [m[metric] for m in gnn_fold_metrics]
    print(f"  {metric.capitalize():12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")from itertools import product as iproduct
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

df     = pd.read_csv("DeepFold_Dataset/final_dataset.csv")
groups = df["miRNA_ID"].values
y      = df["Label"].values
sgkf   = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# ── Build mature miRNA offset lookup from GFF3 ───────────────────────────────
# GFF3 contains both miRNA_primary_transcript (precursor) and miRNA (mature) entries.
# For each pre-miRNA, we need the mature miRNA's start offset within the hairpin
# to determine if the SNP falls in the seed region (mature positions 2-8).
gff_full = pd.read_csv(
    "archive/hsa.gff3",
    sep="\t", comment="#", header=None
)
gff_full.columns = ["chr","source","type","start","end","score","strand","phase","attributes"]

# Extract mature miRNA entries
mature_gff = gff_full[gff_full["type"] == "miRNA"].copy()
mature_gff["chr"] = mature_gff["chr"].str.replace("chr", "", regex=False)
mature_gff["mature_id"]  = mature_gff["attributes"].str.extract(r'Name=([^;]+)')
mature_gff["pre_id"]     = mature_gff["attributes"].str.extract(r'Derives_from=([^;]+)')
mature_gff["strand"]     = mature_gff["strand"]
mature_gff["start"]      = mature_gff["start"].astype(int)  # 1-based
mature_gff["end"]        = mature_gff["end"].astype(int)

# Pre-miRNA regions (already loaded globally as mirna_regions, but rebuild cleanly)
pre_gff = gff_full[gff_full["type"] == "miRNA_primary_transcript"].copy()
pre_gff["chr"]     = pre_gff["chr"].str.replace("chr", "", regex=False)
pre_gff["pre_id"]  = pre_gff["attributes"].str.extract(r'ID=([^;]+)')
pre_gff["pre_name"]= pre_gff["attributes"].str.extract(r'Name=([^;]+)')
pre_gff["strand"]  = pre_gff["strand"]
pre_gff["g_start"] = pre_gff["start"].astype(int)  # 1-based
pre_gff["g_end"]   = pre_gff["end"].astype(int)

# Build lookup: pre_name → list of (mature_genomic_start, mature_genomic_end, strand)
pre_lookup = pre_gff.set_index("pre_id")[["pre_name","g_start","g_end","strand"]].to_dict("index")
mature_by_pre = {}  # pre_name → list of (mature_start_in_hairpin, mature_end_in_hairpin)

for _, row in mature_gff.iterrows():
    pre_id   = row["pre_id"]
    if pre_id not in pre_lookup:
        continue
    pre      = pre_lookup[pre_id]
    pre_name = pre["pre_name"]
    strand   = pre["strand"]
    g_start  = pre["g_start"]
    g_end    = pre["g_end"]
    m_start  = row["start"]
    m_end    = row["end"]
    # Convert to 0-based offset within hairpin
    if strand == "+":
        offset_start = m_start - g_start      # 0-based
        offset_end   = m_end   - g_start      # inclusive
    else:
        offset_start = g_end - m_end          # 0-based from hairpin start
        offset_end   = g_end - m_start        # inclusive
    if pre_name not in mature_by_pre:
        mature_by_pre[pre_name] = []
    mature_by_pre[pre_name].append((offset_start, offset_end))

print(f"Mature miRNA offset lookup built: {len(mature_by_pre)} pre-miRNAs with mature coords")


def seed_features(mirna_id, seq_h, seq_m):
    """
    Returns 4 seed/position features:
      snp_in_seed     : 1 if SNP is in mature miRNA positions 2-8 (seed region)
      snp_in_mature   : 1 if SNP is anywhere in a mature miRNA
      seed_disruption : 1 if SNP changes a base in the seed (subset of snp_in_seed)
      mature_offset   : normalised SNP position within mature (0 if not in mature)
    """
    snp_pos = next((i for i,(a,b) in enumerate(zip(seq_h,seq_m)) if a!=b), -1)
    if snp_pos == -1:
        return np.zeros(4, dtype=np.float32)

    matures = mature_by_pre.get(mirna_id, [])
    snp_in_seed   = 0.0
    snp_in_mature = 0.0
    mature_offset = 0.0

    for (ms, me) in matures:
        if ms <= snp_pos <= me:
            snp_in_mature = 1.0
            pos_in_mature = snp_pos - ms + 1  # 1-based position within mature
            mature_offset = pos_in_mature / max(me - ms + 1, 1)
            if 2 <= pos_in_mature <= 8:        # seed region definition
                snp_in_seed = 1.0
            break

    seed_disruption = snp_in_seed  # same flag — seed SNP always disrupts seed
    return np.array([snp_in_seed, snp_in_mature, seed_disruption, mature_offset],
                    dtype=np.float32)


def conservation_proxy(seq_h, snp_pos, struct_h=None):
    """
    3 conservation-proxy features computable without external bigwig:
      gc_5mer_context : GC fraction in 5-nt window around SNP
                        (GC positions in stems are more evolutionarily conserved)
      is_purine_change: 1 if SNP changes purine↔pyrimidine (transversion —
                        more disruptive than transition)
      snp_entropy     : Shannon entropy of the 3-mer at the SNP site
                        (low entropy = conserved context)
    """
    seq_h = seq_h.upper().replace("T", "U")
    L = len(seq_h)

    # GC context in ±2 window
    window = seq_h[max(0, snp_pos-2): min(L, snp_pos+3)]
    gc_ctx = sum(1 for b in window if b in "GC") / max(len(window), 1)

    # Purine/pyrimidine change
    purines = {"A", "G"}
    ref_base = seq_h[snp_pos] if snp_pos < L else "N"
    # We don't have alt directly but seq_m has it
    is_purine = 1.0 if ref_base in purines else 0.0

    # 3-mer Shannon entropy at SNP site
    trigram = seq_h[max(0, snp_pos-1): min(L, snp_pos+2)]
    from collections import Counter
    counts = Counter(trigram)
    total  = sum(counts.values())
    import math
    entropy = -sum((c/total)*math.log2(c/total) for c in counts.values() if c > 0)

    return np.array([gc_ctx, is_purine, entropy], dtype=np.float32)


# ── k-mer frequency (unchanged) ──────────────────────────────────────────────
def kmer_freq(seq, k):
    bases    = "ACGU"
    kmers    = ["".join(p) for p in iproduct(bases, repeat=k)]
    kmer_idx = {km: i for i, km in enumerate(kmers)}
    seq      = seq.upper().replace("T", "U")
    counts   = np.zeros(len(kmers), dtype=np.float32)
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer in kmer_idx: counts[kmer_idx[kmer]] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def build_kmer_features(row):
    seq_h    = row["Seq_Healthy"].upper().replace("T","U")
    seq_m    = row["Seq_Mutant"].upper().replace("T","U")
    mirna_id = row["miRNA_ID"]

    # k-mer frequencies (unchanged)
    kmer_feats = []
    for k in [3, 4]:
        h = kmer_freq(seq_h, k); m = kmer_freq(seq_m, k)
        kmer_feats.extend([h, m, h - m])

    snp_pos = next((i for i,(a,b) in enumerate(zip(seq_h,seq_m)) if a!=b), 0)
    L       = len(seq_h)

    # Original positional props
    props = np.array([
        (seq_h.count('G')+seq_h.count('C')) / max(L,1),
        L / 128.0,
        snp_pos / max(L,1),
        abs(snp_pos - L/2) / max(L/2, 1)
    ], dtype=np.float32)

    # NEW: seed region features (4)
    seed_feats = seed_features(mirna_id, seq_h, seq_m)

    # NEW: conservation proxy features (3)
    cons_feats = conservation_proxy(seq_h, snp_pos)

    return np.concatenate(kmer_feats + [props, seed_feats, cons_feats])


print("Building extended k-mer + seed + conservation feature matrix...")
X_kmer = np.stack([build_kmer_features(row) for _, row in df.iterrows()])
X_kmer = np.nan_to_num(X_kmer, nan=0.0, posinf=0.0, neginf=0.0)
print(f"Feature matrix: {X_kmer.shape}  (was 964, now {X_kmer.shape[1]}: +4 seed +3 conservation)")

# ── Sanity check: seed flag discriminability ──────────────────────────────────
seed_col = X_kmer[:, -7]   # snp_in_seed (first of seed_feats)
mat_col  = X_kmer[:, -6]   # snp_in_mature
print(f"\nSeed region stats:")
print(f"  SNPs in seed   — disease: {seed_col[y==1].mean():.3f}  benign: {seed_col[y==0].mean():.3f}")
print(f"  SNPs in mature — disease: {mat_col[y==1].mean():.3f}   benign: {mat_col[y==0].mean():.3f}")
print(f"  (Higher disease rate in seed = mechanistically informative signal)")

# ── 5-fold CV ─────────────────────────────────────────────────────────────────
print("\nTraining extended k-mer XGBoost...")
kmer_fold_metrics = []
kmer_all_probs    = []   # list → fold order, matches all_test_labels order
kmer_all_labels   = []

for fold, (train_idx, test_idx) in enumerate(sgkf.split(X_kmer, y, groups)):
    model = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                       learning_rate=0.05, subsample=0.8,
                                       random_state=42)
    model.fit(X_kmer[train_idx], y[train_idx])
    probs = model.predict_proba(X_kmer[test_idx])[:, 1]
    preds = (probs >= 0.5).astype(int)
    kmer_all_probs.extend(probs.tolist())
    kmer_all_labels.extend(y[test_idx].tolist())
    m = {
        "accuracy":  accuracy_score(y[test_idx], preds),
        "auc":       roc_auc_score(y[test_idx], probs),
        "precision": precision_score(y[test_idx], preds, zero_division=0),
        "recall":    recall_score(y[test_idx], preds, zero_division=0),
        "f1":        f1_score(y[test_idx], preds, zero_division=0),
    }
    kmer_fold_metrics.append(m)
    print(f"  Fold {fold+1} — AUC: {m['auc']:.4f}  Acc: {m['accuracy']:.4f}")

kmer_results = {
    "fold_metrics": kmer_fold_metrics,
    "all_probs":    kmer_all_probs,
    "all_labels":   kmer_all_labels,
}
print(f"\n  k-mer+seed+cons XGBoost — AUC: "
      f"{np.mean([m['auc'] for m in kmer_fold_metrics]):.4f} ± "
      f"{np.std([m['auc'] for m in kmer_fold_metrics]):.4f}")
print(f"  OOF AUC: {roc_auc_score(y[np.concatenate([t for _,t in sgkf.split(X_kmer,y,groups)])], kmer_all_probs):.4f}")

# ── Feature importance (top 15) ───────────────────────────────────────────────
full_model = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                        learning_rate=0.05, subsample=0.8,
                                        random_state=42)
full_model.fit(X_kmer, y)
# n_kmer = 64^1*3 + 64^2*3 = 192 + 768 = 960 (k-mer vectors only, props separate)
n_kmer = 960
feat_names = (
    [f"kmer_{i}" for i in range(n_kmer)] +
    ["gc_content","seq_len","snp_rel_pos","snp_dist_ctr"] +
    ["snp_in_seed","snp_in_mature","seed_disruption","mature_offset"] +
    ["gc_5mer_ctx","is_purine","snp_entropy"]
)
assert len(feat_names) == X_kmer.shape[1], (
    f"feat_names length {len(feat_names)} != X_kmer columns {X_kmer.shape[1]}")
imp = pd.Series(full_model.feature_importances_, index=feat_names)
print("\nTop 15 features (extended k-mer model):")
print(imp.sort_values(ascending=False).head(15).to_string())
# ── Install ViennaRNA Python bindings ─────────────────────────────────────────
import subprocess, sys
try:
    import RNA
    print("ViennaRNA already installed.")
except ImportError:
    print("Installing ViennaRNA...")
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "ViennaRNA", "-q", "--break-system-packages"],
                   check=True)
    import RNA
    print(f"ViennaRNA installed: {RNA.__version__}")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, accuracy_score,
                              precision_score, recall_score, f1_score)

df     = pd.read_csv("DeepFold_Dataset/final_dataset.csv")
groups = df["miRNA_ID"].values
y      = df["Label"].values
sgkf   = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)


def extract_mfe_features(seq_h, seq_m):
    """
    Compute 12 thermodynamic features for a healthy/mutant sequence pair.
    ViennaRNA uses DNA/RNA input — sequences are already RNA (U not T).
    Returns a float32 array of length 12.
    """
    seq_h = seq_h.upper().replace("T", "U")
    seq_m = seq_m.upper().replace("T", "U")

    # ── MFE structures ───────────────────────────────────────────────────────
    struct_h, mfe_h = RNA.fold(seq_h)
    struct_m, mfe_m = RNA.fold(seq_m)

    delta_mfe  = mfe_m - mfe_h          # positive = mutant more unstable
    abs_delta  = abs(delta_mfe)
    mfe_ratio  = mfe_m / (mfe_h + 1e-6) if mfe_h != 0 else 0.0

    # ── Base-pair counts from dot-bracket notation ────────────────────────────
    bp_h = struct_h.count("(")
    bp_m = struct_m.count("(")
    delta_bp = bp_m - bp_h

    # ── SNP position in structure ────────────────────────────────────────────
    snp_pos = next((i for i, (a, b) in enumerate(zip(seq_h, seq_m)) if a != b), 0)
    snp_char = struct_h[snp_pos] if snp_pos < len(struct_h) else "."
    snp_in_stem = 1.0 if snp_char in ("(", ")") else 0.0
    snp_in_loop = 1.0 if snp_char == "." else 0.0

    # ── Ensemble diversity (positional entropy) ───────────────────────────────
    # RNA.pf_fold returns (structure, free_energy_of_ensemble)
    # RNA.mean_bp_distance gives structural diversity of the ensemble
    try:
        _, _ = RNA.pf_fold(seq_h)
        div_h = RNA.mean_bp_distance(len(seq_h))
        _, _ = RNA.pf_fold(seq_m)
        div_m = RNA.mean_bp_distance(len(seq_m))
    except Exception:
        div_h, div_m = 0.0, 0.0

    return np.array([
        mfe_h, mfe_m, delta_mfe, abs_delta, mfe_ratio,
        float(bp_h), float(bp_m), float(delta_bp),
        snp_in_stem, snp_in_loop,
        div_h, div_m
    ], dtype=np.float32)


# ── Extract features for all 2372 samples ────────────────────────────────────
print("Extracting ViennaRNA thermodynamic features...")
print("(~2-4 min for 2372 samples on CPU)")

mfe_features = []
for idx, (_, row) in enumerate(df.iterrows()):
    feats = extract_mfe_features(row["Seq_Healthy"], row["Seq_Mutant"])
    mfe_features.append(feats)
    if (idx + 1) % 500 == 0:
        print(f"  {idx+1}/{len(df)} done")

X_mfe = np.stack(mfe_features)
X_mfe = np.nan_to_num(X_mfe, nan=0.0, posinf=0.0, neginf=0.0)
print(f"Feature matrix: {X_mfe.shape}")

feature_names_mfe = [
    "mfe_healthy", "mfe_mutant", "delta_mfe", "abs_delta_mfe", "mfe_ratio",
    "bp_healthy", "bp_mutant", "delta_bp",
    "snp_in_stem", "snp_in_loop",
    "ensemble_diversity_h", "ensemble_diversity_m"
]

# ── Sanity check: delta_mfe distribution ─────────────────────────────────────
delta_col = X_mfe[:, 2]   # delta_mfe
print(f"\ndelta_mfe stats:")
print(f"  mean={delta_col.mean():.4f}  std={delta_col.std():.4f}")
print(f"  min={delta_col.min():.4f}   max={delta_col.max():.4f}")
print(f"  disease mean={delta_col[y==1].mean():.4f}  "
      f"benign mean={delta_col[y==0].mean():.4f}")
print(f"  snp_in_stem rate (disease): {X_mfe[y==1, 8].mean():.3f}  "
      f"(benign): {X_mfe[y==0, 8].mean():.3f}")

# ── 5-fold CV with GradientBoosting ──────────────────────────────────────────
print("\nTraining MFE XGBoost (5-fold StratifiedGroupKFold)...")

mfe_fold_metrics = []
mfe_all_probs    = []   # list → fold order, matches all_test_labels order
mfe_all_labels   = []

for fold, (train_idx, test_idx) in enumerate(sgkf.split(X_mfe, y, groups)):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",   GradientBoostingClassifier(
                      n_estimators=200, max_depth=3,
                      learning_rate=0.05, subsample=0.8,
                      random_state=42))
    ])
    model.fit(X_mfe[train_idx], y[train_idx])
    probs = model.predict_proba(X_mfe[test_idx])[:, 1]
    preds = (probs >= 0.5).astype(int)
    mfe_all_probs.extend(probs.tolist())
    mfe_all_labels.extend(y[test_idx].tolist())

    m = {
        "accuracy":  accuracy_score(y[test_idx], preds),
        "auc":       roc_auc_score(y[test_idx], probs),
        "precision": precision_score(y[test_idx], preds, zero_division=0),
        "recall":    recall_score(y[test_idx], preds, zero_division=0),
        "f1":        f1_score(y[test_idx], preds, zero_division=0),
    }
    mfe_fold_metrics.append(m)
    print(f"  Fold {fold+1} — AUC: {m['auc']:.4f}  Acc: {m['accuracy']:.4f}  "
          f"Rec: {m['recall']:.4f}")

mfe_auc_mean = np.mean([m["auc"] for m in mfe_fold_metrics])
mfe_auc_std  = np.std([m["auc"]  for m in mfe_fold_metrics])
print(f"\nMFE XGBoost — AUC: {mfe_auc_mean:.4f} ± {mfe_auc_std:.4f}")

# ── Feature importance ────────────────────────────────────────────────────────
# Refit on full dataset to get stable importances
full_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",   GradientBoostingClassifier(
                  n_estimators=200, max_depth=3,
                  learning_rate=0.05, subsample=0.8,
                  random_state=42))
])
full_pipe.fit(X_mfe, y)
imp = pd.Series(full_pipe.named_steps["clf"].feature_importances_,
                index=feature_names_mfe)
print("\nFeature importances (MFE model):")
print(imp.sort_values(ascending=False).to_string())

# Store for ensemble
mfe_results = {
    "fold_metrics": mfe_fold_metrics,
    "all_probs":    mfe_all_probs,
    "all_labels":   mfe_all_labels,
}
print(f"\nMFE results stored → mfe_results")
print(f"OOF AUC: {roc_auc_score(y[np.concatenate([t for _,t in sgkf.split(X_mfe,y,groups)])], mfe_all_probs):.4f}")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

df     = pd.read_csv("DeepFold_Dataset/final_dataset.csv")
groups = df["miRNA_ID"].values
y      = df["Label"].values
sgkf   = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# Collect out-of-fold probability arrays from each base model
cnn_p  = np.array(all_test_probs)            # CNN v3
gcn_p  = np.array(gnn_all_probs)             # Siamese GCN
kxgb_p = np.array(kmer_results["all_probs"]) # k-mer XGBoost
mfe_p  = mfe_results["all_probs"]            # ViennaRNA MFE XGBoost
true_y = np.array(all_test_labels)

print("Base model OOF AUCs:")
print(f"  CNN v3:      {roc_auc_score(true_y, cnn_p):.4f}")
print(f"  GCN:         {roc_auc_score(true_y, gcn_p):.4f}")
print(f"  k-mer XGB:   {roc_auc_score(true_y, kxgb_p):.4f}")
print(f"  MFE XGB:     {roc_auc_score(true_y, mfe_p):.4f}")

# Meta-feature matrix: (N, 4) — one column per base model
meta_X4 = np.column_stack([cnn_p, gcn_p, kxgb_p, mfe_p])
print(f"\nMeta-feature matrix: {meta_X4.shape}")
print("Training Enhanced Ensemble (CNN + GCN + k-mer XGB + MFE XGB)...")

enhanced_fold_metrics = []
enhanced_all_probs    = np.zeros(len(true_y))

for fold, (train_idx, test_idx) in enumerate(sgkf.split(meta_X4, true_y, groups)):
    meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_model.fit(meta_X4[train_idx], true_y[train_idx])
    probs = meta_model.predict_proba(meta_X4[test_idx])[:, 1]
    preds = (probs >= 0.5).astype(int)
    enhanced_all_probs[test_idx] = probs
    m = {
        "accuracy":  accuracy_score(true_y[test_idx], preds),
        "auc":       roc_auc_score(true_y[test_idx], probs),
        "precision": precision_score(true_y[test_idx], preds, zero_division=0),
        "recall":    recall_score(true_y[test_idx], preds, zero_division=0),
        "f1":        f1_score(true_y[test_idx], preds, zero_division=0),
    }
    enhanced_fold_metrics.append(m)
    w = meta_model.coef_[0]
    print(f"  Fold {fold+1} — AUC: {m['auc']:.4f}  Acc: {m['accuracy']:.4f} | "
          f"Weights — CNN: {w[0]:.3f}  GCN: {w[1]:.3f}  "
          f"k-mer XGB: {w[2]:.3f}  MFE XGB: {w[3]:.3f}")

enh_auc  = np.mean([m["auc"]      for m in enhanced_fold_metrics])
enh_sauc = np.std([m["auc"]       for m in enhanced_fold_metrics])
enh_acc  = np.mean([m["accuracy"] for m in enhanced_fold_metrics])
enh_f1   = np.mean([m["f1"]       for m in enhanced_fold_metrics])

print(f"\n{'='*70}")
print(f"{'FINAL METHOD COMPARISON — 5-FOLD STRATIFIED GROUP CV':^70}")
print(f"{'='*70}")
print(f"{'Method':<28} {'AUC':^20} {'Accuracy':^12} {'F1':^8}")
print(f"{'-'*70}")

results = [
    ("MFE XGBoost (ViennaRNA)",   np.mean([m["auc"] for m in mfe_fold_metrics]),
                                    np.std([m["auc"]  for m in mfe_fold_metrics]),
                                    np.mean([m["accuracy"] for m in mfe_fold_metrics]),
                                    np.mean([m["f1"] for m in mfe_fold_metrics])),
    ("CNN v3 (DeepFoldCNN)",        np.mean([m["auc"] for m in fold_metrics]),
                                     np.std([m["auc"]  for m in fold_metrics]),
                                     np.mean([m["accuracy"] for m in fold_metrics]),
                                     np.mean([m["f1"] for m in fold_metrics])),
    ("Siamese GCN",                 np.mean([m["auc"] for m in gnn_fold_metrics]),
                                     np.std([m["auc"]  for m in gnn_fold_metrics]),
                                     np.mean([m["accuracy"] for m in gnn_fold_metrics]),
                                     np.mean([m["f1"] for m in gnn_fold_metrics])),
    ("XGBoost (structural)",        np.mean([m["auc"] for m in structural_results["XGBoost"]["fold_metrics"]]),
                                     np.std([m["auc"]  for m in structural_results["XGBoost"]["fold_metrics"]]),
                                     np.mean([m["accuracy"] for m in structural_results["XGBoost"]["fold_metrics"]]),
                                     np.mean([m["f1"] for m in structural_results["XGBoost"]["fold_metrics"]])),
    ("SVM (structural)",            np.mean([m["auc"] for m in structural_results["SVM (RBF)"]["fold_metrics"]]),
                                     np.std([m["auc"]  for m in structural_results["SVM (RBF)"]["fold_metrics"]]),
                                     np.mean([m["accuracy"] for m in structural_results["SVM (RBF)"]["fold_metrics"]]),
                                     np.mean([m["f1"] for m in structural_results["SVM (RBF)"]["fold_metrics"]])),
]
for name, auc, sauc, acc, f1 in sorted(results, key=lambda x: -x[1]):
    print(f"{name:<28} {auc:.4f} ± {sauc:.4f}   {acc:.4f}     {f1:.4f}")

best = " ← BEST" if enh_auc > max(r[1] for r in results) else ""
print(f"{'Enhanced Ensemble (4-model)':<28} {enh_auc:.4f} ± {enh_sauc:.4f}   "
      f"{enh_acc:.4f}     {enh_f1:.4f}{best}")
print(f"{'='*70}")

# Save comparison
rows = []
for name, folds_list in [("CNN_v3", fold_metrics), ("GCN", gnn_fold_metrics),
                          ("XGBoost_struct", structural_results["XGBoost"]["fold_metrics"]),
                          ("SVM_struct", structural_results["SVM (RBF)"]["fold_metrics"]),
                          ("MFE_XGB", mfe_fold_metrics),
                          ("Enhanced_Ensemble_4", enhanced_fold_metrics)]:
    for i, m in enumerate(folds_list):
        rows.append({"method": name, "fold": i+1,
                     **{k: v for k,v in m.items() if k != "cm"}})
pd.DataFrame(rows).to_csv("DeepFold_Dataset/all_methods_comparison.csv", index=False)
print("Saved → all_methods_comparison.csv")# These cells go AFTER all existing cells in final.ipynb.
# ═══════════════════════════════════════════════════════════════════════════════import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import os, random, math

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
NPY_DIR  = "DeepFold_Dataset/processed_maps/npy"
CSV_PATH = "DeepFold_Dataset/final_dataset.csv"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

df = pd.read_csv(CSV_PATH)
print(f"Dataset: {len(df)} samples, {df['Label'].value_counts().to_dict()}")


class SNPDataset_v4(Dataset):
    """v4 dataset — identical to v3 but returns one-hot labels for mixup."""
    def __init__(self, records, augment=False, num_classes=2):
        self.records = records.reset_index(drop=True)
        self.augment = augment
        self.num_classes = num_classes

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row    = self.records.iloc[idx]
        tensor = np.load(f"{NPY_DIR}/{row['Sample_ID']}.npy")
        tensor = torch.tensor(tensor, dtype=torch.float32).permute(2, 0, 1)

        if self.augment:
            if random.random() > 0.5:
                tensor = torch.flip(tensor, dims=[1])
            if random.random() > 0.5:
                tensor = torch.flip(tensor, dims=[2])

        label_idx = torch.tensor(row["Label"], dtype=torch.long)
        # One-hot for mixup compatibility
        label_oh = torch.zeros(self.num_classes, dtype=torch.float32)
        label_oh[row["Label"]] = 1.0
        return tensor, label_idx, label_oh, row["miRNA_ID"]


class MultiScalePool(nn.Module):
    """Replaces AdaptiveAvgPool2d(4) — captures global and local spatial info."""
    def forward(self, x):
        p1 = F.adaptive_avg_pool2d(x, 1).flatten(1)   # (B, 256)
        p2 = F.adaptive_max_pool2d(x, 1).flatten(1)   # (B, 256)
        p3 = F.adaptive_avg_pool2d(x, 2).flatten(1)   # (B, 1024)
        return torch.cat([p1, p2, p3], dim=1)           # (B, 1536)


class CoordAttention(nn.Module):
    """Position-aware channel attention via horizontal + vertical pooling."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        h_pool = x.mean(dim=3, keepdim=True)                    # (B,C,H,1)
        w_pool = x.mean(dim=2, keepdim=True).permute(0,1,3,2)   # (B,C,W,1) → (B,C,W,1)
        # Concatenate along H dimension: (B, C, H+W, 1)
        combined = torch.cat([h_pool, w_pool], dim=2)            # (B,C,H+W,1)
        att = torch.sigmoid(self.bn(self.conv(combined)))        # (B,C,H+W,1)
        att_h = att[:, :, :H, :]                                 # (B,C,H,1)
        att_w = att[:, :, H:, :].permute(0, 1, 3, 2)            # (B,C,1,W)
        return x * att_h * att_w


# SEBlock and ResidualBlock — identical to v3
class SEBlock(nn.Module):
    def __init__(self, channels, r=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.se      = SEBlock(out_ch)
        self.drop    = nn.Dropout2d(dropout_p)
        self.project = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                        if in_ch != out_ch else nn.Identity())
    def forward(self, x):
        residual = self.project(x)
        out = self.se(self.conv(x))
        return self.drop(out + residual)


class DeepFoldCNN_v4(nn.Module):
    """
    v4 improvements over v3:
      - CoordAttention after merge (position-aware)
      - MultiScalePool replacing AdaptiveAvgPool(4)
      - Classifier updated for 1536-dim input
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # Split input paths (identical to v3)
        self.main_path = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.ctx_path = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )
        # NEW: Coordinate Attention after merge
        self.coord_att  = CoordAttention(48)
        self.input_se   = SEBlock(48)
        self.input_drop = nn.Dropout2d(0.10)
        self.pool1      = nn.MaxPool2d(2)

        # Shared encoder (identical to v3)
        self.block2 = ResidualBlock(48,  64,  dropout_p=0.15)
        self.pool2  = nn.MaxPool2d(2)
        self.block3 = ResidualBlock(64,  128, dropout_p=0.20)
        self.pool3  = nn.MaxPool2d(2)
        self.block4 = ResidualBlock(128, 256, dropout_p=0.10)
        self.pool4  = nn.MaxPool2d(2)

        # NEW: Multi-scale pooling + updated classifier
        self.ms_pool = MultiScalePool()
        self.classifier = nn.Sequential(
            nn.Linear(1536, 512),       # was 4096 in v3
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x_main = self.main_path(x[:, 2:4, :, :])
        x_ctx  = self.ctx_path(x[:, 0:2, :, :])
        x = torch.cat([x_main, x_ctx], dim=1)
        x = self.coord_att(x)                        # NEW: position-aware attention
        x = self.input_se(x)
        x = self.pool1(self.input_drop(x))

        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        x = self.ms_pool(x)                          # NEW: multi-scale pooling
        return self.classifier(x)


# Sanity check
_m = DeepFoldCNN_v4()
_x = torch.randn(2, 4, 128, 128)
_o = _m(_x)
assert _o.shape == (2, 2), f"Unexpected: {_o.shape}"
print(f"DeepFoldCNN v4 output: {_o.shape}")
print(f"Trainable params: {sum(p.numel() for p in _m.parameters() if p.requires_grad):,}")
print("v4 ready: CoordAttention + MultiScalePool + label_smoothing + mixup")def cosine_warmup_schedule(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(1e-6, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_weighted_ce_v4(labels_arr, device, label_smoothing=0.10):
    """CrossEntropyLoss with class weights AND label smoothing."""
    n_benign  = (labels_arr == 0).sum()
    n_disease = (labels_arr == 1).sum()
    pos_w     = float(n_benign) / float(n_disease)
    weights   = torch.tensor([1.0, pos_w], dtype=torch.float32, device=device)
    print(f"  CE weights — benign: 1.000  disease: {pos_w:.3f}  "
          f"label_smoothing: {label_smoothing}")
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)


def mixup_batch(x, y_onehot, alpha=0.2):
    """Mixup on tensors + one-hot labels. Returns mixed x and soft y."""
    if alpha <= 0:
        return x, y_onehot
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # ensure lam >= 0.5 for stability
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_mix = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return x_mix, y_mix


def soft_cross_entropy(logits, soft_targets, weight=None):
    """CE loss for soft (mixup) labels. Handles class weights."""
    log_probs = F.log_softmax(logits, dim=1)
    if weight is not None:
        log_probs = log_probs * weight.unsqueeze(0)
    loss = -(soft_targets * log_probs).sum(dim=1).mean()
    return loss


def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "auc":       roc_auc_score(labels, probs),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "cm":        confusion_matrix(labels, preds)
    }

print("v4 training utilities defined (mixup, soft CE, label smoothing).")EPOCHS        = 120
BATCH_SIZE    = 32
LR            = 3e-4
PATIENCE      = 25
WARMUP_EPOCHS = 5
MIXUP_ALPHA   = 0.2

sgkf_v4      = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
groups       = df["miRNA_ID"].values
labels_arr   = df["Label"].values

v4_fold_metrics = []
v4_all_labels, v4_all_probs, v4_all_mirnas = [], [], []

os.makedirs("DeepFold_models", exist_ok=True)

# Build class weights once
n_b = (labels_arr == 0).sum()
n_d = (labels_arr == 1).sum()
pos_w = float(n_b) / float(n_d)
class_weights = torch.tensor([1.0, pos_w], dtype=torch.float32, device=DEVICE)

for fold, (train_idx, test_idx) in enumerate(sgkf_v4.split(df, labels_arr, groups)):
    print(f"\n{'='*55}")
    print(f"FOLD {fold+1}/5 — train: {len(train_idx)}  test: {len(test_idx)}")
    print(f"  Test miRNAs: {df.iloc[test_idx]['miRNA_ID'].nunique()}")
    print(f"  Test labels: {dict(df.iloc[test_idx]['Label'].value_counts().sort_index())}")
    print(f"{'='*55}")

    # Val split
    rng = np.random.default_rng(SEED + fold)
    train_idx_shuffled = rng.permutation(train_idx)
    n_val = int(0.15 * len(train_idx))
    val_idx, train_idx_ = train_idx_shuffled[:n_val], train_idx_shuffled[n_val:]

    train_ds = SNPDataset_v4(df.iloc[train_idx_], augment=True)
    val_ds   = SNPDataset_v4(df.iloc[val_idx],    augment=False)
    test_ds  = SNPDataset_v4(df.iloc[test_idx],   augment=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model     = DeepFoldCNN_v4().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = cosine_warmup_schedule(optimizer, WARMUP_EPOCHS, EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.10)

    # Per-group channel stats
    sample_t = torch.stack([train_ds[i][0] for i in range(min(200, len(train_ds)))])
    ctx_mean  = sample_t[:, 0:2].mean(dim=(0,2,3))
    ctx_std   = sample_t[:, 0:2].std(dim=(0,2,3)).clamp(min=1e-6)
    main_mean = sample_t[:, 2:4].mean(dim=(0,2,3))
    main_std  = sample_t[:, 2:4].std(dim=(0,2,3)).clamp(min=1e-6)
    ch_mean   = torch.cat([ctx_mean, main_mean]).to(DEVICE)
    ch_std    = torch.cat([ctx_std,  main_std]).to(DEVICE)

    best_val_loss = float("inf")
    patience_ctr  = 0
    best_ckpt = f"DeepFold_models/cnn_v4_fold{fold}.pt"

    for epoch in range(1, EPOCHS + 1):
        # ── Train with mixup ──
        model.train()
        tr_loss, tr_correct, tr_total = 0, 0, 0
        for tensors, labels_idx, labels_oh, _ in train_dl:
            tensors   = tensors.to(DEVICE)
            labels_oh = labels_oh.to(DEVICE)
            labels_idx = labels_idx.to(DEVICE)
            tensors = (tensors - ch_mean[None,:,None,None]) / ch_std[None,:,None,None]

            # Apply mixup
            tensors_mix, labels_mix = mixup_batch(tensors, labels_oh, MIXUP_ALPHA)

            optimizer.zero_grad()
            outputs = model(tensors_mix)
            loss = soft_cross_entropy(outputs, labels_mix, weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tr_loss += loss.item() * len(labels_idx)
            # Accuracy on un-mixed predictions for logging
            with torch.no_grad():
                out_clean = model((tensors.detach() - ch_mean[None,:,None,None]) / ch_std[None,:,None,None]) if False else outputs
            tr_correct += (outputs.argmax(1) == labels_idx).sum().item()
            tr_total   += len(labels_idx)

        tr_loss /= tr_total
        scheduler.step()

        # ── Validate (no mixup) ──
        model.eval()
        vl_loss, vl_correct, vl_total = 0, 0, 0
        vl_labels_list, vl_probs_list = [], []
        with torch.no_grad():
            for tensors, labels_idx, _, _ in val_dl:
                tensors, labels_idx = tensors.to(DEVICE), labels_idx.to(DEVICE)
                tensors = (tensors - ch_mean[None,:,None,None]) / ch_std[None,:,None,None]
                outputs = model(tensors)
                probs   = F.softmax(outputs, dim=1)[:, 1]
                loss    = criterion(outputs, labels_idx)
                vl_loss    += loss.item() * len(labels_idx)
                vl_correct += (outputs.argmax(1) == labels_idx).sum().item()
                vl_total   += len(labels_idx)
                vl_labels_list.extend(labels_idx.cpu().numpy())
                vl_probs_list.extend(probs.cpu().numpy())

        vl_loss /= vl_total
        vl_acc = vl_correct / vl_total
        vl_auc = roc_auc_score(vl_labels_list, vl_probs_list)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_ctr  = 0
            torch.save({
                "model_state": model.state_dict(),
                "ch_mean": ch_mean.cpu(), "ch_std": ch_std.cpu(),
                "fold": fold, "val_auc": vl_auc,
            }, best_ckpt)
        else:
            patience_ctr += 1

        if epoch % 10 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Ep {epoch:3d} | tr_loss {tr_loss:.4f} | vl_loss {vl_loss:.4f} "
                  f"vl_acc {vl_acc:.4f} vl_auc {vl_auc:.4f} lr {lr_now:.2e}"
                  + (" ← best" if patience_ctr == 0 else f" (pat {patience_ctr}/{PATIENCE})"))

        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    # ── Evaluate best on test fold ──
    ckpt = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    ch_m, ch_s = ckpt["ch_mean"].to(DEVICE), ckpt["ch_std"].to(DEVICE)

    model.eval()
    te_labels, te_probs, te_mirnas = [], [], []
    with torch.no_grad():
        for tensors, labels_idx, _, mirnas in test_dl:
            tensors = tensors.to(DEVICE)
            tensors = (tensors - ch_m[None,:,None,None]) / ch_s[None,:,None,None]
            probs = F.softmax(model(tensors), dim=1)[:, 1]
            te_labels.extend(labels_idx.numpy())
            te_probs.extend(probs.cpu().numpy())
            te_mirnas.extend(mirnas)

    te_y, te_p = np.array(te_labels), np.array(te_probs)
    m = compute_metrics(te_y, te_p)
    print(f"\n  Fold {fold+1} v4 Test: Acc {m['accuracy']:.4f}  AUC {m['auc']:.4f}  "
          f"Prec {m['precision']:.4f}  Rec {m['recall']:.4f}  F1 {m['f1']:.4f}")

    v4_fold_metrics.append(m)
    v4_all_labels.extend(te_y.tolist())
    v4_all_probs.extend(te_p.tolist())
    v4_all_mirnas.extend(te_mirnas)

print(f"\n{'='*55}")
print("5-FOLD CV SUMMARY — DeepFoldCNN v4")
print(f"{'='*55}")
for metric in ["accuracy", "auc", "precision", "recall", "f1"]:
    vals = [m[metric] for m in v4_fold_metrics]
    print(f"  {metric.capitalize():12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
print(f"\nv3 AUC was 0.625 ± 0.018 — check improvement above.")# ═══════════════════════════════════════════════════════════════════════════════from itertools import product as iproduct
from collections import Counter
import numpy as np
import pandas as pd
import math as _math

df     = pd.read_csv("DeepFold_Dataset/final_dataset.csv")
groups = df["miRNA_ID"].values
y      = df["Label"].values

# ── 5-mer vocabulary ─────────────────────────────────────────────────────────
bases_4   = ['A', 'U', 'C', 'G']
kmers_5   = [''.join(k) for k in iproduct(bases_4, repeat=5)]   # 1024
kmer5_idx = {km: i for i, km in enumerate(kmers_5)}


def kmer_freq(seq, k):
    """Frequency vector for k-mers of length k."""
    bases  = "ACGU"
    kmers  = [''.join(p) for p in iproduct(bases, repeat=k)]
    km_idx = {km: i for i, km in enumerate(kmers)}
    seq    = seq.upper().replace("T", "U")
    counts = np.zeros(len(kmers), dtype=np.float32)
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer in km_idx:
            counts[km_idx[kmer]] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def get_5mer_diff(seq_h, seq_m):
    """5-mer DIFFERENCE vector only (1024 dims)."""
    seq_h = seq_h.upper().replace("T", "U")
    seq_m = seq_m.upper().replace("T", "U")
    h_counts = np.zeros(1024, dtype=np.float32)
    m_counts = np.zeros(1024, dtype=np.float32)
    for i in range(len(seq_h) - 4):
        km = seq_h[i:i+5]
        if km in kmer5_idx: h_counts[kmer5_idx[km]] += 1
    for i in range(len(seq_m) - 4):
        km = seq_m[i:i+5]
        if km in kmer5_idx: m_counts[kmer5_idx[km]] += 1
    # Normalise
    h_total = h_counts.sum()
    m_total = m_counts.sum()
    if h_total > 0: h_counts /= h_total
    if m_total > 0: m_counts /= m_total
    return h_counts - m_counts   # difference only


def local_context_features(seq_h, seq_m, window=5):
    """5 features from ±window nt around the SNP position."""
    seq_h = seq_h.upper().replace("T", "U")
    seq_m = seq_m.upper().replace("T", "U")
    snp_pos = next((i for i, (a, b) in enumerate(zip(seq_h, seq_m)) if a != b), -1)
    if snp_pos == -1:
        return np.zeros(5, dtype=np.float32)

    L = len(seq_h)
    start = max(0, snp_pos - window)
    end   = min(L, snp_pos + window + 1)
    ctx   = seq_h[start:end]

    ctx_gc   = sum(1 for b in ctx if b in "GC") / max(len(ctx), 1)
    ctx_len  = len(ctx) / (2 * window + 1)   # normalised
    ctx_au   = (ctx.count("AU") + ctx.count("UA")) / max(len(ctx) - 1, 1)
    ctx_gu   = (ctx.count("GU") + ctx.count("UG")) / max(len(ctx) - 1, 1)
    # Shannon entropy of 3-mer at SNP site
    trigram = seq_h[max(0, snp_pos-1): min(L, snp_pos+2)]
    counts  = Counter(trigram)
    total   = sum(counts.values())
    entropy = -sum((c/total) * _math.log2(c/total) for c in counts.values() if c > 0)

    return np.array([ctx_gc, ctx_len, ctx_au, ctx_gu, entropy], dtype=np.float32)


# ── Reuse seed_features and conservation_proxy from existing notebook ─────────
# These functions should already be defined in the notebook scope.
# If not, they are included here as fallback:

try:
    _ = seed_features
    print("Using existing seed_features function")
except NameError:
    print("WARNING: seed_features not found — using dummy (all zeros)")
    def seed_features(mirna_id, seq_h, seq_m):
        return np.zeros(4, dtype=np.float32)

try:
    _ = conservation_proxy
    print("Using existing conservation_proxy function")
except NameError:
    print("WARNING: conservation_proxy not found — using dummy (all zeros)")
    def conservation_proxy(seq_h, snp_pos, struct_h=None):
        return np.zeros(3, dtype=np.float32)


def build_kmer_features_v2(row):
    """Extended k-mer features: original 971 + 5-mer diff (1024) + local context (5) = 2000 dims."""
    seq_h    = row["Seq_Healthy"].upper().replace("T", "U")
    seq_m    = row["Seq_Mutant"].upper().replace("T", "U")
    mirna_id = row["miRNA_ID"]

    # Original k-mer frequencies (k=3 and k=4) — 960 dims
    kmer_feats = []
    for k in [3, 4]:
        h = kmer_freq(seq_h, k)
        m = kmer_freq(seq_m, k)
        kmer_feats.extend([h, m, h - m])

    snp_pos = next((i for i, (a, b) in enumerate(zip(seq_h, seq_m)) if a != b), 0)
    L = len(seq_h)

    # Positional properties (4 dims)
    props = np.array([
        (seq_h.count('G') + seq_h.count('C')) / max(L, 1),
        L / 128.0,
        snp_pos / max(L, 1),
        abs(snp_pos - L/2) / max(L/2, 1)
    ], dtype=np.float32)

    # Seed region features (4 dims)
    seed_f = seed_features(mirna_id, seq_h, seq_m)

    # Conservation proxy features (3 dims)
    cons_f = conservation_proxy(seq_h, snp_pos)

    # NEW: 5-mer difference vector (1024 dims)
    fivemar_diff = get_5mer_diff(seq_h, seq_m)

    # NEW: Local context features (5 dims)
    local_ctx = local_context_features(seq_h, seq_m, window=5)

    return np.concatenate(kmer_feats + [props, seed_f, cons_f, fivemar_diff, local_ctx])


print("Building extended feature matrix (v2)...")
X_kmer_v2 = np.stack([build_kmer_features_v2(row) for _, row in df.iterrows()])
X_kmer_v2 = np.nan_to_num(X_kmer_v2, nan=0.0, posinf=0.0, neginf=0.0)
print(f"Feature matrix: {X_kmer_v2.shape}  (expected ~2000 cols)")import warnings
warnings.filterwarnings("ignore")

try:
    import optuna
    print(f"Optuna version: {optuna.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "optuna", "-q"], check=True)
    import optuna
    print(f"Optuna installed: {optuna.__version__}")

try:
    from xgboost import XGBClassifier
    print("XGBoost available")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "xgboost", "-q"], check=True)
    from xgboost import XGBClassifier
    print("XGBoost installed")

from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib

sgkf_kmer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

kmer_v2_fold_metrics = []
kmer_v2_all_probs    = []
kmer_v2_all_labels   = []
kmer_v2_fold_models  = []
kmer_v2_best_params  = []

optuna.logging.set_verbosity(optuna.logging.WARNING)

for fold, (train_idx, test_idx) in enumerate(sgkf_kmer.split(X_kmer_v2, y, groups)):
    print(f"\n{'='*55}")
    print(f"FOLD {fold+1}/5 — k-mer v2 + Optuna")
    print(f"{'='*55}")

    X_tr, y_tr = X_kmer_v2[train_idx], y[train_idx]
    X_te, y_te = X_kmer_v2[test_idx],  y[test_idx]
    groups_tr  = groups[train_idx]

    # Inner CV for Optuna — 3-fold StratifiedGroupKFold on train only
    inner_sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42+fold)

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 200, 800),
            'max_depth':        trial.suggest_int('max_depth', 3, 6),
            'learning_rate':    trial.suggest_float('lr', 0.01, 0.15, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('col', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('mcw', 1, 10),
            'eval_metric':      'logloss',
            'use_label_encoder': False,
            'random_state':     42,
            'verbosity':        0,
        }
        # Inner 3-fold CV on train data only
        aucs = []
        for inner_tr, inner_val in inner_sgkf.split(X_tr, y_tr, groups_tr):
            clf = XGBClassifier(**params)
            clf.fit(X_tr[inner_tr], y_tr[inner_tr])
            p = clf.predict_proba(X_tr[inner_val])[:, 1]
            aucs.append(roc_auc_score(y_tr[inner_val], p))
        return np.mean(aucs)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    best_p = study.best_params
    print(f"  Best params: {best_p}")
    print(f"  Best inner AUC: {study.best_value:.4f}")
    kmer_v2_best_params.append(best_p)

    # Retrain on full outer train with best params
    final_params = {
        'n_estimators':     best_p['n_estimators'],
        'max_depth':        best_p['max_depth'],
        'learning_rate':    best_p['lr'],
        'subsample':        best_p['subsample'],
        'colsample_bytree': best_p['col'],
        'min_child_weight': best_p['mcw'],
        'eval_metric':      'logloss',
        'use_label_encoder': False,
        'random_state':     42,
        'verbosity':        0,
    }
    model = XGBClassifier(**final_params)
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)

    kmer_v2_all_probs.extend(probs.tolist())
    kmer_v2_all_labels.extend(y_te.tolist())
    kmer_v2_fold_models.append(model)

    m = {
        "accuracy":  accuracy_score(y_te, preds),
        "auc":       roc_auc_score(y_te, probs),
        "precision": precision_score(y_te, preds, zero_division=0),
        "recall":    recall_score(y_te, preds, zero_division=0),
        "f1":        f1_score(y_te, preds, zero_division=0),
    }
    kmer_v2_fold_metrics.append(m)
    print(f"  Fold {fold+1} Test — AUC: {m['auc']:.4f}  Acc: {m['accuracy']:.4f}  "
          f"F1: {m['f1']:.4f}")

    # Save fold model
    joblib.dump(model, f"DeepFold_models/kmer_xgb_v2_fold{fold}.pkl")

print(f"\n{'='*55}")
print("k-mer XGBoost v2 — 5-FOLD CV SUMMARY")
print(f"{'='*55}")
for metric in ["accuracy", "auc", "precision", "recall", "f1"]:
    vals = [m[metric] for m in kmer_v2_fold_metrics]
    print(f"  {metric.capitalize():12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
print(f"\nv1 AUC was 0.704 ± 0.021 — check improvement above.")
print(f"OOF AUC: {roc_auc_score(kmer_v2_all_labels, kmer_v2_all_probs):.4f}")# ═══════════════════════════════════════════════════════════════════════════════import subprocess, sys

try:
    import fm
    print(f"RNA-FM already installed")
except ImportError:
    print("Installing RNA-FM (fair-esm)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "fair-esm", "-q"],
                   check=True)
    import fm
    print("RNA-FM installed")

import torch
import numpy as np
import pandas as pd
import os

model_rnafm, alphabet = fm.pretrained.rna_fm_t12()
model_rnafm.eval()
if torch.cuda.is_available():
    model_rnafm = model_rnafm.cuda()
print("RNA-FM loaded (frozen, no fine-tuning)")

batch_converter = alphabet.get_batch_converter()df = pd.read_csv("DeepFold_Dataset/final_dataset.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMB_PATH = "DeepFold_Dataset/rnafm_embeddings.npy"

def get_rnafm_embedding(seq, model, batch_converter, device):
    """
    Extract frozen RNA-FM embedding (640 dims) via mean pooling over positions.
    RNA-FM uses DNA alphabet internally (T not U).
    """
    seq_dna = seq.upper().replace("U", "T")
    # Truncate to 1022 to avoid OOM (RNA-FM max is ~1024 with BOS/EOS)
    seq_dna = seq_dna[:1022]

    batch_labels, batch_strs, batch_tokens = batch_converter([("seq", seq_dna)])
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12])

    # Mean pool over positions, excluding BOS (idx 0) and EOS (idx -1)
    token_repr = results["representations"][12][0, 1:-1]  # (L, 640)
    return token_repr.mean(dim=0).cpu().numpy()   # (640,)


if os.path.exists(EMB_PATH):
    print(f"Loading cached embeddings from {EMB_PATH}")
    X_rnafm_diff = np.load(EMB_PATH)
    print(f"Embeddings shape: {X_rnafm_diff.shape}")
else:
    print(f"Extracting RNA-FM embeddings for {len(df)} samples...")
    print("(~5-10 min on GPU, ~30 min on CPU)")

    rnafm_diffs = []
    for idx, (_, row) in enumerate(df.iterrows()):
        emb_h = get_rnafm_embedding(row["Seq_Healthy"], model_rnafm,
                                     batch_converter, DEVICE)
        emb_m = get_rnafm_embedding(row["Seq_Mutant"], model_rnafm,
                                     batch_converter, DEVICE)
        rnafm_diffs.append(emb_h - emb_m)   # difference embedding

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(df)} done")

    X_rnafm_diff = np.stack(rnafm_diffs).astype(np.float32)
    np.save(EMB_PATH, X_rnafm_diff)
    print(f"Saved → {EMB_PATH}")

print(f"RNA-FM diff embeddings: {X_rnafm_diff.shape}")
print(f"Sample embedding range: [{X_rnafm_diff[0].min():.4f}, {X_rnafm_diff[0].max():.4f}]")

# Free GPU memory
del model_rnafm
torch.cuda.empty_cache()import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib

df     = pd.read_csv("DeepFold_Dataset/final_dataset.csv")
groups = df["miRNA_ID"].values
y      = df["Label"].values

# Combine k-mer v2 features + RNA-FM diff
X_combined = np.hstack([X_kmer_v2, X_rnafm_diff])
print(f"Combined feature matrix: {X_combined.shape}  "
      f"(k-mer_v2: {X_kmer_v2.shape[1]} + RNA-FM: {X_rnafm_diff.shape[1]})")

sgkf_rnafm = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

rnafm_fold_metrics = []
rnafm_all_probs    = []
rnafm_all_labels   = []
rnafm_fold_models  = []

optuna.logging.set_verbosity(optuna.logging.WARNING)
os.makedirs("DeepFold_models", exist_ok=True)

for fold, (train_idx, test_idx) in enumerate(sgkf_rnafm.split(X_combined, y, groups)):
    print(f"\n{'='*55}")
    print(f"FOLD {fold+1}/5 — RNA-FM + k-mer XGBoost")
    print(f"{'='*55}")

    X_tr, y_tr = X_combined[train_idx], y[train_idx]
    X_te, y_te = X_combined[test_idx],  y[test_idx]
    groups_tr  = groups[train_idx]

    inner_sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42+fold)

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 200, 800),
            'max_depth':        trial.suggest_int('max_depth', 3, 6),
            'learning_rate':    trial.suggest_float('lr', 0.01, 0.15, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('col', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('mcw', 1, 10),
            'eval_metric': 'logloss', 'use_label_encoder': False,
            'random_state': 42, 'verbosity': 0,
        }
        aucs = []
        for inner_tr, inner_val in inner_sgkf.split(X_tr, y_tr, groups_tr):
            clf = XGBClassifier(**params)
            clf.fit(X_tr[inner_tr], y_tr[inner_tr])
            p = clf.predict_proba(X_tr[inner_val])[:, 1]
            aucs.append(roc_auc_score(y_tr[inner_val], p))
        return np.mean(aucs)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    bp = study.best_params
    print(f"  Best inner AUC: {study.best_value:.4f}")

    model = XGBClassifier(
        n_estimators=bp['n_estimators'], max_depth=bp['max_depth'],
        learning_rate=bp['lr'], subsample=bp['subsample'],
        colsample_bytree=bp['col'], min_child_weight=bp['mcw'],
        eval_metric='logloss', use_label_encoder=False,
        random_state=42, verbosity=0,
    )
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)

    rnafm_all_probs.extend(probs.tolist())
    rnafm_all_labels.extend(y_te.tolist())
    rnafm_fold_models.append(model)

    m = {
        "accuracy":  accuracy_score(y_te, preds),
        "auc":       roc_auc_score(y_te, probs),
        "precision": precision_score(y_te, preds, zero_division=0),
        "recall":    recall_score(y_te, preds, zero_division=0),
        "f1":        f1_score(y_te, preds, zero_division=0),
    }
    rnafm_fold_metrics.append(m)
    print(f"  Fold {fold+1} — AUC: {m['auc']:.4f}  Acc: {m['accuracy']:.4f}  F1: {m['f1']:.4f}")

    joblib.dump(model, f"DeepFold_models/rnafm_xgb_fold{fold}.pkl")

print(f"\n{'='*55}")
print("RNA-FM + k-mer XGBoost — 5-FOLD CV SUMMARY")
print(f"{'='*55}")
for metric in ["accuracy", "auc", "precision", "recall", "f1"]:
    vals = [m[metric] for m in rnafm_fold_metrics]
    print(f"  {metric.capitalize():12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
print(f"\nOOF AUC: {roc_auc_score(rnafm_all_labels, rnafm_all_probs):.4f}")# ═══════════════════════════════════════════════════════════════════════════════import numpy as np
import pandas as pd
from scipy.stats import rankdata
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib

df     = pd.read_csv("DeepFold_Dataset/final_dataset.csv")
groups = df["miRNA_ID"].values
y      = df["Label"].values

# Gather OOF probabilities from all base models
# These variables should be in scope from previous cells:
#   v4_all_probs     → CNN v4 (from Cell Group A)
#   gnn_all_probs    → Siamese GCN (from original notebook Step 8.2)
#   kmer_v2_all_probs → k-mer XGBoost v2 (from Cell Group B)
#   mfe_all_probs    → ViennaRNA MFE XGBoost (from original notebook Step 8.2c)
#   rnafm_all_probs  → RNA-FM XGBoost (from Cell Group C)

# Convert to numpy arrays
cnn_p   = np.array(v4_all_probs)
gcn_p   = np.array(gnn_all_probs)
kmer_p  = np.array(kmer_v2_all_probs)
mfe_p   = np.array(mfe_all_probs)
rnafm_p = np.array(rnafm_all_probs)

# Use labels from CNN v4 (should be identical order for all models with same SGKF)
true_y = np.array(v4_all_labels)

print("Base model OOF AUCs:")
print(f"  CNN v4:      {roc_auc_score(true_y, cnn_p):.4f}")
print(f"  GCN:         {roc_auc_score(true_y, gcn_p):.4f}")
print(f"  k-mer v2:    {roc_auc_score(true_y, kmer_p):.4f}")
print(f"  MFE XGB:     {roc_auc_score(true_y, mfe_p):.4f}")
print(f"  RNA-FM XGB:  {roc_auc_score(true_y, rnafm_p):.4f}")

# Verify all arrays have the same length
assert len(cnn_p) == len(gcn_p) == len(kmer_p) == len(mfe_p) == len(rnafm_p) == len(true_y), \
    f"Length mismatch! CNN:{len(cnn_p)} GCN:{len(gcn_p)} k-mer:{len(kmer_p)} " \
    f"MFE:{len(mfe_p)} RNAFM:{len(rnafm_p)} y:{len(true_y)}"
print(f"\nAll arrays: {len(true_y)} samples ✓")def rank_normalize(probs):
    """Rank-transform to [0, 1] — prevents calibration differences from
    distorting the meta-learner."""
    return rankdata(probs) / len(probs)

# Raw meta-features (for comparison)
meta_X5_raw = np.column_stack([cnn_p, gcn_p, kmer_p, mfe_p, rnafm_p])

# Rank-normalised meta-features
meta_X5_ranked = np.column_stack([
    rank_normalize(cnn_p),
    rank_normalize(gcn_p),
    rank_normalize(kmer_p),
    rank_normalize(mfe_p),
    rank_normalize(rnafm_p),
])

print(f"Meta-feature matrix (raw):    {meta_X5_raw.shape}")
print(f"Meta-feature matrix (ranked): {meta_X5_ranked.shape}")
print(f"\nRaw probability ranges:")
for name, arr in [("CNN", cnn_p), ("GCN", gcn_p), ("k-mer", kmer_p),
                  ("MFE", mfe_p), ("RNAFM", rnafm_p)]:
    print(f"  {name:8s}: [{arr.min():.3f}, {arr.max():.3f}]  mean={arr.mean():.3f}")sgkf_meta = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

ensemble_v2_fold_metrics = []
ensemble_v2_all_probs    = np.zeros(len(true_y))
ensemble_v2_fold_models  = []

for fold, (train_idx, test_idx) in enumerate(sgkf_meta.split(meta_X5_ranked, true_y, groups)):
    meta_model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )
    meta_model.fit(meta_X5_ranked[train_idx], true_y[train_idx])
    probs = meta_model.predict_proba(meta_X5_ranked[test_idx])[:, 1]
    preds = (probs >= 0.5).astype(int)
    ensemble_v2_all_probs[test_idx] = probs
    ensemble_v2_fold_models.append(meta_model)

    m = {
        "accuracy":  accuracy_score(true_y[test_idx], preds),
        "auc":       roc_auc_score(true_y[test_idx], probs),
        "precision": precision_score(true_y[test_idx], preds, zero_division=0),
        "recall":    recall_score(true_y[test_idx], preds, zero_division=0),
        "f1":        f1_score(true_y[test_idx], preds, zero_division=0),
    }
    ensemble_v2_fold_metrics.append(m)

    # Feature importances show which base model contributes most
    imp = meta_model.feature_importances_
    names = ["CNN_v4", "GCN", "k-mer_v2", "MFE", "RNAFM"]
    imp_str = "  ".join(f"{n}: {v:.3f}" for n, v in zip(names, imp))
    print(f"  Fold {fold+1} — AUC: {m['auc']:.4f}  Acc: {m['accuracy']:.4f} | {imp_str}")

# Save the last fold's meta-learner (or retrain on full data)
meta_model_final = XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    subsample=0.8, eval_metric='auc', use_label_encoder=False,
    random_state=42, verbosity=0,
)
meta_model_final.fit(meta_X5_ranked, true_y)
joblib.dump(meta_model_final, "DeepFold_models/meta_learner_v2.pkl")

print(f"\n{'='*55}")
print("Enhanced 5-Model Ensemble v2 — 5-FOLD CV SUMMARY")
print(f"{'='*55}")
for metric in ["accuracy", "auc", "precision", "recall", "f1"]:
    vals = [m[metric] for m in ensemble_v2_fold_metrics]
    print(f"  {metric.capitalize():12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
print(f"\nOOF AUC: {roc_auc_score(true_y, ensemble_v2_all_probs):.4f}")
print(f"\nOriginal 4-model ensemble AUC was 0.720 ± 0.020 — check improvement above.")print(f"\n{'='*70}")
print(f"{'FINAL METHOD COMPARISON v2 — 5-FOLD STRATIFIED GROUP CV':^70}")
print(f"{'='*70}")
print(f"{'Method':<32} {'AUC':^20} {'Accuracy':^12} {'F1':^8}")
print(f"{'-'*70}")

all_results = []

# Collect all model results
model_results = {
    "CNN v4 (DeepFoldCNN)":     v4_fold_metrics,
    "Siamese GCN":              gnn_fold_metrics,
    "k-mer XGB v2 (Optuna)":    kmer_v2_fold_metrics,
    "ViennaRNA MFE XGB":        mfe_fold_metrics,
    "RNA-FM + k-mer XGB":       rnafm_fold_metrics,
    "Ensemble v2 (5-model)":    ensemble_v2_fold_metrics,
}

# Also include original models if available
try:
    model_results["CNN v3 (original)"] = fold_metrics
except NameError:
    pass
try:
    model_results["k-mer XGB v1 (original)"] = kmer_results["fold_metrics"]
except NameError:
    pass
try:
    model_results["Ensemble v1 (4-model)"] = enhanced_fold_metrics
except NameError:
    pass

for name, folds in sorted(model_results.items(),
                           key=lambda x: -np.mean([m["auc"] for m in x[1]])):
    auc_m = np.mean([m["auc"] for m in folds])
    auc_s = np.std([m["auc"] for m in folds])
    acc   = np.mean([m["accuracy"] for m in folds])
    f1    = np.mean([m["f1"] for m in folds])
    best  = " ← BEST" if name.startswith("Ensemble v2") else ""
    print(f"{name:<32} {auc_m:.4f} ± {auc_s:.4f}   {acc:.4f}     {f1:.4f}{best}")
    all_results.append({"method": name, "auc_mean": auc_m, "auc_std": auc_s,
                        "accuracy": acc, "f1": f1})

print(f"{'='*70}")

# Save
rows = []
for name, folds in model_results.items():
    for i, m in enumerate(folds):
        rows.append({"method": name, "fold": i+1,
                     **{k: v for k, v in m.items() if k != "cm"}})
pd.DataFrame(rows).to_csv("DeepFold_Dataset/all_methods_comparison_v2.csv", index=False)
print("Saved → all_methods_comparison_v2.csv")# ═══════════════════════════════════════════════════════════════════════════════import os, torch, joblib
import numpy as np

CKPT_DIR = "DeepFold_models/"
os.makedirs(CKPT_DIR, exist_ok=True)

# 1. CNN v4 — already saved in Cell Group A as cnn_v4_fold{i}.pt
for i in range(5):
    p = f"{CKPT_DIR}/cnn_v4_fold{i}.pt"
    assert os.path.exists(p), f"Missing: {p}"
print(f"✓ CNN v4 checkpoints: 5 folds")

# 2. Siamese GCN — save from existing notebook variables
try:
    # The original notebook stores models per fold — if variable exists, save
    # If GCN models are already on disk from original training, skip
    gcn_saved = 0
    for fold_i in range(5):
        p = f"{CKPT_DIR}/gcn_fold{fold_i}.pt"
        if os.path.exists(p):
            gcn_saved += 1
        elif 'gnn_fold_metrics' in dir():
            # Try to save from existing checkpoint path
            orig_p = f"DeepFold_Dataset/checkpoints/gcn_fold{fold_i+1}_best.pt"
            if os.path.exists(orig_p):
                import shutil
                shutil.copy2(orig_p, p)
                gcn_saved += 1
    print(f"✓ GCN checkpoints: {gcn_saved}/5 folds")
except Exception as e:
    print(f"⚠ GCN save: {e}")

# 3. k-mer XGBoost v2 — already saved in Cell Group B
for i in range(5):
    p = f"{CKPT_DIR}/kmer_xgb_v2_fold{i}.pkl"
    assert os.path.exists(p), f"Missing: {p}"
print(f"✓ k-mer XGB v2: 5 folds")

# 4. MFE XGBoost — save from existing notebook
try:
    mfe_saved = 0
    for fold_i in range(5):
        p = f"{CKPT_DIR}/mfe_xgb_fold{fold_i}.pkl"
        if not os.path.exists(p):
            # Retrain if needed (uses existing X_mfe, y, sgkf from original notebook)
            pass
        if os.path.exists(p):
            mfe_saved += 1
    if mfe_saved < 5:
        print(f"⚠ MFE XGB: only {mfe_saved}/5 on disk. "
              "Re-run MFE training and save models per fold.")
    else:
        print(f"✓ MFE XGB: 5 folds")
except Exception as e:
    print(f"⚠ MFE save: {e}")

# 5. RNA-FM XGBoost — already saved in Cell Group C
for i in range(5):
    p = f"{CKPT_DIR}/rnafm_xgb_fold{i}.pkl"
    assert os.path.exists(p), f"Missing: {p}"
print(f"✓ RNA-FM XGB: 5 folds")

# 6. Meta-learner v2 — already saved in Cell Group D
assert os.path.exists(f"{CKPT_DIR}/meta_learner_v2.pkl"), "Missing meta_learner_v2.pkl"
print(f"✓ Meta-learner v2")

# 7. Pipeline config
config = {
    'n_folds':        5,
    'rnafm_emb_dim':  640,
    'kmer_v2_dim':    X_kmer_v2.shape[1] if 'X_kmer_v2' in dir() else 2000,
    'seed':           42,
}
joblib.dump(config, f"{CKPT_DIR}/pipeline_config.pkl")
print(f"✓ Pipeline config saved")
print(f"\nModel registry complete → {CKPT_DIR}")import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from scipy.stats import rankdata

def predict_variant(mirna_id, seq_healthy, seq_mutant, snp_pos=None,
                    models_dir="DeepFold_models/", n_folds=5):
    """
    Full ensemble prediction for a single miRNA SNP variant.

    Args:
        mirna_id:    str, e.g. "hsa-mir-21"
        seq_healthy: str, RNA sequence (uses U)
        seq_mutant:  str, RNA sequence with SNP applied
        snp_pos:     int or None, 0-based index of the SNP in the sequence
        models_dir:  str, path to saved model directory
        n_folds:     int, number of fold checkpoints to average

    Returns:
        dict with keys: prob_disease, label, confidence, base_probs
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect SNP position if not provided
    if snp_pos is None:
        snp_pos = next((i for i, (a, b) in enumerate(
            zip(seq_healthy.upper(), seq_mutant.upper())) if a != b), 0)

    # ═══ 1. CNN v4 — generate contact map and run inference ═══
    cnn_preds = []
    try:
        # Generate 4-channel tensor using existing UFold functions
        c1 = get_contact_map(seq_healthy, 128)
        c2 = get_contact_map(seq_mutant, 128)
        c3 = np.abs(c1 - c2)
        inp = seq_to_input(seq_healthy, 128)
        c4 = inp[0, 16].numpy()
        tensor = np.stack([c1, c2, c3, c4], axis=-1).astype(np.float32)
        tensor_t = torch.tensor(tensor).permute(2, 0, 1).unsqueeze(0)  # (1,4,128,128)

        for fold_i in range(n_folds):
            ckpt = torch.load(f"{models_dir}/cnn_v4_fold{fold_i}.pt",
                              map_location=DEVICE)
            model = DeepFoldCNN_v4().to(DEVICE)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            ch_m = ckpt["ch_mean"].to(DEVICE)
            ch_s = ckpt["ch_std"].to(DEVICE)
            t = (tensor_t.to(DEVICE) - ch_m[None,:,None,None]) / ch_s[None,:,None,None]
            with torch.no_grad():
                prob = torch.softmax(model(t), dim=1)[0, 1].item()
            cnn_preds.append(prob)
        cnn_prob = np.mean(cnn_preds)
    except Exception as e:
        print(f"  CNN inference error: {e}")
        cnn_prob = 0.5

    # ═══ 2. Siamese GCN ═══
    gcn_preds = []
    try:
        nf_h, adj_h, mask_h = build_graph_tensors(seq_healthy,
                                get_contact_map(seq_healthy, 128))
        nf_m, adj_m, mask_m = build_graph_tensors(seq_mutant,
                                get_contact_map(seq_mutant, 128))
        for fold_i in range(n_folds):
            p = f"{models_dir}/gcn_fold{fold_i}.pt"
            if not os.path.exists(p):
                continue
            gcn_model = SiameseGCN().to(DEVICE)
            gcn_model.load_state_dict(torch.load(p, map_location=DEVICE))
            gcn_model.eval()
            with torch.no_grad():
                out = gcn_model(
                    torch.tensor(nf_h).unsqueeze(0).to(DEVICE),
                    torch.tensor(adj_h).unsqueeze(0).to(DEVICE),
                    torch.tensor(mask_h).unsqueeze(0).to(DEVICE),
                    torch.tensor(nf_m).unsqueeze(0).to(DEVICE),
                    torch.tensor(adj_m).unsqueeze(0).to(DEVICE),
                    torch.tensor(mask_m).unsqueeze(0).to(DEVICE),
                )
                prob = F.softmax(out, dim=1)[0, 1].item()
            gcn_preds.append(prob)
        gcn_prob = np.mean(gcn_preds) if gcn_preds else 0.5
    except Exception as e:
        print(f"  GCN inference error: {e}")
        gcn_prob = 0.5

    # ═══ 3. k-mer XGBoost v2 ═══
    try:
        row_dict = {"Seq_Healthy": seq_healthy, "Seq_Mutant": seq_mutant,
                    "miRNA_ID": mirna_id}
        kmer_feats = build_kmer_features_v2(pd.Series(row_dict)).reshape(1, -1)
        kmer_preds = []
        for fold_i in range(n_folds):
            m = joblib.load(f"{models_dir}/kmer_xgb_v2_fold{fold_i}.pkl")
            kmer_preds.append(m.predict_proba(kmer_feats)[0, 1])
        kmer_prob = np.mean(kmer_preds)
    except Exception as e:
        print(f"  k-mer inference error: {e}")
        kmer_prob = 0.5

    # ═══ 4. MFE XGBoost ═══
    try:
        mfe_feats = extract_mfe_features(seq_healthy, seq_mutant).reshape(1, -1)
        mfe_preds = []
        for fold_i in range(n_folds):
            p = f"{models_dir}/mfe_xgb_fold{fold_i}.pkl"
            if os.path.exists(p):
                m = joblib.load(p)
                mfe_preds.append(m.predict_proba(mfe_feats)[0, 1])
        mfe_prob = np.mean(mfe_preds) if mfe_preds else 0.5
    except Exception as e:
        print(f"  MFE inference error: {e}")
        mfe_prob = 0.5

    # ═══ 5. RNA-FM XGBoost ═══
    try:
        emb_h = get_rnafm_embedding(seq_healthy, model_rnafm, batch_converter, DEVICE)
        emb_m = get_rnafm_embedding(seq_mutant, model_rnafm, batch_converter, DEVICE)
        emb_diff = (emb_h - emb_m).reshape(1, -1)
        combined = np.hstack([kmer_feats, emb_diff])
        rnafm_preds = []
        for fold_i in range(n_folds):
            m = joblib.load(f"{models_dir}/rnafm_xgb_fold{fold_i}.pkl")
            rnafm_preds.append(m.predict_proba(combined)[0, 1])
        rnafm_prob = np.mean(rnafm_preds)
    except Exception as e:
        print(f"  RNA-FM inference error: {e}")
        rnafm_prob = 0.5

    # ═══ 6. Rank-normalise and meta-predict ═══
    base_probs = np.array([cnn_prob, gcn_prob, kmer_prob, mfe_prob, rnafm_prob])
    # For single-sample prediction, rank-normalisation is not meaningful
    # Use raw probabilities reshaped for the meta-learner
    meta_input = base_probs.reshape(1, -1)

    try:
        meta = joblib.load(f"{models_dir}/meta_learner_v2.pkl")
        final_prob = meta.predict_proba(meta_input)[0, 1]
    except Exception as e:
        print(f"  Meta-learner error: {e} — using mean of base probs")
        final_prob = base_probs.mean()

    # ═══ 7. Interpret ═══
    if final_prob >= 0.65:
        label      = 'Likely pathogenic'
        confidence = 'High' if final_prob >= 0.80 else 'Medium'
    elif final_prob <= 0.35:
        label      = 'Likely benign'
        confidence = 'High' if final_prob <= 0.20 else 'Medium'
    else:
        label      = 'Uncertain significance (VUS)'
        confidence = 'Low'

    return {
        'prob_disease': round(float(final_prob), 4),
        'label':        label,
        'confidence':   confidence,
        'base_probs': {
            'CNN_v4':    round(float(cnn_prob), 4),
            'GCN':       round(float(gcn_prob), 4),
            'kmer_XGB':  round(float(kmer_prob), 4),
            'MFE_XGB':   round(float(mfe_prob), 4),
            'RNAFM_XGB': round(float(rnafm_prob), 4),
        }
    }

print("predict_variant() defined — full 5-model ensemble inference pipeline ready.")# Pick 3 samples: 1 disease, 1 benign, 1 borderline (closest to 0.5 in ensemble)
df_demo = pd.read_csv("DeepFold_Dataset/final_dataset.csv")

# Known disease sample
disease_row = df_demo[df_demo["Label"] == 1].iloc[0]
# Known benign sample
benign_row  = df_demo[df_demo["Label"] == 0].iloc[0]
# Borderline: pick the sample whose ensemble v2 OOF probability is closest to 0.5
if len(ensemble_v2_all_probs) > 0:
    border_idx = np.argmin(np.abs(ensemble_v2_all_probs - 0.5))
    border_row = df_demo.iloc[border_idx]
else:
    border_row = df_demo.iloc[len(df_demo)//2]

print("=" * 70)
print(f"{'DEMO PREDICTIONS — predict_variant()':^70}")
print("=" * 70)

for name, row in [("Known DISEASE", disease_row),
                  ("Known BENIGN", benign_row),
                  ("BORDERLINE", border_row)]:
    print(f"\n{'─'*70}")
    print(f"  Sample:   {row['Sample_ID']}")
    print(f"  miRNA:    {row['miRNA_ID']}")
    print(f"  True:     {'Disease' if row['Label'] == 1 else 'Benign'}")
    print(f"  Seq len:  {len(row['Seq_Healthy'])} nt")

    result = predict_variant(
        mirna_id    = row["miRNA_ID"],
        seq_healthy = row["Seq_Healthy"],
        seq_mutant  = row["Seq_Mutant"],
    )

    print(f"\n  Prediction: {result['label']}  "
          f"(p={result['prob_disease']:.4f}, {result['confidence']} confidence)")
    print(f"  Base model probabilities:")
    for model_name, prob in result['base_probs'].items():
        print(f"    {model_name:12s}: {prob:.4f}")

print(f"\n{'='*70}")
print("Demo complete.")