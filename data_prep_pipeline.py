import sys, os
import pandas as pd
import numpy as np
from Bio import SeqIO
from tqdm import tqdm

archive_dir = "d:/Capstone/archive"
kaggle_raw_dir = "d:/Capstone/kaggle/working/DeepFold_Dataset/raw_data"
out_dir = "DeepFold_Dataset"
os.makedirs(out_dir, exist_ok=True)

print("Step 1: Loading disease and benign SNPs")
# 1. Load disease SNPs
disease_df = pd.read_csv(f"{archive_dir}/DeepFold_pre_miRNA_disease_SNPs.csv")
disease_df.columns = disease_df.columns.str.lower().str.strip()
if "chr" in disease_df.columns:
    disease_df["chr"] = disease_df["chr"].astype(str).str.replace("chr", "", regex=False)
elif "chrom" in disease_df.columns:
    disease_df["chr"] = disease_df["chrom"].astype(str).str.replace("chr", "", regex=False)
disease_df = disease_df.drop_duplicates(subset=["chr", "position", "ref", "alt"]).copy()
disease_df["label"] = "disease"
disease_df["class"] = 1

# 2. Load benign SNPs 
benign_df = pd.read_csv(f"{kaggle_raw_dir}/benign_snps.csv")
benign_df.columns = benign_df.columns.str.lower().str.strip()
if "chr" in benign_df.columns:
    benign_df["chr"] = benign_df["chr"].astype(str).str.replace("chr", "", regex=False)
elif "chrom" in benign_df.columns:
    benign_df["chr"] = benign_df["chrom"].astype(str).str.replace("chr", "", regex=False)
benign_df = benign_df.drop_duplicates(subset=["chr", "position", "ref", "alt"]).copy()
benign_df["label"] = "benign"
benign_df["class"] = 0

print(f"Loaded {len(disease_df)} disease SNPs and {len(benign_df)} benign SNPs")

# Remove disease overlap from benign set
overlap = benign_df.merge(disease_df, on=["chr", "position", "ref", "alt"], how="inner")
if not overlap.empty:
    print(f"Removing {len(overlap)} overlapping disease SNPs from benign set")
    benign_df = benign_df[~benign_df.set_index(["chr", "position", "ref", "alt"]).index.isin(overlap.set_index(["chr", "position", "ref", "alt"]).index)]

combined = pd.concat([disease_df, benign_df], ignore_index=True)

# 3. Parse GFF3 to map to hairpin
print("Step 3: Parsing GFF3 to map SNPs to miRNA hairpins")
gff_path = f"{archive_dir}/hsa.gff3"
gff = pd.read_csv(gff_path, sep="\t", comment="#", header=None)
gff.columns = ["chr", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]
mirna_regions = gff[gff["type"] == "miRNA_primary_transcript"].copy()
mirna_regions["chr"] = mirna_regions["chr"].astype(str).str.replace("chr", "", regex=False)
mirna_regions["mirna_id"] = mirna_regions["attributes"].str.extract(r'Name=([^;]+)')

# Cross join logic to map efficiently
# We can iterate or do interval mapping. Since dataset is small ~7000 snps, we can just iterate.
mapped = []
for idx, row in tqdm(combined.iterrows(), total=len(combined), desc="Mapping SNPs..."):
    chrom = str(row["chr"])
    pos = int(row["position"])
    match = mirna_regions[(mirna_regions["chr"] == chrom) & (mirna_regions["start"] <= pos) & (mirna_regions["end"] >= pos)]
    for _, m_row in match.iterrows():
        rel_pos = pos - m_row["start"] if m_row["strand"] == "+" else m_row["end"] - pos
        m = row.to_dict()
        m["mirna_id"] = m_row["mirna_id"]
        m["strand"] = m_row["strand"]
        m["rel_pos"] = rel_pos # 0-indexed relative position!
        mapped.append(m)

mapped_df = pd.DataFrame(mapped)
mapped_df = mapped_df.drop_duplicates(subset=["chr", "position", "ref", "alt", "mirna_id"])
print(f"Mapped {len(mapped_df)} SNPs to hairpins")

# 4. Generate sequences
print("Step 4: Loading hairpin sequences")
hairpin_dict = {}
for rec in SeqIO.parse(f"{archive_dir}/hairpin.fa", "fasta"):
    # miRBase ids are like hsa-mir-21, ID is "hsa-mir-21" etc.
    name = rec.id.split()[0].lower()
    hairpin_dict[name] = str(rec.seq).upper().replace("U", "T")

final_dataset = []
failed_validation = 0
for idx, row in tqdm(mapped_df.iterrows(), total=len(mapped_df), desc="Generating sequences..."):
    mid = row["mirna_id"].lower()
    if mid not in hairpin_dict:
        # try without hsa
        continue
    seq = hairpin_dict[mid]
    rel_pos = int(row["rel_pos"])
    
    
    if len(row["ref"]) != 1 or len(row["alt"]) != 1:
        continue

    if rel_pos >= len(seq) or rel_pos < 0:
        continue
        
    ref_allele = row["ref"].upper().replace("U", "T")
    alt_allele = row["alt"].upper().replace("U", "T")
    
    if row["strand"] == "-":
        complement = {"A":"T", "T":"A", "C":"G", "G":"C", "N":"N"}
        ref_allele = complement.get(ref_allele, ref_allele)
        alt_allele = complement.get(alt_allele, alt_allele)
        
    actual_ref = seq[rel_pos]
    if actual_ref != ref_allele:
        failed_validation += 1
        continue
        
    mut_seq = seq[:rel_pos] + alt_allele + seq[rel_pos+1:]
    
    # Extra safety length check
    if len(seq) != len(mut_seq):
        continue
    
    m = row.to_dict()
    m["Sample_ID"] = f"{mid}_{m['label']}_{idx}"
    m["Seq_Healthy"] = seq
    m["Seq_Mutant"] = mut_seq
    final_dataset.append(m)

print(f"Failed reference validation: {failed_validation}")
final_df = pd.DataFrame(final_dataset)

# 5. Balance dataset and save
print("Step 5: Balancing dataset")
disease = final_df[final_df["class"] == 1]
benign = final_df[final_df["class"] == 0]

min_count = min(len(disease), len(benign))
if min_count > 0:
    disease = disease.sample(n=min_count, random_state=42)
    benign = benign.sample(n=min_count, random_state=42)
    
final_df = pd.concat([disease, benign]).sample(frac=1, random_state=42).reset_index(drop=True)

# Integrity checks
print(f"Final Count: {len(final_df)} (Disease: {len(disease)}, Benign: {len(benign)})")
assert len(disease) == len(benign), "Classes are not balanced!"
for _, row in final_df.iterrows():
    diffs = sum(1 for a, b in zip(row["Seq_Healthy"], row["Seq_Mutant"]) if a != b)
    assert diffs == 1, f"Expected 1 diff, got {diffs}"

final_df.to_csv(f"{out_dir}/final_dataset.csv", index=False)
print(f"Saved to {out_dir}/final_dataset.csv")

# Now trigger generate_data.py to make tensors
os.system("python generate_data.py")
