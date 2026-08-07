import sys, os, torch
sys.path.insert(0, 'UFold')
import numpy as np
import pandas as pd
from Network import U_Net as FCNNet

WEIGHTS_PATH = "archive/ufold_train_alldata.pt"
CSV_PATH = "DeepFold_Dataset/final_dataset.csv"
OUT_DIR = "DeepFold_Dataset/processed_maps/npy"
TARGET_SIZE = 128
DEVICE = torch.device("cpu")

os.makedirs(OUT_DIR, exist_ok=True)

# Load UFold
contact_net = FCNNet(img_ch=17)
state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
if any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
contact_net.load_state_dict(state_dict)
contact_net.to(DEVICE)
contact_net.eval()
print(f"UFold loaded on {DEVICE}")

def seq_to_input(seq, max_len=128):
    seq = seq.upper().replace("U", "T")
    L = min(len(seq), max_len)
    seq = seq[:L]
    base_map = {'A': 0, 'T': 1, 'U': 1, 'C': 2, 'G': 3}
    one_hot = np.zeros((4, L), dtype=np.float32)
    for i, b in enumerate(seq):
        if b in base_map:
            one_hot[base_map[b], i] = 1.0
    mat = np.zeros((16, L, L), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            mat[i*4+j] = np.outer(one_hot[i], one_hot[j])
    canonical = {(0,1),(1,0),(2,3),(3,2),(3,1),(1,3)}
    pair_mask = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        for j in range(L):
            if abs(i-j) >= 4:
                bi = base_map.get(seq[i], -1)
                bj = base_map.get(seq[j], -1)
                if (bi, bj) in canonical:
                    pair_mask[i, j] = 1.0
    inp = np.concatenate([mat, pair_mask[np.newaxis]], axis=0)
    padded = np.zeros((17, max_len, max_len), dtype=np.float32)
    padded[:, :L, :L] = inp
    return torch.tensor(padded).unsqueeze(0)

def get_contact_map(seq, max_len=128):
    inp = seq_to_input(seq, max_len).to(DEVICE)
    with torch.no_grad():
        pred = contact_net(inp)
        score_map = torch.sigmoid(pred)
    return score_map.squeeze().cpu().numpy()

df = pd.read_csv(CSV_PATH)
num_samples = len(df)
print(f"Processing {num_samples} samples...")

for idx, row in df.iterrows():
    sample_id = row["Sample_ID"]
    out_path  = os.path.join(OUT_DIR, f"{sample_id}.npy")
    if os.path.exists(out_path): continue
    try:
        c1 = get_contact_map(row["Seq_Healthy"], TARGET_SIZE)
        c2 = get_contact_map(row["Seq_Mutant"],  TARGET_SIZE)
        c3 = np.abs(c1 - c2)
        inp_h = seq_to_input(row["Seq_Healthy"], TARGET_SIZE)
        c4  = inp_h[0, 16].numpy()
        tensor = np.stack([c1, c2, c3, c4], axis=-1).astype(np.float32)
        np.save(out_path, tensor)
    except Exception as e:
        print(f"Failed {sample_id}: {e}")
    if (idx + 1) % 50 == 0:
        print(f"Progress: {idx+1}/{num_samples}")

print("Data generation complete.")
