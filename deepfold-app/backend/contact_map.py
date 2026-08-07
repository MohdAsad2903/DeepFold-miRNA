import os
import sys
import torch
import numpy as np

# Add UFold to path to import Network.py
# Assuming backend is in <root>/deepfold-app/backend
# and UFold is in <root>/UFold
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UFOLD_PATH = os.path.join(ROOT_PATH, "UFold")
if UFOLD_PATH not in sys.path:
    sys.path.insert(0, UFOLD_PATH)

try:
    from Network import U_Net
except ImportError:
    # Absolute import fallback for when running from within backend package
    import sys
    sys.path.append(os.path.join(os.getcwd(), ".."))
    from UFold.Network import U_Net

# CONFIG
WEIGHTS_PATH = os.path.join(ROOT_PATH, "archive", "ufold_train_alldata.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instance
_contact_net = None

def _get_model():
    global _contact_net
    if _contact_net is None:
        _contact_net = U_Net(img_ch=17)
        if os.path.exists(WEIGHTS_PATH):
            state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
            # Remove module prefix if present
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            _contact_net.load_state_dict(state_dict)
            _contact_net.to(DEVICE)
            _contact_net.eval()
        else:
            print(f"WARNING: UFold weights not found at {WEIGHTS_PATH}")
    return _contact_net

def seq_to_input(seq: str, max_len: int = 128):
    seq = seq.upper().replace("U", "T")
    L = min(len(seq), max_len)
    seq = seq[:L]
    
    base_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
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
    return torch.tensor(padded).unsqueeze(0).to(DEVICE)

def get_contact_map(seq: str, max_len: int = 128) -> np.ndarray:
    model = _get_model()
    inp = seq_to_input(seq, max_len)
    with torch.no_grad():
        pred = model(inp)
        score_map = torch.sigmoid(pred)
    return score_map.squeeze().cpu().numpy()

def build_4channel_tensor(seq_h: str, seq_m: str, max_len: int = 128):
    c1 = get_contact_map(seq_h, max_len)
    c2 = get_contact_map(seq_m, max_len)
    c3 = np.abs(c1 - c2)
    
    # Mask C4 (canonical pairs mask for healthy seq)
    inp_h = seq_to_input(seq_h, max_len)
    c4 = inp_h[0, 16].cpu().numpy()
    
    # Stack [C, H, W] for torch models
    tensor = np.stack([c1, c2, c3, c4], axis=0).astype(np.float32)
    return torch.tensor(tensor).unsqueeze(0) # [1, 4, 128, 128]
