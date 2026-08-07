import os

code = open('d:/Capstone/train_all_extracted.py', encoding='utf-8').read()
idx = code.find('class SEBlock(nn.Module):')

imports = [l for l in code[:idx].split('\n') if l.startswith('import ') or l.startswith('from ')]
imports = [i for i in imports if 'Bio' not in i]

header = '\n'.join(imports)
data_load = '''
import os
import pandas as pd
if not os.path.exists("DeepFold_models"):
    os.makedirs("DeepFold_models")
df = pd.read_csv("DeepFold_Dataset/final_dataset.csv")
df = df[df["Sample_ID"].apply(lambda x: os.path.exists(f"DeepFold_Dataset/processed_maps/npy/{x}.npy"))]
print(f"Using {len(df)} samples with valid NPY files")

# Fallback definition for true_y and predictions arrays
true_y = df["class"].values
cnn_all_probs = np.zeros(len(df))
gcn_all_probs = np.zeros(len(df))
kmer_all_probs = np.zeros(len(df))
mfe_all_probs = np.zeros(len(df))
rnafm_all_probs = np.zeros(len(df))

'''

body = code[idx:]
body = body.replace('/kaggle/working/', 'DeepFold_models/') # So checkpoints are dumped directly there
body = body.replace('DeepFold_Dataset/processed_maps/npy', 'DeepFold_Dataset/processed_maps/npy')

with open('d:/Capstone/run_training_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(header + '\n' + data_load + '\n' + body)

print("Saved run_training_pipeline.py")
