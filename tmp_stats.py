import pandas as pd

df = pd.read_csv('DeepFold_Dataset/final_dataset.csv')
print('COLUMNS:', df.columns.tolist())
print('ROWS:', len(df), '| COLS:', len(df.columns))
print('LABEL DIST:', df['label'].value_counts().to_dict())
print('UNIQUE miRNA IDs:', df['mirna_id'].nunique())
df['seq_len'] = df['Seq_Healthy'].str.len()
print('SEQ LEN min/max/mean:', df['seq_len'].min(), df['seq_len'].max(), round(df['seq_len'].mean(),1))
print('CHROMOSOMES:', df['chr'].nunique())
print('TOP miRNAs:', df['mirna_id'].value_counts().head(8).to_dict())
if 'source' in df.columns:
    print('SOURCE:', df['source'].value_counts().to_dict())
if 'region' in df.columns:
    print('REGION:', df['region'].value_counts().to_dict())
if 'class' in df.columns:
    print('CLASS:', df['class'].value_counts().to_dict())
