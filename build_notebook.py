import json
import glob
import os

print("Loading original final.ipynb...")
with open('final.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

files = ['cells_group_A.py', 'cells_group_B.py', 'cells_group_C.py', 'cells_group_D.py', 'cells_group_E.py']

for filepath in files:
    print(f"Processing {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by cell header
    parts = content.split('# ═══ CELL ')
    
    for part in parts[1:]: # Skip the first header comment
        lines = part.split('\n')
        header_line = lines[0]
        cell_id = header_line.split()[0] # e.g. A0, A1, B1
        
        # Determine content
        cell_content = '\n'.join(lines[1:]).strip('\n')
        is_markdown = 'Markdown' in header_line
        
        if is_markdown:
            # Strip the leading `# ` from lines in markdown cells
            cell_lines = []
            for line in cell_content.split('\n'):
                if line.startswith('# '):
                    cell_lines.append(line[2:])
                elif line.startswith('#'):
                    cell_lines.append(line[1:])
                else:
                    cell_lines.append(line)
            
            source = [line + '\n' for line in cell_lines]
            if source:
                source[-1] = source[-1].rstrip('\n') # Jupyter doesn't put newline on last line
                
            cell = {
                "cell_type": "markdown",
                "id": f"gen_md_{cell_id}",
                "metadata": {},
                "source": source
            }
            nb['cells'].append(cell)
        else:
            # Code cell
            source = [line + '\n' for line in cell_content.split('\n')]
            if source:
                source[-1] = source[-1].rstrip('\n')
                
            cell = {
                "cell_type": "code",
                "execution_count": None,
                "id": f"gen_code_{cell_id}",
                "metadata": {},
                "outputs": [],
                "source": source
            }
            nb['cells'].append(cell)

out_file = 'final_v4.ipynb'
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Success! Appended new cells and created {out_file}")
