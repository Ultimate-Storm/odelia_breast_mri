from pathlib import Path
import shutil 
from tqdm import tqdm


path_root = Path('/home/homesOnMaster/gfranzes/Documents/datasets/ODELIA')
path_root_institution = path_root/'DUKE'
path_root_data = path_root_institution/'data'


for p in tqdm(path_root_data.rglob('*.nii.gz')):
    name = p.name

    if name and name[0].islower():
        new_name = name[0].upper() + name[1:]
        target = p.with_name(new_name)
        if not target.exists():
            try:
                p.rename(target)
            except Exception:
                pass
