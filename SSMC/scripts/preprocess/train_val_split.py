import random
import shutil
from pathlib import Path
import yaml
'''
train-val can be run once for every iteration of softsrv generation. Each time, I will add the new data 
to the root folder, after reviewing by hand, then call this script again.
'''
def load_config():
    config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_memos(dir):
    '''
    Retrieves all the memos from the root folder in SoftSRV/data/
    '''
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    return [f.name for f in dir.iterdir() if f.is_file() and f.suffix == ".txt"]

def split():
    '''
    Train-Validation Split. Starting with 30 memos in root split into 4 sections, 
    splits these memos into 6 validation and 24 training memos per section.
    '''
    config = load_config()
    holdout = config['split']['validation_num']
    seed = config['split']['seed']
    random.seed(seed)

    softsrv = Path(__file__).resolve().parent.parent.parent
    root_dirs = {section: softsrv / Path(path) for section, path in config['data']['root'].items()}
    train_dirs = {section: softsrv / Path(path) for section, path in config['data']['train'].items()}
    val_dirs = {section: softsrv / Path(path) for section, path in config['data']['validation'].items()}

    for section, path in root_dirs.items():
        train_path = train_dirs[section]
        val_path = val_dirs[section]

        path.mkdir(parents=True, exist_ok=True)
        train_path.mkdir(parents=True, exist_ok=True)
        val_path.mkdir(parents=True, exist_ok=True)

        files = get_memos(path)
        if not files:
            print(f"No .txt files found in {path}")
            continue

        random.shuffle(files)
        val_set, train_set = files[:holdout], files[holdout:]

        for file in val_set:
            shutil.copy(str(path / file), str(val_path / file))
        for file in train_set:
            shutil.copy(str(path / file), str(train_path / file))

    print("Split Complete.")

if __name__ == "__main__":
    split()
