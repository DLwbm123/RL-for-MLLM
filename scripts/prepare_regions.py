import os
from pathlib import Path
from src.regions import prepare_regions
if __name__ == '__main__':
    print(prepare_regions(os.environ['DATA_ROOT'],os.environ['OUTPUT_ROOT']))
