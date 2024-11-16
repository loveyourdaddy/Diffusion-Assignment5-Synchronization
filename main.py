import sys 
import os 
import json
from tqdm import tqdm
import numpy as np 
from PIL import Image 

import torch
import argparse


def seed_everything(seed=2024):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def main():
    seed_everything()
    app = sys.argv[sys.argv.index("--app")+1]
    
    if app == "ambiguous_image":
        from guidance.ambiguous_image_model import AmbiguousImageModel
        from configs.ambiguous_image_config import load_ambiguious_image_config

        config = load_ambiguious_image_config()
        model = AmbiguousImageModel(config)
        
    elif app == "wide_image":
        from guidance.wide_image_model import WideImageModel
        from configs.wide_image_config import load_wide_image_config

        config = load_wide_image_config()
        model = WideImageModel(config)
        
    else:
        raise NotImplementedError(f"Invalid application: {app}")
    
    model()
    

if __name__ == "__main__":
    main()
    # python main.py --app wide_image --prompt "A_sweeping_panoramic_view_of_the_Grand_Canyon_at_sunset" --tag wide_image
    # Python eval.py –-fdir1 {$FDIR1} –app wide_images

    # python main.py --app ambiguous_image --prompts "an oil painting of a snowy mountain village" "an oil painting of a horse" --tag ambiguous_image --save_dir_now
    # Python eval.py –-fdir1 {$FDIR1} –app ambiguous_images