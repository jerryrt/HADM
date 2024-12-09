import os
import json
from tqdm import tqdm
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str)
args = parser.parse_args()

def generate_empty_anno(data_root):
    anno_root = data_root.replace("/images/", "/annotations/")

    if not os.path.exists(anno_root):
        os.makedirs(anno_root)

    print("Generating empty annotation for {}".format(data_root))

    for img in tqdm(os.listdir(data_root)):
        ext = os.path.splitext(img)[1]
        img_name = img.replace(ext, "")
        img_path = os.path.join(data_root, img)
        im = Image.open(img_path)
        width, height = im.size
        anno = {
            "image": {
                "file_name": img,
                "height": height,
                "width": width,
                "tag": "empty annotation"
            },
            "annotation": [],
            "human": []
        }
        with open(os.path.join(anno_root, img_name + '.json'), 'w') as f:
            json.dump(anno, f)

if __name__ == "__main__":
    generate_empty_anno(args.data_root)

# python datasets/generate_empty_anno.py --data_root datasets/human_artifact_dataset/images/<data_folder>