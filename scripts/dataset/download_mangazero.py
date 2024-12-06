import argparse
import os
import json
from PIL import Image
import requests
from io import BytesIO
import time


def main(args):
    with open(args.ann_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    num_anns = len(annotations)
    
    for ann_idx, ann in enumerate(annotations):
        image_path = ann["image_path"]
        # get support character ids
        meta = ann["meta"]
        url1 = meta['url1']
        url2 = meta['url2']
        # width1 = meta['width1']
        # width2 = meta['width2']
        try:
            response = requests.get(url1, timeout=30)
            img1 = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            print(f"Error downloading image from {url1}: {e}.\nSkip.")
            num_error_download_images += 1
            manga_num_error_download_images += 1
            img1 = None
        time.sleep(0.2)

        try:
            response = requests.get(url2, timeout=30)
            img2 = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            print(f"Error downloading image from {url2}: {e}.\nSkip.")
            num_error_download_images += 1
            manga_num_error_download_images += 1
            img2 = None
        time.sleep(0.2)

        if img1 != None and img2 != None:
            save_path = os.path.join(args.output_image_root, image_path)
            total_width = img1.width + img2.width
            max_height = max(img1.height, img2.height)

            new_img = Image.new('RGB', (total_width, max_height))
            new_img.paste(img1, (0, 0))
            new_img.paste(img2, (img1.width, 0))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            new_img.save(save_path, 'JPEG')
            print(f"images for {image_path} downloaded, {ann_idx}/{num_anns}")
        else:
            print(f"Error downloading {image_path}, skip. {ann_idx}/{num_anns}")
    
    print("The End")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_path', type=str, required=True)
    parser.add_argument('--output_image_root', type=str, required=True)
    args = parser.parse_args()

    main(args)