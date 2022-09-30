import os
import argparse

from datasets.open_images_subset import create_openimages_json


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    # n_samples = 190  # (animals coco)
    # n_samples = 3110  # (traffic coco)
    # n_samples = 5000 # (all coco)

    subset_dict = {
        'animals': ['giraffe', 'elephant'],
        'traffic': ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'traffic light', 'fire hydrant',
                    'stop sign'],
        # 'all': [d['name'] for d in coco_categories],
        'ood': ['camel', 'hippopotamus', 'rhinoceros', 'goat', 'hamster', 'kangaroo', 'koala', 'squirrel', 'monkey', 'bull'],
    }
    n_images_dict = {'animals': 190, 'traffic': 3110, 'ood': 5000}

    category_names = subset_dict[args.subset]
    ann_file = create_openimages_json(
        args.ann_path,
        args.ds_path,
        args.output_dir,
        category_names=category_names,
        subset_name=args.subset,
        n_images=n_images_dict[args.subset],
        remove_coco_classes=(args.subset == 'ood'),
    )
    print(f'Json in coco format saved to {ann_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_path')
    parser.add_argument('--ann_path')
    parser.add_argument('--output_dir')
    parser.add_argument('--subset', choices=['animals', 'traffic', 'all', 'ood'])
    args = parser.parse_args()
    main(args)
