# Adapted from https://github.com/asharakeh/probdet/blob/master/src/core/datasets/convert_openimages_odd_to_coco.py
import os
import csv
import json

import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from datasets.meta import coco_categories

def create_openimages_json(ann_dir: str,
                           image_dir: str,
                           output_dir: str,
                           category_names: list,
                           subset_name: str = None,
                           n_images: int = None,
                           remove_coco_classes: bool = False,
                           ):
    os.makedirs(output_dir, exist_ok=True)
    cat_mapping = {cat_name: cat_id+1 for cat_id, cat_name in enumerate(category_names)}

    class_desc_file = os.path.join(ann_dir, 'class-descriptions-boxable.csv')
    hierarchy_file = os.path.join(ann_dir, 'bbox_labels_600_hierarchy.json')
    box_annotations_file = os.path.join(ann_dir, 'oidv6-train-annotations-bbox.csv')
    if subset_name:
        subset_name += '-'
    output_json_file = os.path.join(output_dir, f'open-images_{subset_name}subset_coco-format.json')

    # Get category mapping from openimages symbol to openimages names.
    openimages_class_mapping_dict = dict()
    with open(class_desc_file, 'r', encoding='utf-8') as f:
        csv_f = csv.reader(f)
        for row in csv_f:
            openimages_class_mapping_dict.update({row[0]: row[1].lower()})

    with open(hierarchy_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        hierarchy_list = data['Subcategory']

    print('Loading Annotations into memory.')
    df_annos = pd.read_csv(
        box_annotations_file,
        usecols=['ImageID', 'LabelName', 'XMin', 'YMin', 'XMax', 'YMax'],
        nrows=500000,
    )
    df_annos['CategoryName'] = [openimages_class_mapping_dict[lbl_name] for lbl_name in df_annos['LabelName']]
    print('Done.')

    if remove_coco_classes:
        mask = np.isin(df_annos['CategoryName'], [d['name'] for d in coco_categories])
        df_annos = df_annos[~mask]

    print('Selecting sutiable images.')
    assert sum(np.isin(category_names, np.unique(df_annos['CategoryName'].values))) == len(category_names)
    mask = np.isin(df_annos['CategoryName'].values, category_names)
    df_annos_subset = df_annos[mask]
    image_ids = df_annos_subset.ImageID.unique()
    print(f'Found {len(image_ids)} images with {mask.sum()} annotations for the categories `{", ".join(category_names)}`.')
    info = {c: n for c, n in zip(*np.unique(df_annos_subset['CategoryName'].values, return_counts=True))}
    print(info)

    print('Filtering images to match coco subsets.')
    df_annos_subset = df_annos_subset[np.isin(df_annos_subset.ImageID, image_ids[:n_images])]
    image_ids = df_annos_subset.ImageID.unique()
    print(f'Found {len(image_ids)} images with {mask.sum()} annotations for the categories `{", ".join(category_names)}`.')
    info = {c: n for c, n in zip(*np.unique(df_annos_subset['CategoryName'].values, return_counts=True))}
    print(info)

    print('Creating coco format json.')
    images = []
    annotations = []
    for image_id in tqdm(image_ids):
        df_img_id = df_annos_subset[df_annos_subset['ImageID'] == image_id]
        file_name = image_id + '.jpg'

        image = Image.open(os.path.join(image_dir, file_name))
        # import pylab as plt
        # plt.imshow(image)
        # plt.show()

        width, height = image.size
        images.append({
            'id': image_id,
            'width': width,
            'height': height,
            'file_name': file_name,
            'license': 1
        })

        for i_anno, anno_row in df_img_id.iterrows():
            category_name = anno_row['CategoryName']
            category_id = cat_mapping[category_name]

            xmin, ymin, xmax, ymax = anno_row[['XMin', 'YMin', 'XMax', 'YMax']]
            xmin, ymin, xmax, ymax = xmin*width, ymin*height, xmax*width, ymax*height
            bbox = [xmin, ymin, xmax-xmin, ymax-ymin]
            area = (xmax - xmin) * (ymax - ymin)
            annotations.append({
                "id": len(annotations),
                "image_id": image_id,
                "category_id": category_id,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            })
    print('Done.')

    print(f"Writing json file to {output_json_file}.")
    categories = [
        {'id': cat_id, 'name': cat_name, 'supercategory': cat_name} for cat_name, cat_id in cat_mapping.items()
    ]
    coco_json = {
        'info': {'year': 2021},
        'images': images,
        'annotations': annotations,
        'licenses': [{'id': 1, 'name': 'none', 'url': 'none'}],
        'categories': categories,
    }
    with open(output_json_file, 'w') as f:
        f.write(json.dumps(coco_json))
        f.flush()
    print('Done.')
    return output_json_file


def search_hierarchy(word, hierarchy_list, mapping_dict):
    """really hacky hierarchy searching"""
    tmp = []
    word_found = False
    # if not isinstance(hierarchy_list, list):
    #     return []
    for cat in hierarchy_list:
        cat_name = mapping_dict[cat['LabelName']]
        all_subcats = get_subcategories(cat, mapping_dict)
        # if word is in there, than add also all subcat names to it
        if word == cat_name:
            word_found = True
            tmp.append(word)
            # add rest to it
            tmp.extend(all_subcats)
        elif word in all_subcats:
            tmp.extend(search_hierarchy(word, cat['Subcategory'], mapping_dict))
    return tmp


def get_subcategories(cat, mapping_dict):
    tmp = []
    for p in ['Subcategory', 'Part']:
        if p in cat.keys():
            for subcat in cat[p]:
                subcat_name = mapping_dict[subcat['LabelName']]
                tmp.append(subcat_name)
                tmp.extend(get_subcategories(subcat, mapping_dict))
    return tmp


if __name__ == "__main__":
    path = '/mnt/datasets/open-images/'
    ann_file = create_openimages_json(
        ann_dir=os.path.join(path, 'annotations'),
        image_dir=os.path.join(path, 'train'),
        output_dir='./output/open-images/',
        category_names=['giraffe', 'elephant']
    )
