"""
There are images without annotations, we removed them from training and evaulation:

See: https://github.com/cocodataset/cocoapi/issues/76
train_ids missing: 1021 images
val_ids missing: 48 images
"""
import os
import json

import numpy as np
import torchvision

from pycocotools.coco import COCO
from detectron2.data.datasets import register_coco_instances

from detr.datasets.coco import ConvertCocoPolysToMask

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
CLASSES_CLEANED = list(filter(lambda x: 'N/A' not in x, CLASSES))
CLASSES_DICT = dict(zip(CLASSES, range(len(CLASSES))))


class CocoDetectionSubset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, category_names=None, output_dir=None):
        """A ususal dataset with additional subset options. If `categroy_names`
            and `output_dir` is given, we create a new coco file and coco api object
            which is then used instead of the original one. This way, we make coco
            contiguous s.t. the first class (i.e., person) has label 0 and the last
            class (i.e., toothbrush) has label 79.

        Args:
            img_folder (str]): Folder where images are located.
            ann_file (str): The corresponding annotation file in json format.
            transforms (transform): Transforms which are to be performed.
                Note, Faster-RCNN performs own transformations in the forward pass.
            return_masks (bool): Whether to return masks, for segmentation.
            category_names (list, optional): Categories to consider for subset.
                First element gets mapped to label 0 so background is not encoded here. Defaults to None.
            output_dir (str, optional): The output directory to save the new coco json file. Defaults to None.
        """
        super(CocoDetectionSubset, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.category_names = category_names
        self.output_dir = output_dir
        self.ds_name = os.path.basename(ann_file)

        # Change ids to only consider subset
        if self.category_names:
            assert output_dir, 'There should be an output directory for subsets so a coco json can be generated.'
            if 'all' in self.category_names:
                print("Creating 80 class coco with the following classes: {}".format(', '.join(CLASSES_CLEANED)))
                self.category_names = CLASSES_CLEANED
            # We want to map cat_name[0] to cat_id 0, but getCatIds sorts the returned ids.
            # This means that [giraffe, zebra] would be mapped to [1, 0] because zebra = 24 and giraffe = 25.
            # self.cat_ids = self.coco.getCatIds(catNms=category_names)
            self.cat_ids = [self.coco.getCatIds(catNms=[cat])[0] for cat in self.category_names]
            self.ids = [self.coco.getImgIds(catIds=id) for id in self.cat_ids]
            self.ids = [id for id_list in self.ids for id in id_list]
            self.ids = list(set(self.ids))  # remove duplicate img ids

            # Create new coco GT for evaluation and loading purposes
            print('Creating coco subset with {} categories'.format(len(set(self.cat_ids))))
            print('Note: Images with no annotations are removed.')
            self.path = "subset_{}".format(self.ds_name)
            coco_file = os.path.join(self.output_dir, self.path)
            self._create_subset_coco_format(coco_file)
            self.coco = COCO(coco_file)
        self.num_classes = max([cat for cat in self.coco.cats])+1

    def __getitem__(self, idx):
        img, target = super(CocoDetectionSubset, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def _create_subset_coco_format(self, path):
        """Generate a subset coco file where the classes are contiguous.
            For example, the cat_ids [25, 30] will be mapped to [0, 1])
            We do not assume a background class at 0.

        Args:
            path (str): Path to save the subset json file.
        """
        # mapping for contiguous classes
        mapping = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        # Get info
        info = self.coco.dataset['info']
        info['description'] = 'Subset of COCO with categories: {}'.format(','.join(self.category_names))

        # Get images
        idx = np.isin([img['id'] for img in self.coco.dataset['images']], self.ids)
        images = np.array(self.coco.dataset['images'])[idx].tolist()
        # images2 = [img for img in deepcopy(self.coco.dataset['images']) if img['id'] in self.ids]

        # Get annotations with cat_ids
        annotation_ids = self.coco.getAnnIds(imgIds=self.ids, catIds=self.cat_ids)
        annotations = self.coco.loadAnns(annotation_ids)
        for anno in annotations:
            anno['category_id'] = mapping[anno['category_id']]

        # Categories:
        categories = self.coco.dataset['categories']
        categories = [cat for cat in categories if cat['id'] in self.cat_ids]
        for cat in categories:
            cat['id'] = mapping[cat['id']]

        # Licences
        licences = self.coco.dataset['licenses']

        assert len(images) != 0
        assert len(annotations) != 0
        assert len(categories) != 0

        coco_file = {
            'info': info,
            'images': images,
            'annotations': annotations,
            'licenses': licences,
            'categories': categories,
        }
        print(f"Writing json file to {path}.")
        with open(path, 'w') as f:
            f.write(json.dumps(coco_file))
            f.flush()


def create_subset_json(ann_file, category_names, output_dir):
    ds = CocoDetectionSubset(
        img_folder=None,
        ann_file=ann_file,
        transforms=None,
        return_masks=False,
        category_names=category_names,
        output_dir=output_dir
    )
    return os.path.join(output_dir, ds.path)