# %% 
import torch
import numpy as np
import pylab as plt
from PIL import Image
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from torchvision.ops.boxes import box_iou
from detectron2.utils.visualizer import Visualizer

# %%
ds_path_coco = "/mnt/media/datasets/coco/val2017/"
ann_path_coco = "/mnt/work/dhuseljic/detectron2/outputs/8207/animals/subset_instances_val2017.json"
ds_path_open_images = "/mnt/datasets/open-images/train"
ann_path_open_images = "/home/denis/Documents/projects/2021_uncertainty-in-object-detection/output/open-images_animals-subset_coco-format.json"

register_coco_instances('coco_animals', {}, ann_path_coco, ds_path_coco)
register_coco_instances('open_images_animals', {}, ann_path_open_images, ds_path_open_images)
# %%

dataset_coco = DatasetCatalog.get('coco_animals')
dataset_oi = DatasetCatalog.get('open_images_animals')
# %%

# unpack annotations of coco
def unpack_annotations(dataset):
    annotations = []
    for ann in [d['annotations'] for d in dataset]:
        annotations.extend(ann)
    return annotations

# %%
# Get areas of bounding boxes
annotations_coco = unpack_annotations(dataset_coco)
areas_coco = [np.prod(anno['bbox'][-2:]) for anno in annotations_coco]
areas_coco /= np.max(areas_coco)

annotations_open_images = unpack_annotations(dataset_oi)
areas_open_images = [np.prod(anno['bbox'][-2:]) for anno in annotations_open_images]
areas_open_images /= np.max(areas_open_images)

print('Normalized Mean size of boxes in coco:', np.mean(areas_coco))
print('Normalized Mean size of boxes in open-images:', np.mean(areas_open_images))

plt.hist([areas_coco, areas_open_images], label=['coco', 'open_images'], density=True)
plt.legend()

# %%
diff = np.mean(areas_open_images) - np.mean(areas_coco)
print(f'Boxes in shifted are {diff/np.mean(areas_coco):.2f} bigger than in original')

# %%
# boxes per images
n_boxes_per_image_coco = [len(d['annotations']) for d in dataset_coco]
n_boxes_per_image_oi = [len(d['annotations']) for d in dataset_oi]

print(f'Mean number of boxes per images in coco: {np.mean(n_boxes_per_image_coco)}')
print(f'Mean number of boxes per images in open-images: {np.mean(n_boxes_per_image_oi)}')

a, b = np.unique(n_boxes_per_image_coco, return_counts=True)
plt.bar(a, b, width=.25, edgecolor='black', label='coco')
a, b = np.unique(n_boxes_per_image_oi, return_counts=True)
plt.bar(a+.25, b, width=.25, edgecolor='black', label='open-images')
plt.xticks(np.arange(17)+.25/2, np.arange(17))
plt.legend()
plt.show()

# %%

# %%

# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
# Occlusion per image
def get_mean_iou_per_image(dataset, plot_high_ious=False):
    # returns the sum of all ious that are greater than zero per image
    mean_ious = []
    for d in dataset:
        annos = d['annotations']
        if len(annos) <= 1:
            continue
        boxes = torch.Tensor([anno['bbox'] for anno in annos]).float()
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        iou = box_iou(boxes, boxes)
        iou = torch.tril(iou, diagonal=-1)
        ious = iou[iou > 0]
        iou_per_image = ious.sum().item()
        mean_ious.append(iou_per_image)

        if plot_high_ious and iou_per_image > 1:
            img = Image.open(d['file_name'])
            v = Visualizer(img)
            o = v.draw_dataset_dict(d)
            plt.figure()
            plt.imshow(o.get_image())
            plt.axis('off')
            plt.show()
        
    return mean_ious

iou_per_image_coco = get_mean_iou_per_image(dataset_coco)
iou_per_image_oi = get_mean_iou_per_image(dataset_oi)
plt.hist([iou_per_image_coco, iou_per_image_oi], label=['coco', 'open-images'])
plt.legend()
plt.show()

# %%
iou_per_image_coco = get_mean_iou_per_image(dataset_coco, True)
# iou_per_image_oi = get_mean_iou_per_image(dataset_oi, True)
# %%
