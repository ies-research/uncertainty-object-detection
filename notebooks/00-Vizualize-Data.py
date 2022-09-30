# %%
import numpy as np
import pylab as plt

from PIL import Image

from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog
from detectron2.utils.visualizer import Visualizer

# %% 
ds_path = "/mnt/datasets/open-images/train"
ann_path_animals = '../../output/open-images_animals-subset_coco-format.json'
ann_path_traffic = '../../output/open-images_traffic-subset_coco-format.json'
ann_path_ood = '../../output/open-images_ood-subset_coco-format.json'

register_coco_instances('animals', {}, ann_path_animals, ds_path)
register_coco_instances('traffic', {}, ann_path_traffic, ds_path)
register_coco_instances('ood', {}, ann_path_ood, ds_path)

# %%

def plot_random_images(dataset, seed=42):
    rows, cols = 2, 4
    np.random.seed(seed)
    rnd_indices = np.random.choice(len(dataset), rows*cols, replace=False)
    print(rnd_indices)

    for i, idx in enumerate(rnd_indices):
        plt.subplot(rows, cols, i+1)
        data_dict = dataset[idx]
        img = Image.open(data_dict['file_name'])
        vis = Visualizer(img_rgb=img, scale=1)
        # o = vis.draw_dataset_dict(data_dict)
        vis._default_font_size = 32
        for anno in data_dict['annotations']:
            x, y, w, h = anno['bbox']
            o = vis.draw_box([x, y, x+w, y+h], alpha=1, edge_color='red')
        plt.imshow(o.get_image())
        plt.axis('off')
        plt.tight_layout()

plt.figure(figsize=(5, 2))
plot_random_images(DatasetCatalog.get('animals'))
plt.savefig('plots/animals_shifted.pdf')
plt.show()

plt.figure(figsize=(5, 2))
plot_random_images(DatasetCatalog.get('traffic'))
plt.savefig('plots/traffic_shifted.pdf')
plt.show()

plt.figure(figsize=(5, 2))
plot_random_images(DatasetCatalog.get('ood'))
plt.savefig('plots/ood.pdf')
plt.show()
# %%
