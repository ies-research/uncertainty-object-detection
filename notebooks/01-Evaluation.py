# %%
# fmt: off
import glob
import sys
import os
sys.path.append('..')

import torch
import matplotlib
import numpy as np
import pylab as plt
import pandas as pd

from PIL import Image
from pathlib import Path
from torchvision.transforms import Resize
from detectron2.utils.visualizer import Visualizer

from evaluate import get_config
from datasets.meta import subset_categories

# fmt: on 
plot_path = Path('/home/denis/Documents/projects/2021_uncertainty-in-object-detection/src/notebooks/plots')
start_path_on_server = '/mnt/home/dhuseljic/projects/uncertainty-in-object-detection/src/'


def load_results(path: str, return_detections=False):
    result_dir = Path(path)
    results = torch.load(result_dir / 'results_final.pth')
    # coco_results = results['coco_results']
    # cal_results = results['calibration_results']

    args = results['args']
    args.config_file = result_dir / 'config.yaml'
    if args.eval_from_config:
        rel_path = os.path.relpath(args.eval_from_config, start_path_on_server)
        args.eval_from_config = '..' / Path(rel_path)
    cfg = get_config(args)

    if return_detections:
        detections = torch.load(result_dir / 'detections.pt')
        return results, cfg, detections
    return results, cfg


def plot_detection(d: dict,
                   ds: str = 'coco',
                   subset='animals',
                   plot_all_class_preds=False,
                   size=(512, 512),
                   offset_x=0,
                   offset_y=-31):
    if ds == 'coco':
        ds_path = '/mnt/datasets/COCO/val2017'
        fname = '{}.jpg'.format(str(d['image_id']).zfill(12))
    elif ds == 'open-images':
        ds_path = '/mnt/datasets/open-images/train'
        fname = f'{d["image_id"]}.jpg'

    if subset == 'animals':
        categories = subset_categories['animals'] + ['background']
    elif subset == 'traffic':
        categories = subset_categories['traffic'] + ['background']
    elif subset == 'all':
        categories = subset_categories['all'] + ['background']

    resize = Resize(size)
    ds_path = Path(ds_path)
    img = Image.open(ds_path / fname)
    img = resize(img)
    v = Visualizer(img)
    if 'gt_box' in d.keys():
        gt_box = d['gt_box'] * torch.tensor([*img.size, *img.size])
        v._default_font_size = 20
        o = v.draw_box(gt_box, alpha=1, line_style='--', edge_color='lightgreen')
        # o = v.draw_text(d['label'].item(), gt_box[:2] + torch.tensor([10, 0]), font_size=24, color='r')
    if 'pred_box' in d.keys():
        pred_box = d['pred_box'] * torch.tensor([*img.size, *img.size])
        v._default_font_size = 20
        o = v.draw_box(pred_box, edge_color='r', alpha=.5)
        if plot_all_class_preds:
            for p_c, c in zip(d['pred_probas'], categories):
                s = f'{int(p_c*100)}\% {c}'
                o = v.draw_text(s, pred_box[:2] + torch.Tensor([offset_x, offset_y]),
                                font_size=24, color='r', horizontal_alignment='left')
                offset_y += 30
        else:
            proba, pred = d['pred_probas'].max(-1)
            s = f'{int(proba*100)}\% {categories[pred]}'
            o = v.draw_text(s, pred_box[:2] + torch.Tensor([offset_x, offset_y]),
                            font_size=30, color='r', horizontal_alignment='left')
    return o.get_image()


def plot_detections_img(dets: list, ds: str = 'coco', subset='animals', size=(512, 512)):
    img_id = dets[0]['image_id']
    if ds == 'coco':
        ds_path = '/mnt/datasets/COCO/val2017'
        fname = '{}.jpg'.format(str(img_id).zfill(12))
    elif ds == 'open-images':
        ds_path = '/mnt/datasets/open-images/train'
        fname = f'{img_id}.jpg'

    if subset == 'animals':
        categories = subset_categories['animals'] + ['background']
    elif subset == 'traffic':
        categories = subset_categories['traffic'] + ['background']
    elif subset == 'ood':
        categories = subset_categories['ood'] + ['background']
    elif subset == 'all':
        categories = range(80)

    resize = Resize(size)
    img = Image.open(Path(ds_path) / fname)
    img = resize(img)
    v = Visualizer(img)
    for d in dets:
        if 'gt_box' in d.keys():
            gt_box = d['gt_box'] * torch.tensor([*img.size, *img.size])
            o = v.draw_box(gt_box, alpha=1, line_style=':', edge_color='k')
            # o = v.draw_text(d['label'].item(), gt_box[:2] + torch.tensor([10, 0]), font_size=24, color='r')
        if 'pred_box' in d.keys():
            pred_box = d['pred_box'] * torch.tensor([*img.size, *img.size])
            color = 'g' if d['matched'] else 'r'
            o = v.draw_box(pred_box, edge_color=color, alpha=.5)
            proba, pred = d['pred_probas'].max(-1)
            s = f'{int(proba*100)}\% {categories[pred]}'
            o = v.draw_text(s, pred_box[:2] + torch.Tensor([115, -31]), font_size=24, color=color)
    return o.get_image()


matplotlib.rc('font', **{'family': 'normal', 'weight': 'bold', 'size': 23})

# %%
# Define paths to experiments
paths = {
    # Animals (coco)
    'DETR_animals': '/mnt/work/dhuseljic/uod_results/detr_animals/',
    'FRCNN_animals_application': '/mnt/work/dhuseljic/uod_results/frcnn_animals_application/',
    'FRCNN_animals_modeling': '/mnt/work/dhuseljic/uod_results/frcnn_animals_modeling/',
    # Animals (open-images)
    'DETR_animals_shifted': '/mnt/work/dhuseljic/uod_results/detr_animals_shifted/',
    'FRCNN_animals_shifted_application': '/mnt/work/dhuseljic/uod_results/frcnn_animals_shifted_application/',
    'FRCNN_animals_shifted_modeling': '/mnt/work/dhuseljic/uod_results/frcnn_animals_shifted_modeling/',
    # Traffic (coco)
    'DETR_traffic': '/mnt/work/dhuseljic/uod_results/detr_traffic/',
    'FRCNN_traffic_application': '/mnt/work/dhuseljic/uod_results/frcnn_traffic_application/',
    'FRCNN_traffic_modeling': '/mnt/work/dhuseljic/uod_results/frcnn_traffic_modeling/',
    # Traffic (open-images)
    'DETR_traffic_shifted': '/mnt/work/dhuseljic/uod_results/detr_traffic_shifted',
    'FRCNN_traffic_shifted_application': '/mnt/work/dhuseljic/uod_results/frcnn_traffic_shifted_application/',
    'FRCNN_traffic_shifted_modeling': '/mnt/work/dhuseljic/uod_results/frcnn_traffic_shifted_modeling/',
    # All (coco)
    'DETR_all': '/mnt/work/dhuseljic/uod_results/detr_all/',
    'FRCNN_all_application': '/mnt/work/dhuseljic/uod_results/frcnn_all_application/',
    'FRCNN_all_modeling': '/mnt/work/dhuseljic/uod_results/frcnn_all_modeling/',
    # OOD
    'DETR_ood': '/mnt/work/dhuseljic/uod_results/detr_ood',
    'FRCNN_ood_application': '/mnt/work/dhuseljic/uod_results/frcnn_ood_application',
    'FRCNN_ood_modeling': '/mnt/work/dhuseljic/uod_results/frcnn_ood_modeling',
}
# 'DETR_ood': '/mnt/work/dhuseljic/uod_results/detr_ood',
load_results(paths['DETR_all'])
pass
# %%
# Print results in latex table
models, scores = [], []
for key in paths:
    results, cfg = load_results(paths[key])
    models.append(key)
    res = {}
    res.update({k: results['coco_results']['bbox'][k]/100 for k in ['AP']})
    res.update({k: results['calibration_results'][k] for k in ['NLL', 'BS', 'TCE', 'MCE', 'dMCE']})
    scores.append(res)
df = pd.DataFrame(scores, index=models)
print(df.to_latex(float_format="%.3f"))

# %%


def plot_TCE(ax_cp, ax_bins, confs, accs, n_samples):
    n_bins = len(confs)
    # Plot calibration plot
    plt.sca(ax_cp)
    plt.plot([0, 1], [0, 1], '--k', lw=3)
    plt.plot(confs, accs, '-o', c='C0')
    plt.xticks(np.linspace(0, 1, 6))
    plt.xlabel('Mean Probability')
    plt.ylabel('Accuracy')
    # Plot bins
    plt.sca(ax_bins)
    plt.bar(np.linspace(0, 1, n_bins), n_samples, width=1/n_bins)
    plt.xticks(np.linspace(0, 1, n_bins), np.arange(1, n_bins+1))  # , map(lambda x: f"{x:.0f}", n_samples), rotation=0)
    plt.xlabel("Bins")
    plt.ylabel("Number of Samples")


def plot_MCE(ax_cp, ax_bins, mean_probas, accs, n_samples, classes, plot_legend=False, bbox_to_anchor=(.5, 1.35), ncol=5):
    mean_probas_mean = np.nanmean(mean_probas, axis=0)
    accs_mean = np.nanmean(accs, axis=0)
    accs_q25 = np.nanquantile(accs, .25, axis=0)
    accs_q75 = np.nanquantile(accs, .75, axis=0)
    n_bins = len(mean_probas_mean)
    n_classes = len(mean_probas)

    plt.sca(ax_cp)
    plt.xticks(np.linspace(0, 1, 6))
    plt.plot([0, 1], [0, 1], '--k', lw=3, label='Perfect Calibration')
    plt.plot(mean_probas_mean, accs_mean, '-o', c='C0', label='(Avg.) Model Calibration')
    plt.fill_between(mean_probas_mean, accs_q25, accs_q75, alpha=.5, color='C0')
    for mean_prob, acc, c in zip(mean_probas, accs, classes):
        if len(classes) == 3:
            label = ['Giraffe', 'Elephant', 'Background'][c]
        else: 
            label = f'Class {c}' 
        plt.plot(mean_prob, acc, c=f'C{c+1}', label=label)
    plt.xlabel('Mean Probability')
    plt.ylabel('Accuracy')
    if plot_legend:
        plt.legend(loc='upper center', bbox_to_anchor=bbox_to_anchor, ncol=ncol)

    plt.sca(ax_bins)

    for i_cls, n_sam in enumerate(n_samples):
        bar_pos = np.linspace(0, 1, n_bins) + (i_cls/n_bins*1/n_classes) - (1/n_bins*1/n_classes)
        plt.bar(bar_pos, n_sam, width=1/n_bins/len(classes), color=f'C{i_cls+1}')
    plt.xticks(np.linspace(0, 1, n_bins), np.arange(1, n_bins+1))  # , map(lambda x: f"{x:.0f}", n_samples), rotation=0)
    plt.xlabel("Bins")


# %%
"""Plotting TCE vs MCE vs dMCE"""


def plot_tce_mce_dmce(path, figsize=(25, 10), plot_legend=True, bbox_to_anchor=(.5, 1.35)):
    matplotlib.rc('lines', linewidth=2)

    results, cfg = load_results(path)

    """TCP"""
    tce_results = results['calibration_results']['TCE_bins'][0]
    confs, accs, n_samples = tce_results['confs'], tce_results['accs'], tce_results['n_samples']
    plt.figure(figsize=figsize)

    ax_cp = plt.subplot(231)
    plt.title('TCP')
    ax_bins = plt.subplot(234)

    plot_TCE(ax_cp, ax_bins, confs, accs, n_samples)

    """MCP"""
    mce_results = results['calibration_results']['MCE_bins']
    mean_probas = np.array([res['mean_probas'] for res in mce_results])
    accs = np.array([res['accs'] for res in mce_results])
    n_samples = np.array([res['n_samples'] for res in mce_results])
    classes = np.array([res['class'] for res in mce_results])
    n_classes = len(classes)

    ax_cp = plt.subplot(232, sharey=ax_cp)
    plt.title('MCP')
    ax_bins = plt.subplot(235)

    plot_MCE(ax_cp, ax_bins, mean_probas, accs, n_samples, classes,
             plot_legend=plot_legend, bbox_to_anchor=bbox_to_anchor)

    # PLot MCE
    mce_results = results['calibration_results']['dMCE_bins']

    mean_probas = np.array([res['mean_probas'] for res in mce_results])
    precisions = np.array([res['precisions'] for res in mce_results])
    n_samples = np.array([res['n_samples'] for res in mce_results])
    classes = np.array([res['class'] for res in mce_results])
    n_classes = len(classes)

    ax_cp = plt.subplot(233, sharey=ax_cp)
    plt.title('dMCP')
    ax_bins = plt.subplot(236)
    plot_MCE(ax_cp, ax_bins, mean_probas, precisions, n_samples, classes)
    ax_cp.set_ylabel('Precision')
    # plt.savefig('./')


# %%
print('DETR')
model, dataset = 'DETR', 'animals'
plot_tce_mce_dmce(paths[f'{model}_{dataset}'])
plt.savefig(plot_path / f'CP_{model}_{dataset}.pdf', bbox_inches='tight')
plt.show()

model, dataset = 'DETR', 'traffic'
plot_tce_mce_dmce(paths[f'{model}_{dataset}'], bbox_to_anchor=(.5, 1.6))
plt.savefig(plot_path / f'CP_{model}_{dataset}.pdf', bbox_inches='tight')
plt.show()

model, dataset = 'DETR', 'all'
plot_tce_mce_dmce(paths[f'{model}_{dataset}'], plot_legend=False)
plt.savefig(plot_path / f'CP_{model}_{dataset}.pdf', bbox_inches='tight')
plt.show()

print('FRCNN Application')
model, perspective, dataset = 'FRCNN', 'application', 'animals'
plot_tce_mce_dmce(paths[f'{model}_{dataset}_{perspective}'])
plt.savefig(plot_path / f'CP_{model}_{perspective}_{dataset}.pdf', bbox_inches='tight')
plt.show()
model, perspective, dataset = 'FRCNN', 'application', 'traffic'
plot_tce_mce_dmce(paths[f'{model}_{dataset}_{perspective}'], bbox_to_anchor=(.5, 1.6))
plt.savefig(plot_path / f'CP_{model}_{perspective}_{dataset}.pdf', bbox_inches='tight')
plt.show()
model, perspective, dataset = 'FRCNN', 'application', 'all'
plot_tce_mce_dmce(paths[f'{model}_{dataset}_{perspective}'], plot_legend=False)
plt.savefig(plot_path / f'CP_{model}_{perspective}_{dataset}.pdf', bbox_inches='tight')
plt.show()

print('FRCNN Modeling')
model, perspective, dataset = 'FRCNN', 'modeling', 'animals'
plot_tce_mce_dmce(paths[f'{model}_{dataset}_{perspective}'])
plt.savefig(plot_path / f'CP_{model}_{perspective}_{dataset}.pdf', bbox_inches='tight')
plt.show()
model, perspective, dataset = 'FRCNN', 'modeling', 'traffic'
plot_tce_mce_dmce(paths[f'{model}_{dataset}_{perspective}'], bbox_to_anchor=(.5, 1.6))
plt.savefig(plot_path / f'CP_{model}_{perspective}_{dataset}.pdf', bbox_inches='tight')
plt.show()
model, perspective, dataset = 'FRCNN', 'modeling', 'all'
plot_tce_mce_dmce(paths[f'{model}_{dataset}_{perspective}'], plot_legend=False)
plt.savefig(plot_path / f'CP_{model}_{perspective}_{dataset}.pdf', bbox_inches='tight')
plt.show()


# %%
"""Plotting TCE MCE dMCE on shifted datasets."""


def plot_mce_shifted(path_detr, path_frcnnA, path_frcnnM, figsize=(25, 10), plot_legend=True, bbox_to_anchor=(.5, 1.35)):
    matplotlib.rc('lines', linewidth=2)

    # DETR
    results, cfg = load_results(path_detr)
    mce_results = results['calibration_results']['MCE_bins']
    mean_probas = np.array([res['mean_probas'] for res in mce_results])
    accs = np.array([res['accs'] for res in mce_results])
    n_samples = np.array([res['n_samples'] for res in mce_results])
    classes = np.array([res['class'] for res in mce_results])

    plt.figure(figsize=figsize)
    ax_cp = plt.subplot(231)
    plt.title('DETR')
    ax_bins = plt.subplot(234)

    plot_MCE(ax_cp, ax_bins, mean_probas, accs, n_samples, classes, plot_legend=False, bbox_to_anchor=bbox_to_anchor)

    # FASTER-RCNN
    results, cfg = load_results(path_frcnnA)
    mce_results = results['calibration_results']['MCE_bins']
    mean_probas = np.array([res['mean_probas'] for res in mce_results])
    accs = np.array([res['accs'] for res in mce_results])
    n_samples = np.array([res['n_samples'] for res in mce_results])
    classes = np.array([res['class'] for res in mce_results])
    n_classes = len(classes)

    ax_cp = plt.subplot(232, sharey=ax_cp)
    plt.title('Faster-RCNN Ap.')
    ax_bins = plt.subplot(235)

    plot_MCE(ax_cp, ax_bins, mean_probas, accs, n_samples, classes,
             plot_legend=plot_legend, bbox_to_anchor=bbox_to_anchor)

    """MCP"""
    results, cfg = load_results(path_frcnnM)
    mce_results = results['calibration_results']['MCE_bins']
    mean_probas = np.array([res['mean_probas'] for res in mce_results])
    accs = np.array([res['accs'] for res in mce_results])
    n_samples = np.array([res['n_samples'] for res in mce_results])
    classes = np.array([res['class'] for res in mce_results])
    n_classes = len(classes)

    ax_cp = plt.subplot(233, sharey=ax_cp)
    plt.title('Faster-RCNN Mo.')
    ax_bins = plt.subplot(236)

    plot_MCE(ax_cp, ax_bins, mean_probas, accs, n_samples, classes, plot_legend=False, bbox_to_anchor=bbox_to_anchor)


plot_mce_shifted(
    path_detr=paths['DETR_animals_shifted'],
    path_frcnnA=paths['FRCNN_animals_shifted_application'],
    path_frcnnM=paths['FRCNN_animals_shifted_modeling'],
)
plt.savefig('./shifted_detr_frcnn.pdf')
plt.show()

# %%
"""Plotting Entropy Histogram"""


def plot_entropy_hists(results_id, results_shifted, results_ood, normalize_entropy=True, log_entropies=False):
    entropies_id = results_id['calibration_results']['entropies']
    entropies_shifted = results_shifted['calibration_results']['entropies']
    entropies_ood = results_ood['calibration_results']['entropies']
    n_classes = 11
    lw = 3
    ls_id, ls_shifted, ls_ood = '-', '--', ':'
    # bins = np.linspace(0, 1, 20)
    bins = 20
    if normalize_entropy:
        entropies_id /= np.log(n_classes)
        entropies_shifted /= np.log(n_classes)
        entropies_ood /= np.log(n_classes)

    if log_entropies:
        entropies_id = np.log(entropies_id)
        entropies_shifted = np.log(entropies_shifted)
        entropies_ood = np.log(entropies_ood)

    plt.hist(entropies_id, histtype='step', bins=bins, lw=lw, ls=ls_id, density=True, label='In-Distribution')
    plt.hist(entropies_shifted, histtype='step', bins=bins, lw=lw, ls=ls_shifted, density=True, label='Shifted')
    plt.hist(entropies_ood, histtype='step', bins=bins, lw=lw, ls=ls_ood, density=True, label='Out-of-Distribution')
    plt.xlabel('Log Entropy' if log_entropies else 'Entropy')
    plt.ylabel('Relative Frequency')
    bbox_to_anchor = (.5, 1.35)
    plt.legend(loc='upper center', bbox_to_anchor=bbox_to_anchor, ncol=4)

# %%


plt.figure(figsize=(20, 5))
plt.sca(plt.subplot(131))
plt.title(r'\textsc{DETR}')
results_id, cfg = load_results(path=paths['DETR_traffic'])
results_shifted, cfg = load_results(path=paths['DETR_traffic_shifted'])
results_ood, cfg = load_results(path=paths['DETR_ood'])
plot_entropy_hists(results_id, results_shifted, results_ood, log_entropies=True)
plt.gca().get_legend().remove()

plt.sca(plt.subplot(132))
plt.title(r'\textsc{Faster-RCNN}')
results_id, cfg = load_results(path=paths['FRCNN_traffic_application'])
results_shifted, cfg = load_results(path=paths['FRCNN_traffic_shifted_application'])
results_ood, cfg = load_results(path=paths['FRCNN_ood_application'])
plot_entropy_hists(results_id, results_shifted, results_ood, log_entropies=True)

plt.sca(plt.subplot(133))
plt.title(r'\textsc{Faster-RCNN w.o. po.}')
results_id, cfg = load_results(path=paths['FRCNN_traffic_modeling'])
results_shifted, cfg = load_results(path=paths['FRCNN_traffic_shifted_modeling'])
results_ood, cfg = load_results(path=paths['FRCNN_ood_modeling'])
plot_entropy_hists(results_id, results_shifted, results_ood, log_entropies=True)
plt.gca().get_legend().remove()
plt.savefig(plot_path / 'entropy_hists.pdf', bbox_inches='tight')
plt.show()

# %%
"""Plotting Detections for ID and OOD"""

results_ood, cfg, detections = load_results(path=paths['DETR_animals_shifted'], return_detections=True)
detections_matched = [d for d in detections if d['matched']]

plt.figure(figsize=(20, 5))
np.random.seed(6)
random_indices = np.random.choice(range(len(detections_matched)), size=4, replace=False)
for i, idx in enumerate(random_indices):
    d = detections_matched[idx]
    plt.subplot(141+i)
    plt.imshow(plot_detection(d, ds='open-images'))
    plt.axis('off')
plt.show()

# %%
"""Recalibration Study by adapting the background weight"""

path = "/mnt/work/dhuseljic/detectron2/outputs/recal_study/evaluation_id"
res_paths = glob.glob(path + '/*')
paths_recal = dict()
for path in res_paths:
    _, cfg = load_results(path)
    paths_recal[cfg.MODEL.DETR.NO_OBJECT_WEIGHT] = path
paths_recal = {k: paths_recal[k] for k in sorted(paths_recal)}

# %%
keys, scores = [], []
for key in paths_recal:
    results, cfg = load_results(paths_recal[key])
    keys.append(key)
    res = {}
    res.update({k: results['coco_results']['bbox'][k]/100 for k in ['AP']})
    res.update({k: results['calibration_results'][k] for k in ['NLL', 'BS', 'TCE', 'MCE', 'dMCE']})
    scores.append(res)
df = pd.DataFrame(scores, index=keys)
df.index.name = 'background_weights'
print(df.to_latex(float_format="%.3f"))

# %%
fontdict = {'family': 'normal', 'weight': 'bold', 'size': 30}

plt.figure(figsize=(22, 10))
plt.subplot(231)
plt.plot(df.index, df.AP)
plt.xlabel('Background weight', fontdict=fontdict)
plt.ylabel('mAP', fontdict=fontdict)
plt.subplot(234)
plt.plot(df.index, df.NLL)
plt.xlabel('Background weight', fontdict=fontdict)
plt.ylabel('NLL', fontdict=fontdict)
plt.subplot(232)
plt.plot(df.index, df.BS)
plt.xlabel('Background weight', fontdict=fontdict)
plt.ylabel('BS', fontdict=fontdict)
plt.subplot(235)
plt.plot(df.index, df.TCE)
plt.xlabel('Background weight', fontdict=fontdict)
plt.ylabel('TCE', fontdict=fontdict)
plt.subplot(233)
plt.plot(df.index, df.MCE)
plt.ylabel('MCE', fontdict=fontdict)
plt.xlabel('Background weight', fontdict=fontdict)
plt.subplot(236)
plt.plot(df.index, df.dMCE)
plt.ylabel('dMCE', fontdict=fontdict)
plt.xlabel('Background weight', fontdict=fontdict)
plt.tight_layout()
plt.savefig(plot_path / 'results_recalibration_study.pdf', bbox_inches='tight')
plt.show()

# %%
idx = df.MCE.argmin()
best_mce_weight = df.index[idx]
print(f'Best background weight: {best_mce_weight}')

path = paths_recal[best_mce_weight]

results, cfg = load_results(path)
mce_results = results['calibration_results']['MCE_bins']
mean_probas = np.array([res['mean_probas'] for res in mce_results])
accs = np.array([res['accs'] for res in mce_results])
n_samples = np.array([res['n_samples'] for res in mce_results])
classes = np.array([res['class'] for res in mce_results])
n_classes = len(classes)

plt.figure(figsize=(10, 12))
plt.title('MCP')
plot_MCE(plt.subplot(211), plt.subplot(212), mean_probas, accs,
         n_samples, classes, plot_legend=True, bbox_to_anchor=(0.5, 1.40), ncol=3)
plt.subplot(211)
fontdict = {'family': 'normal', 'weight': 'bold', 'size': 30}
plt.xlabel("Mean probability", fontdict=fontdict)
plt.ylabel("Accuracy", fontdict=fontdict)
plt.subplot(212)
plt.xlabel("Bins", fontdict=fontdict)
plt.ylabel("Number of Samples", fontdict=fontdict)
plt.tight_layout()
plt.savefig(plot_path / 'CP_recalibration_study.pdf', bbox_inches='tight')

# %%
"""Graphical Abstract"""


subset = 'animals'
results, cfg, detections = load_results(path=paths[f'DETR_{subset}'], return_detections=True)
tce_results = results['calibration_results']['TCE_bins'][0]
confs, accs = tce_results['confs'], tce_results['accs']

plt.figure(figsize=(5, 5))
# plt.title('Top-Label Calibration Plot', pad=15)
plt.xticks(np.linspace(0, 1, 6))
plt.plot([0, 1], [0, 1], '--k', lw=3, label='Perfect Calibration')
plt.plot(confs, accs, '-o', c='C0', label='Model Calibration')
# plt.fill_between(mean_probas_mean, accs_q25, accs_q75, alpha=.5, color='c0')
# for mean_prob, acc, c in zip(mean_probas, accs, classes):
#     plt.plot(mean_prob, acc, c=f'c{c+1}', label=f'class {c}')
plt.xlabel('Mean Probability')
plt.ylabel('Accuracy')
plt.legend(loc='lower center',  ncol=1, fontsize=18)
plt.savefig(plot_path / 'graphical_abstract_a.pdf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(5, 5))
plt.axis('off')
dets = [d for d in detections if (
    d['pred_probas'].max() > .8 and
    d['pred_probas'].argmax() == d['label'] and
    d['matched']
)]

# 5,
det = dets[3]
# print(subset_categories['all'][['label']])
img = plot_detection(det, subset=subset, plot_all_class_preds=True)
# img = plot_detections_img([d for d in detections if det['image_id'] == d['image_id']])
plt.imshow(img, aspect='auto')
plt.savefig(plot_path / 'graphical_abstract_b.pdf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(5, 5))
plt.axis('off')
dets = [d for d in detections if (
    d['pred_probas'].max() < .8 and
    d['pred_probas'].argmax() != d['label'] and
    d['matched']
)]

det = dets[3]
# print(subset_categories['all'][['label']])
img = plot_detection(det, subset=subset, plot_all_class_preds=True, offset_y=-31*3)
plt.imshow(img, aspect='auto')

plt.savefig(plot_path / 'graphical_abstract_c.pdf', bbox_inches='tight')
plt.show()


# %%
dets = [d for d in detections if det['image_id'] == d['image_id'] and d['pred_probas'].argmax() != 2]
img = plot_detections_img(dets)
plt.axis('off')
plt.imshow(img)
plt.show()

# %%


results, cfg, detections = load_results(path=paths['FRCNN_ood_application'], return_detections=True)
detections_matched = [d for d in detections if d['matched']]
ds = 'open-images'
subset = 'traffic'


def entropy_fn(p):
    return -torch.sum(p * p.clamp(min=1e-6).log())


entropies = torch.stack([entropy_fn(d['pred_probas']) for d in detections_matched])
idx = entropies.argsort()[-5]

d = detections_matched[idx]
img_id = d['image_id']
detections_img = [d for d in detections if d['image_id'] == img_id]

print(d['pred_probas'])
plt.title(entropies[idx])
# plt.imshow(plot_detection(d))
plt.imshow(plot_detections_img(detections_img, ds=ds, subset=subset))
plt.axis('off')
plt.show()

# %%
_, _, detections_id = load_results(path=paths['FRCNN_traffic_application'], return_detections=True)
_, _, detections_shifted = load_results(path=paths['FRCNN_traffic_shifted_application'], return_detections=True)
_, _, detections_ood = load_results(path=paths['FRCNN_ood_application'], return_detections=True)

# %%
cats = subset_categories['traffic'] + ['background']
class_ = 'bicycle'
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.title('In-Distribution')
plt.axis('off')
dets_id = [d for d in detections_id
           if 'gt_box' in d.keys() and cats[d['label']] == class_]
img = plot_detection(dets_id[4], subset='traffic')
plt.imshow(img)

plt.subplot(132)
plt.title('Shifted-Distribution')
plt.axis('off')
dets_shifted = [d for d in detections_shifted
                if 'gt_box' in d.keys() and cats[d['label']] == class_]
img = plot_detection(dets_shifted[3], ds='open-images', subset='traffic')
plt.imshow(img)

plt.subplot(133)
plt.title('Out-of-Distribution')
plt.axis('off')
dets_ood = [d for d in detections_ood
            if 'gt_box' in d.keys() and cats[d['pred_probas'].argmax(-1)] == class_]
#img = plot_detection(dets_ood[0], ds='open-images', subset='traffic')
img_id = detections_ood[-100]['image_id']
dets = [d for d in detections_ood if d['image_id'] == img_id]
img = plot_detection(dets[0], ds='open-images', subset='traffic')
plt.imshow(img)

plt.tight_layout()
plt.savefig(plot_path / 'graphical_abstract.pdf', bbox_inches='tight')
plt.show()


# %%
