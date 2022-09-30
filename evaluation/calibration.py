import os

import copy
import json
import torch
import torch.nn as nn
import numpy as np
import pylab as plt

from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou

from detr.util import misc as utils


class CalibrationEvaluator(object):
    def __init__(self, perspective: str, num_classes: int, matcher: nn.Module , n_bins: int = 10, output_dir: str = None):
        """Evaluation of several calibration metrics in object detection."""
        super().__init__()
        self.perspective = perspective
        self.num_classes = num_classes
        self.n_bins = n_bins
        self.output_dir = output_dir

        self.matcher = matcher
        self.detections = []
        self.results = {}

    def reset(self):
        self.detections = []
        self.results = {}

    def process(self, targets, outputs):
        """Collect detections for objects. Matching is done via Hungarian Matcher and IOU.

        Postprocess outputs of model such that probas and boxes are in there.

        Args:
            targets (list): TODO
            outputs (list): TODO
        """
        # Assumes currently only one pred in outputs
        target_instances = copy.deepcopy(targets[0]['instances'].to('cpu'))
        output_instances = copy.deepcopy(outputs[0]['instances'].to('cpu'))
        assert 'pred_probas' in output_instances.get_fields().keys()

        # Scale boxes to [0, 1] for matching
        target_instances.gt_boxes.scale(1/target_instances.image_size[1], 1/target_instances.image_size[0])
        output_instances.pred_boxes.scale(1/output_instances.image_size[1], 1/output_instances.image_size[0])

        gt_boxes = target_instances.gt_boxes.tensor
        pred_boxes = output_instances.pred_boxes.tensor

        # Matching between targets and predictions
        pred_indices, gt_indices = torch.arange(len(pred_boxes)), torch.arange(len(gt_boxes))
        gt_indices_matched, pred_indices_matched = self.matcher(gt_boxes, pred_boxes)
        pred_indices_unmatched = pred_indices[~np.isin(pred_indices, pred_indices_matched)]
        gt_indices_unmatched = gt_indices[~np.isin(gt_indices, gt_indices_matched)]

        # Create evaluation set
        img_id = targets[0]["image_id"]
        # matched predictions
        for pred_idx, gt_idx in zip(pred_indices_matched, gt_indices_matched):
            det = {
                "image_id": img_id,
                "matched": True,
                # Targets
                "label": target_instances.gt_classes[gt_idx],
                "gt_box": gt_boxes[gt_idx],
                # Predictions
                "pred_box": pred_boxes[pred_idx],
                "pred_probas": output_instances.pred_probas[pred_idx],
            }
            self.detections.append(det)
        # unmatched predictions
        for pred_idx in pred_indices_unmatched:
            # unmatched predictions
            det = {
                "image_id": img_id,
                "matched": False,
                # Targets
                "label": -1,
                # Predictions
                "pred_box": pred_boxes[pred_idx],
                "pred_probas": output_instances.pred_probas[pred_idx],
            }
            self.detections.append(det)
        # unmatched ground truth
        for gt_idx in gt_indices_unmatched:
            det = {
                "image_id": img_id,
                "matched": False,
                # Targets
                "label": target_instances.gt_classes[gt_idx],
                "gt_box": gt_boxes[gt_idx],
                # Predictions
                "pred_class": -1,
                "pred_probas": torch.eye(self.num_classes + 1)[-1]
            }
            self.detections.append(det)

    def synchronize_between_processes(self):
        detections_list = utils.all_gather(self.detections)
        detections_ = []
        for l in detections_list:
            detections_.extend(l)
        self.detections = detections_

    def accumulate(self):
        probas_accumulated, labels_accumulated, matches_accumulated = [], [], []
        # scores_accumulated, pred_classes_accumulated = [], []
        for d in self.detections:
            labels_accumulated.append(d['label'])
            matches_accumulated.append(d['matched'])
            probas_accumulated.append(d['pred_probas'])
        labels_accumulated = torch.Tensor(labels_accumulated).long()
        matches_accumulated = torch.Tensor(matches_accumulated).bool()
        probas_accumulated = torch.stack(probas_accumulated).float()

        self.background_label = labels_accumulated.max()+1
        labels_accumulated[labels_accumulated == -1] = self.background_label

        assert torch.isclose(probas_accumulated.sum(-1), torch.tensor(1).float()).all()
        probas_accumulated[probas_accumulated == 0] += 1e-3  # TODO: Remove zeros for numerical stability
        return probas_accumulated, labels_accumulated, matches_accumulated

    def evaluate(self):
        self.synchronize_between_processes()
        if len(self.detections) == 0:
            print('No detections for for evaluation of calibration metrics.')
            return

        probas_accumulated, labels_accumulated, matches_accumulated = self.accumulate()

        # Save stats about detections
        mask_unmatched_preds = (labels_accumulated == self.background_label) & ~matches_accumulated
        mask_missing_preds = (labels_accumulated != self.background_label) & ~matches_accumulated
        d = {
            'matched_preds': matches_accumulated.tolist(), 'n_matched_preds': matches_accumulated.sum().item(),
            'unmatched_preds': mask_unmatched_preds.tolist(), 'n_unmatched_preds': mask_unmatched_preds.sum().item(),
            'missing_preds': mask_missing_preds.tolist(), 'n_missing_preds': mask_missing_preds.sum().item(),
        }
        self.results.update(d)

        # Compute proper scoring rules
        d = self.proper_scoring(probas_accumulated, labels_accumulated)
        self.results.update(d)

        # Calibration errors (acc) on all predictions (FP have the background label as ground truth)
        d = self.calibration_errors(probas_accumulated, labels_accumulated)
        self.results.update(d)

        # Calibration errors (precision) on all predictions (ignores background proba at last column)
        d = self.calibration_errors_detection(probas_accumulated, labels_accumulated, matches_accumulated)
        self.results.update(d)

        # Compute entropies for the entropy histogram

        self.summarize()
        if self.output_dir:
            print(f'Saving calibration results to {self.output_dir}')
            # Save detections
            file_path = os.path.join(self.output_dir, "detections.pt")
            torch.save(self.detections, file_path)

            # Save calibration results
            file_path = os.path.join(self.output_dir, "calibration_results.json")
            with open(file_path, "w") as f:
                f.write(json.dumps(self.results))
                f.flush()

            # Save calibration plots
            # fig = plot_TCE_bins(self.results['TCE_bins_matched'])
            # fig.savefig(os.path.join(self.output_dir, 'TCE_bins_matched.pdf'))
            fig = plot_TCE_bins(self.results['TCE_bins'])
            fig.savefig(os.path.join(self.output_dir, 'TCE_bins.pdf'))
            fig = plot_TCE_bins(self.results['dTCE_bins'], k2='precisions')
            fig.savefig(os.path.join(self.output_dir, 'dTCE_bins.pdf'))

            # fig = plot_MCE_bins(self.results['MCE_bins_matched'])
            # fig.savefig(os.path.join(self.output_dir, 'MCE_bins_matched.pdf'))
            fig = plot_MCE_bins(self.results['MCE_bins'])
            fig.savefig(os.path.join(self.output_dir, 'MCE_bins.pdf'))
            fig = plot_MCE_bins(self.results['dMCE_bins'], k2='precisions')
            fig.savefig(os.path.join(self.output_dir, 'dMCE_bins.pdf'))


        return self.results

    def proper_scoring(self, probas_accumulated, labels_accumulated):
        criterion_nll = NegativeLogLikelihood()
        criterion_bs = BrierScore()

        nll = criterion_nll(probas_accumulated, labels_accumulated)
        bs = criterion_bs(probas_accumulated, labels_accumulated)
        res = {
            'NLL': nll.item(),
            'BS': bs.item(),
        }
        return res


    def calibration_errors(self, probas_accumulated, labels_accumulated):
        # Convert everything from torch.Tensor
        def to_list(x): return x.tolist() if isinstance(x, torch.Tensor) else x
        # Top label calibration error
        criterion_tce = TopLabelCalibrationError(self.n_bins)
        tce = criterion_tce(probas_accumulated, labels_accumulated).item()
        tce_bins = [{k: to_list(v) for k, v in criterion_tce.results.items()}]
        # Marginal calibratinon error
        criterion_mce = MarginalCalibrationError(self.n_bins)
        mce = criterion_mce(probas_accumulated, labels_accumulated).item()
        mce_bins = [{k: to_list(v) for k, v in res.items()} for res in criterion_mce.results]

        res = {
            'MCE': mce,
            'MCE_bins': mce_bins,
            'TCE': tce,
            'TCE_bins': tce_bins,
        }
        return res

    def calibration_errors_detection(self, probas_accumulated, labels_accumulated, matches_accumulated):
        def to_list(x): return x.tolist() if isinstance(x, torch.Tensor) else x

        criterion_tce = TopLabelCalibrationErrorDetection(self.n_bins)
        tce = criterion_tce(probas_accumulated, labels_accumulated, matches_accumulated)

        criterion_mce = MarginalCalibrationErrorDetection(self.n_bins)
        mce = criterion_mce(probas_accumulated, labels_accumulated, matches_accumulated)

        tce = tce.item()
        tce_plot = [{k: to_list(v) for k, v in res.items()} for res in [criterion_tce.results]]
        mce = mce.item()
        mce_plot = [{k: to_list(v) for k, v in res.items()} for res in criterion_mce.results]

        res = {
            'dMCE': mce,
            'dMCE_bins': mce_plot,
            'dTCE': tce,
            'dTCE_bins': tce_plot,
        }
        return res

    def summarize(self):
        template = ' {:<10} = {:0.3f}'
        for name, val in self.results.items():
            if isinstance(val, list):
                continue
            print(template.format(name, val))


def calibration_error(confs: torch.Tensor, accs: torch.Tensor, n_samples: torch.Tensor, p: int = 2, bin_weight: str = 'samples'):
    if bin_weight == 'samples':
        probas_bin = n_samples/n_samples.nansum()
    elif bin_weight == 'equal':
        probas_bin = 1 / (n_samples != 0).sum()
    ce = (torch.nansum(probas_bin * (confs-accs)**p))**(1/p)
    return ce


class TopLabelCalibrationError(nn.Module):
    """Computes the calibration plot for each class."""

    def __init__(self, n_bins=10, p=2):
        super().__init__()
        self.n_bins = n_bins
        self.p = p

    def forward(self, probas: torch.Tensor, labels: torch.Tensor):
        bins = torch.linspace(0, 1, self.n_bins+1)

        confs = torch.Tensor(self.n_bins)
        accs = torch.Tensor(self.n_bins)
        n_samples = torch.Tensor(self.n_bins)

        pred_confs, pred_labels = probas.max(dim=-1)

        for i_bin, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
            in_bin = (bin_start < pred_confs) & (pred_confs <= bin_end)
            n_samples[i_bin] = in_bin.sum()

            if in_bin.sum() == 0:
                confs[i_bin] = float('nan')
                accs[i_bin] = float('nan')
                continue

            bin_conf = pred_confs[in_bin].mean()
            bin_acc = (pred_labels[in_bin] == labels[in_bin]).float().mean()

            confs[i_bin] = bin_conf
            accs[i_bin] = bin_acc

        self.results = {'confs': confs, 'accs': accs, 'n_samples': n_samples}
        return calibration_error(confs, accs, n_samples, self.p)


class MarginalCalibrationError(nn.Module):
    """Computes the calibration plot for each class."""

    def __init__(self, n_bins=10, p=2):
        super().__init__()
        self.n_bins = n_bins
        self.p = p
        self.results = []

    def forward(self, probas: torch.Tensor, labels: torch.Tensor):
        bins = torch.linspace(0, 1, self.n_bins+1)
        _, n_classes = probas.shape

        # Save calibration plots in results
        self.results = []
        for i_cls in range(n_classes):
            label = (labels == i_cls).long()
            proba = probas[:, i_cls]

            mean_probas = torch.Tensor(self.n_bins)
            accs = torch.Tensor(self.n_bins)
            n_samples = torch.Tensor(self.n_bins)
            for i_bin, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
                in_bin = (bin_start < proba) & (proba <= bin_end)
                n_samples[i_bin] = in_bin.sum()

                if in_bin.sum() == 0:
                    mean_probas[i_bin] = float('nan')
                    accs[i_bin] = float('nan')
                    continue

                bin_conf = proba[in_bin].mean()
                bin_acc = (label[in_bin] == 1).float().mean()

                mean_probas[i_bin] = bin_conf
                accs[i_bin] = bin_acc
            self.results.append({'mean_probas': mean_probas, 'accs': accs, 'n_samples': n_samples, 'class': i_cls})

        sq_ces = [calibration_error(d['mean_probas'], d['accs'], d['n_samples'], self.p)**self.p for d in self.results]
        mce = torch.Tensor(sq_ces).mean()**(1/self.p)
        return mce


class TopLabelCalibrationErrorDetection(nn.Module):

    def __init__(self, n_bins=10, p=2):
        super().__init__()
        self.n_bins = n_bins
        self.p = p

    def forward(self, probas, labels, matchings):
        bins = torch.linspace(0, 1, self.n_bins + 1)
        mean_confs_, precisions_, n_samples_ = [], [], []

        pred_probas = probas[:, :-1]  # remove background class, should not be used for computing precision
        pred_confs, pred_labels = pred_probas.max(-1)

        # Compute the precision and recalls
        for bin_start, bin_end in zip(bins[:-1], bins[1:]):
            indices = (bin_start < pred_confs) & (pred_confs <= bin_end)

            matchings_bin = matchings[indices]

            pred_labels_bin = pred_labels[indices][matchings_bin]
            gt_labels_bin = labels[indices][matchings_bin]
            corrects_bin = pred_labels_bin == gt_labels_bin

            true_positives = torch.sum(corrects_bin)
            false_positives = torch.sum(~matchings_bin) + torch.sum(~corrects_bin)

            n_samples = torch.sum(indices)
            n_samples_.append(n_samples)
            if n_samples != 0:
                mean_confs_.append(pred_confs[indices].mean())
                precisions_.append(true_positives / (true_positives + false_positives))
            else:
                mean_confs_.append(float("nan"))
                precisions_.append(float("nan"))
        self.results = {
            'confs': torch.Tensor(mean_confs_),
            'precisions': torch.Tensor(precisions_),
            'n_samples': torch.Tensor(n_samples_)
        }
        ce = calibration_error(self.results['confs'], self.results['precisions'], self.results['n_samples'])
        return ce


class MarginalCalibrationErrorDetection(nn.Module):
    """Computes the calibration plot for each class."""

    def __init__(self, n_bins=10, p=2):
        super().__init__()
        self.n_bins = n_bins
        self.p = p
        self.results = []

    def forward(self, probas, labels, matchings):
        """Computes a calibration plot (conf -> precision) per class."""
        bins = torch.linspace(0, 1, self.n_bins + 1)

        pred_probas = probas[:, :-1]  # remove proba for background class since we want to compute precision

        self.results = []
        for i_cls in range(pred_probas.size(1)):
            mean_probas_, precisions_, n_samples_ = [], [], []

            for bin_start, bin_end in zip(bins[:-1], bins[1:]):
                probas_cls_bin = pred_probas[:, i_cls]
                indices = (bin_start < probas_cls_bin) & (probas_cls_bin <= bin_end)

                n_samples_bin = torch.sum(indices)
                matchings_bin = matchings[indices]
                gt_labels_bin = labels[indices][matchings_bin]
                corrects_bin = i_cls == gt_labels_bin

                true_positives = torch.sum(corrects_bin)
                false_positives = torch.sum(~matchings_bin) + torch.sum(~corrects_bin)

                n_samples_.append(n_samples_bin)
                if n_samples_bin != 0:
                    mean_probas_.append(probas_cls_bin[indices].mean())
                    precisions_.append(true_positives / (true_positives + false_positives))
                else:
                    mean_probas_.append(float("nan"))
                    precisions_.append(float("nan"))

            self.results.append({
                "mean_probas": torch.Tensor(mean_probas_),
                "precisions": torch.Tensor(precisions_),
                "n_samples": torch.Tensor(n_samples_),
                "class": i_cls,
            })

        sq_ces = [
            calibration_error(d['mean_probas'], d['precisions'], d['n_samples'], self.p)**self.p
            for d in self.results
        ]
        mce = torch.Tensor(sq_ces).mean()**(1/self.p)
        return mce


class NegativeLogLikelihood(nn.Module):
    def __init__(self):
        super().__init__()
        self.nll_criterion = nn.NLLLoss(reduction='none')

    def forward(self, probas, labels):
        N, D = probas.shape
        assert len(labels) == N, "Probas and Labels must be of the same size"
        assert len(labels.unique()) == D, "Missing labels?"
        log_probas = probas.log()
        score = torch.mean(self.nll_criterion(log_probas, labels.long()))
        return score


class BrierScore(nn.Module):
    def forward(self, probas, labels):
        N, D = probas.shape
        assert len(labels) == N, "Probas and Labels must be of the same size"
        assert len(labels.unique()) == D, "Missing labels?"
        onehot_encoder = torch.eye(D, device=probas.device)
        y_onehot = onehot_encoder[labels.long()]
        score = torch.mean(torch.sum((probas - y_onehot)**2, dim=-1))
        return score


class IOUMatcher(nn.Module):

    def __init__(self, iou_thresh: float = 0.5, use_hungarian: bool = True):
        """Matcher used for evaluation.

        Args:
            iou_thresh (float, optional): Defines the IOU threshold used for matching. Defaults to 0.5.
            use_hungarian (bool, optional): Defines whether to use hungarian algorithm (1 assignment) or all assignments
                that are over the IOU threshold. Defaults to True.
        """
        super().__init__()
        self.iou_thresh = iou_thresh
        self.min_iou = 0.0
        self.use_hungarian = use_hungarian

    @torch.no_grad()
    def forward(self, bbox_gt: torch.Tensor, bbox_pred: torch.Tensor):
        # Ref: https://gist.github.com/AruniRC/c629c2df0e68e23aff7dcaeef87c72d4
        n_gts, n_preds = bbox_gt.shape[0], bbox_pred.shape[0]

        # NUM_GT x NUM_PRED
        iou_matrix = box_iou(bbox_gt.cpu(), bbox_pred.cpu())
        iou_matrix = iou_matrix.numpy()

        if self.use_hungarian:  # Hungarian matching
            idx_gts, idx_preds = linear_sum_assignment(1 - iou_matrix)
        else:
            idx_gts, idx_preds = np.where(iou_matrix > .5)

        # remove dummy assignments
        real_pred_mask = idx_preds < n_preds
        idx_pred_actual = idx_preds[real_pred_mask]
        idx_gt_actual = idx_gts[real_pred_mask]
        ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
        valid_mask = (ious_actual > self.iou_thresh)

        return idx_gt_actual[valid_mask], idx_pred_actual[valid_mask]


def plot_detection(d: dict):
    from PIL import Image
    from detectron2.utils.visualizer import Visualizer
    fname = '/mnt/media/datasets/coco/val2017/{}.jpg'.format(str(d['image_id']).zfill(12))
    img = Image.open(fname)
    v = Visualizer(img)
    if 'gt_box' in d.keys():
        gt_box = d['gt_box'] * torch.tensor([*img.size, *img.size])
        o = v.draw_box(gt_box, alpha=1, line_style=':', edge_color='k')
        o = v.draw_text(d['label'].item(), gt_box[:2] + torch.tensor([10, 0]), font_size=24, color='r')
    if 'pred_box' in d.keys():
        pred_box = d['pred_box'] * torch.tensor([*img.size, *img.size])
        o = v.draw_box(pred_box)
        # o = v.draw_text(d['logits'].argmax(-1).item(), pred_box[:2] + torch.tensor([40, 0]), font_size=24, color='g')
    plt.figure()
    plt.imshow(o.get_image())
    plt.axis('off')
    plt.savefig('tmp.png')


def plot_TCE_bins(result, k1='confs', k2='accs', k3='n_samples'):
    confs, accs, n_samples = result[0][k1], result[0][k2], result[0][k3]
    n_bins = len(confs)
    fig = plt.figure()
    plt.subplot(211)
    plt.plot(confs, accs, '-o')
    plt.plot([0, 1], '--', linewidth=3, color='black')
    plt.ylabel("Accuracy") if k2 == 'accs' else plt.ylabel("Precision")
    plt.subplot(212)
    plt.bar(np.linspace(0, 1, n_bins), n_samples, width=1/n_bins)
    plt.xticks(np.linspace(0, 1, n_bins), map(lambda x: f"{x:.0f}", n_samples), rotation=45)
    plt.ylabel("Number of Samples")
    fig.tight_layout()
    return fig


def plot_MCE_bins(result, k1='mean_probas', k2='accs', k3='n_samples', q=.25):
    n_bins = len(result[0][k1])
    probas = np.nanmean([res[k1] for res in result], axis=0)
    precisions = np.nanmean([res[k2] for res in result], axis=0)
    quantile_top = np.nanquantile([res[k2] for res in result], 1-q, axis=0)
    quantile_bottom = np.nanquantile([res[k2] for res in result], q, axis=0)
    n_samples = np.nanmean([res[k3] for res in result], axis=0)

    fig = plt.figure()
    plt.subplot(211)
    plt.plot(probas, precisions, '-o', color='C0')
    plt.fill_between(probas, quantile_top, quantile_bottom, alpha=0.3, color='C0')
    plt.plot([0, 1], '--', linewidth=3, color='black')
    [plt.plot(res[k1], res[k2], label=res['class'], linewidth=1) for res in result]
    plt.xlabel("Mean Probabilities")
    plt.ylabel("Accuracy") if k2 == 'accs' else plt.ylabel("Precision")
    plt.legend(loc='upper right', bbox_to_anchor=[1.15, 1])

    plt.subplot(212)
    plt.bar(np.linspace(0, 1, n_bins), n_samples, width=1/n_bins)
    plt.xticks(np.linspace(0, 1, n_bins), map(lambda x: f"{x:.0f}", n_samples), rotation=45)
    plt.ylabel("Number of Samples")
    fig.tight_layout()
    return fig

def plot_predictions_from_instances(fname, instances):
    from detectron2.utils.visualizer import Visualizer
    from PIL import Image
    import pylab as plt
    plt.axis('off')
    img = Image.open(fname)
    inst = copy.deepcopy(instances)
    if inst.has('gt_boxes'):
        inst.pred_boxes = inst.gt_boxes
        inst.pred_classes = inst.gt_classes
    w, h = img.size
    v = Visualizer(img)
    inst.pred_boxes.scale(w, h)
    o = v.draw_instance_predictions(inst.to('cpu'))
    plt.imshow(o.get_image())
    plt.savefig('tmp.png')
