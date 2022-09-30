import os
import copy
import torch
import numpy as np
import pylab as plt

from detr.util import misc as utils


class OutOfDistributionEvaluator(object):
    def __init__(self, matcher, num_classes, output_dir):
        self.matcher = matcher
        self.num_classes = num_classes
        self.output_dir = output_dir

        self.label_background = -1
        self.proba_background = torch.eye(self.num_classes + 1)[self.label_background]

    def process(self, targets, outputs):
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
                "label": self.label_background,
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
                "pred_class": self.label_background,
                "pred_probas": self.proba_background,
            }
            self.detections.append(det)

    def reset(self):
        self.detections = []
        self.results = {}

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

        d = self.entropy_fn(probas_accumulated[matches_accumulated | mask_missing_preds])
        self.results.update(d)

        self.summarize()
        if self.output_dir:
            # Plot entropy histogram
            fig = plt.figure()
            plt.hist(self.results['entropies'], bins=50)
            fig.savefig(os.path.join(self.output_dir, 'entropy_hist.pdf'))

    def summarize(self):
        template = ' {:<10} = {:0.3f}'
        for name, val in self.results.items():
            if isinstance(val, list):
                continue
            print(template.format(name, val))

    def entropy_fn(self, probas):
        entropies = - torch.sum(probas * probas.log(), dim=-1)
        res = {
            'avg_entropy': entropies.mean().item(),
            'entropies': entropies.tolist(),
        }
        return res
