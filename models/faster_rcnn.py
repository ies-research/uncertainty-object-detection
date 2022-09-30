from typing import Tuple
import os
import logging
import torch

from detectron2.structures import Instances, Boxes
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.layers import batched_nms
from detectron2.modeling import detector_postprocess


class FasterRCNNTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_optimizer(cls, cfg, model):
        for p in model.parameters():
            p.requires_grad = False
        for p in model.roi_heads.box_predictor.parameters():
            p.requires_grad = True

        optimizer = DefaultTrainer.build_optimizer(cfg, model)
        return optimizer


@torch.no_grad()
def evaluate_faster_rcnn(model, val_loader, evaluators: dict, perspective: str):
    model.eval()
    logger = logging.getLogger("detectron2")
    logger.info(f"Start inference on {len(val_loader)} batches")
    for i_batch, inputs in enumerate(val_loader, start=1):
        # Preprocess image
        img_list = model.preprocess_image(inputs)

        features = model.backbone(img_list.tensor)

        # generate proposals for images
        proposals, _ = model.proposal_generator(img_list, features, None)

        # get features to look at (FPN)
        box_in_features = model.roi_heads.box_in_features
        features_ = [features[f] for f in box_in_features]

        # ROI Pooling
        box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
        box_features = model.roi_heads.box_head(box_features)

        # Predict for box_features
        logits, proposal_deltas = model.roi_heads.box_predictor(box_features)
        logits_ = logits.split([len(p) for p in proposals], dim=0)
        probas = [l.softmax(-1) for l in logits_]
        boxes = model.roi_heads.box_predictor.predict_boxes((logits, proposal_deltas), proposals)
        image_shapes = [x.image_size for x in proposals]

        # Get unprocessed results
        unprocessed_results = []
        for b, p, img_shape in zip(boxes, probas, image_shapes):
            res = fast_rcnn_inference_unprocessed(b, p, img_shape)
            unprocessed_results.append({'instances': res})

        # Post-Processing
        processed_results = []
        for b, p, img_shape in zip(boxes, probas, image_shapes):
            result_per_image = fast_rcnn_inference_postprocessed(
                b, p, img_shape,
                score_thresh=model.roi_heads.box_predictor.test_score_thresh,
                nms_thresh=model.roi_heads.box_predictor.test_nms_thresh,
                topk_per_image=model.roi_heads.box_predictor.test_topk_per_image
            )
            res = result_per_image[0]
            #res = model.roi_heads.forward_with_given_boxes(features, res)
            height, width = inputs[0].get('height', img_shape[0]), inputs[0].get('width', img_shape[1])
            res = detector_postprocess(res, height, width)
            processed_results.append({'instances': res})
        

        if 'coco' in evaluators.keys():
            evaluators['coco'].process(inputs, processed_results)

        if 'calibration' in evaluators.keys():
            if perspective == 'modeling':
                evaluators['calibration'].process(inputs, unprocessed_results)
            elif perspective == 'application':
                evaluators['calibration'].process(inputs, processed_results)

        if (i_batch) % 10 == 0:
            logger.info(f'Current batch: {i_batch} / {len(val_loader)}')
    return evaluators


def fast_rcnn_inference_unprocessed(boxes, probas, image_shape: Tuple[int, int]):
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(probas).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        probas = probas[valid_mask]

    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
    
    # Extend probas for each box, as it is class agnostic
    pred_boxes = torch.cat([boxes[:, i] for i in range(num_bbox_reg_classes)])
    pred_probas = torch.cat([probas for _ in range(num_bbox_reg_classes)])

    result = Instances(image_shape)
    result.pred_boxes = Boxes(pred_boxes)
    result.pred_probas = pred_probas
    return result

def fast_rcnn_inference_postprocessed(
    boxes,
    probas,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(probas).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        probas = probas [valid_mask]

    scores = probas[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    pred_probas = probas[filter_inds[:, 0]]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds, pred_probas = boxes[keep], scores[keep], filter_inds[keep], pred_probas[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_probas = pred_probas
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]