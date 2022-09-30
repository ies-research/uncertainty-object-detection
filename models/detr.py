import copy
import logging
import torch

from detectron2.structures import Instances, Boxes
from detectron2.modeling import detector_postprocess

from detr.d2.train_net import Trainer
from detr.util import box_ops


class DETRTrainer(Trainer):
    """
    Extension of the Trainer class adapted to DETR.
    """

    @classmethod
    def build_optimizer(cls, cfg, model):
        # TODO: if statement for finetuning only last layer
        for p in model.parameters():
            p.requires_grad = False
        for p in model.detr.class_embed.parameters():
            p.requires_grad = True

        optimizer = Trainer.build_optimizer(cfg, model)
        return optimizer


@torch.no_grad()
def evaluate_detr(model, val_loader, evaluators: dict, perspective: str):
    model.eval()
    logger = logging.getLogger("detectron2")
    logger.info(f"Start inference on {len(val_loader)} batches")
    for i_batch, inputs in enumerate(val_loader, start=1):
        # Preprocess image
        img_list = model.preprocess_image(inputs)

        # Forward prop through detr
        outputs = model.detr(img_list)

        box_cls = outputs['pred_logits']
        box_pred = outputs['pred_boxes']

        # Create instance object before post processing
        unprocessed_results = []
        for logits, boxes in zip(box_cls, box_pred):
            r = inference_unprocessed(boxes, logits)
            unprocessed_results.append({"instances": r})

        # Postprocessing Step
        results = inference_postprocessed(box_cls, box_pred, None, img_list.image_sizes)
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(results, inputs, img_list.image_sizes):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            # deepcopy is important since boxes of results are also changed and do not fit to original image
            r = detector_postprocess(copy.deepcopy(results_per_image), height, width)
            processed_results.append({"instances": r})

        if 'coco' in evaluators.keys():
            evaluators['coco'].process(inputs, processed_results)
        

        # Modelling Perspective
        if 'calibration' in evaluators.keys():
            if perspective == 'modeling':
                evaluators['calibration'].process(inputs, unprocessed_results)
            elif perspective == 'application':
                evaluators['calibration'].process(inputs, processed_results)
        
        if 'ood' in evaluators.keys():
            if perspective == 'modeling':
                evaluators['ood'].process(inputs, unprocessed_results)
            elif perspective == 'application':
                evaluators['ood'].process(inputs, processed_results)

        if (i_batch) % 10 == 0:
            logger.info(f'Current batch: {i_batch} / {len(val_loader)}')
    return evaluators


def inference_unprocessed(boxes, logits):
    r = Instances(image_size=(1, 1))
    r.pred_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(boxes))
    r.pred_logits = logits
    r.pred_probas = logits.softmax(-1)
    r.pred_classes = logits.argmax(-1)
    return r


def inference_postprocessed(box_cls, box_pred, mask_pred, image_sizes):
    """Adapts the detr inference to output probability per detection."""
    assert len(box_cls) == len(image_sizes)
    results = []

    probas = box_cls.softmax(-1)
    # For each box we assign the best class or the second best if the best on is `no_object`. For coco.
    scores, labels = probas[:, :, :-1].max(-1)

    for probas_per_image, scores_per_image, labels_per_image, box_pred_per_image, image_size in zip(
        probas, scores, labels, box_pred, image_sizes
    ):
        result = Instances(image_size)
        result.pred_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(box_pred_per_image))

        result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])

        result.scores = scores_per_image
        result.pred_classes = labels_per_image
        result.pred_probas = probas_per_image
        results.append(result)
    return results
