import os
import torch

from torch.utils.data import SequentialSampler
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog
from detectron2.data.build import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances

from detr.d2.detr import add_detr_config
from detr.d2.detr import DetrDatasetMapper

from models.detr import evaluate_detr
from models.faster_rcnn import evaluate_faster_rcnn
from evaluation.calibration import CalibrationEvaluator, IOUMatcher
from evaluation.ood import OutOfDistributionEvaluator


def main(args):
    cfg = get_config(args)
    build_dataset(args, cfg)
    default_setup(cfg, args)

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    mapper = DetrDatasetMapper(cfg, is_train=False)
    dataset = DatasetCatalog.get(cfg.DATASETS.TEST[0])
    val_loader = build_detection_test_loader(dataset, mapper=mapper, sampler=SequentialSampler(dataset))
    evaluators = {}


    coco_evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], tasks=('bbox',), output_dir=cfg.OUTPUT_DIR)
    coco_evaluator.reset()
    evaluators.update({'coco': coco_evaluator})

    # Use hungarian if NMS is present or DETR it used as it avoids duplicate predictions
    use_hungarian = (args.perspective == 'application') or (cfg.MODEL.META_ARCHITECTURE == 'Detr')
    matcher = IOUMatcher(iou_thresh=.5, use_hungarian=use_hungarian)
    calibration_evaluator = CalibrationEvaluator(
        num_classes=cfg.MODEL.DETR.NUM_CLASSES,
        matcher=matcher,
        perspective=args.perspective,
        output_dir=cfg.OUTPUT_DIR
    )
    calibration_evaluator.reset()
    evaluators.update({'calibration': calibration_evaluator})

    if args.eval_ood:
        ood_evaluator = OutOfDistributionEvaluator(
            matcher=matcher,
            num_classes=cfg.MODEL.DETR.NUM_CLASSES,
            output_dir=cfg.OUTPUT_DIR,
        )
        ood_evaluator.reset()
        evaluators.update({'ood': ood_evaluator})

    if cfg.MODEL.META_ARCHITECTURE == 'Detr':
        evaluators = evaluate_detr(model, val_loader, evaluators=evaluators, perspective=args.perspective)
    elif cfg.MODEL.META_ARCHITECTURE == 'GeneralizedRCNN':
        evaluators = evaluate_faster_rcnn(model, val_loader, evaluators=evaluators, perspective=args.perspective)

    # Accumulate results
    results = {'args': args}

    print('Computing coco results.')
    coco_results = evaluators['coco'].evaluate()
    results.update({'coco_results': coco_results})

    print('Computing calibration results.')
    calibration_results = evaluators['calibration'].evaluate()
    results.update({'calibration_results': calibration_results})

    if args.eval_ood:
        print('Computing OOD results.')
        ood_results = evaluators['ood'].evaluate()
        results.update({'ood_results': ood_results})

    torch.save(results, os.path.join(cfg.OUTPUT_DIR, 'results_final.pth'))


def get_config(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_detr_config(cfg)

    if args.eval_from_config:
        assert os.path.isfile(args.eval_from_config), 'For `--eval_from_config`, specify an existing config path'
        args.config_file = args.eval_from_config
        cfg.merge_from_file(args.eval_from_config)
    else:
        assert args.experiment_path, 'You need to specify the experiment path incl. `config.yaml` and `model_final.pth`'
        cfg.merge_from_file(os.path.join(args.experiment_path, 'config.yaml'))
        cfg.MODEL.WEIGHTS = os.path.join(args.experiment_path, 'model_final.pth')
    cfg.merge_from_list(args.opts)
    assert os.path.isfile(cfg.MODEL.WEIGHTS), 'The config\'s model weights do not exist.'
    cfg.OUTPUT_DIR = os.path.join(args.result_path)
    cfg.freeze()
    return cfg


def build_dataset(args, cfg):
    if args.dataset == 'coco':
        cfg = build_coco(args, cfg)
    elif args.dataset == 'open-images':
        cfg = build_openimages(args, cfg)
    return cfg


def build_openimages(args, cfg):
    cfg.defrost()
    cfg.DATASETS.TRAIN = None
    if 'animals' in cfg.DATASETS.TEST[0]:
        cfg.DATASETS.TEST = ('open-images_animals',)
    elif 'traffic' in cfg.DATASETS.TEST[0]:
        cfg.DATASETS.TEST = ('open-images_traffic',)
    else:
        raise ValueError('Not implemented yet.')
    register_coco_instances(cfg.DATASETS.TEST[0], {}, args.ann_path, args.ds_path)
    cfg.freeze()
    return cfg


def build_coco(args, cfg):
    cfg.defrost()
    cfg.DATASETS.TRAIN = None
    image_root_val = os.path.join(args.ds_path, "val2017")
    if cfg.DATASETS.TEST != ('coco_2017_val',):
        ann_file_val = os.path.join(args.experiment_path, "subset_instances_val2017.json")
        register_coco_instances(cfg.DATASETS.TEST[0], {}, ann_file_val, image_root_val)
    cfg.freeze()
    return cfg


if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument('--dataset', type=str, choices=['coco', 'open-images'], default='coco')
    parser.add_argument('--ds_path', type=str, default='coco')
    parser.add_argument('--ann_path', type=str, default='coco/annotations/')
    parser.add_argument('--experiment_path', type=str)
    parser.add_argument('--eval_from_config', type=str, default=None)
    parser.add_argument('--result_path', type=str, default='./results')
    parser.add_argument('--perspective', type=str, choices=['application', 'modeling'])
    parser.add_argument('--eval_ood', action='store_true')
    args = parser.parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
