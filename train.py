import os 

from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from detr.d2.detr import add_detr_config
from models.detr import DETRTrainer
from models.faster_rcnn import FasterRCNNTrainer
from models.retinanet import RetinaNetTrainer
from datasets.coco_subset import create_subset_json, register_coco_instances


def main(args):
    print("Command Line Args:", args)
    # Setup
    cfg = get_config(args)
    cfg.defrost()
    if args.background_weight:
        cfg.MODEL.DETR.NO_OBJECT_WEIGHT = args.background_weight
    cfg.freeze()
    build_coco(args, cfg)
    default_setup(cfg, args)

    if cfg.MODEL.META_ARCHITECTURE == 'Detr':
        trainer = DETRTrainer(cfg)
    elif cfg.MODEL.META_ARCHITECTURE == 'GeneralizedRCNN':
        trainer = FasterRCNNTrainer(cfg)
    elif cfg.MODEL.META_ARCHITECTURE == 'RetinaNet':
        trainer = RetinaNetTrainer(cfg)

    trainer.resume_or_load(resume=False)
    trainer.train()


def get_config(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_detr_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def build_coco(args, cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.defrost()
    ann_file_train = os.path.join(args.ann_path, "instances_train2017.json")
    image_root_train = os.path.join(args.coco_path, "train2017")
    ann_file_val = os.path.join(args.ann_path, "instances_val2017.json")
    image_root_val = os.path.join(args.coco_path, "val2017")

    if args.subset == "all":
        cfg.DATASETS.TRAIN = ("coco_2017_train",)
        cfg.DATASETS.TEST = ("coco_2017_val",)
        cfg.MODEL.DETR.NUM_CLASSES = 80
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
        cfg.MODEL.RETINANET.NUM_CLASSES = 80
        cfg.freeze()
        return cfg

    ds_name_train, ds_name_val = f"coco_2017_train_{args.subset}", f"coco_2017_val_{args.subset}"
    if args.subset == "animals":
        category_names = ["giraffe", "elephant"]
    elif args.subset == "traffic":
        category_names = ['person', 'bicycle', 'car', 'motorcycle', 'bus',
                          'train', 'truck', 'traffic light', 'fire hydrant', 'stop sign']
    else:
        raise ValueError(f'Subset {args.subset} not defined.')

    ann_file_subset = create_subset_json(ann_file_train, category_names, cfg.OUTPUT_DIR)
    register_coco_instances(ds_name_train, {}, ann_file_subset, image_root_train)
    ann_file_subset = create_subset_json(ann_file_val, category_names, cfg.OUTPUT_DIR)
    register_coco_instances(ds_name_val, {}, ann_file_subset, image_root_val)
    cfg.MODEL.DETR.NUM_CLASSES = len(category_names)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(category_names)
    cfg.MODEL.RETINANET.NUM_CLASSES = len(category_names)
    cfg.DATASETS.TRAIN = (ds_name_train,)
    cfg.DATASETS.TEST = (ds_name_val,)
    cfg.freeze()
    return cfg


if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument('--coco_path', type=str, default='coco')
    parser.add_argument('--ann_path', type=str, default='coco/annotations/')
    parser.add_argument('--subset', type=str, choices=['animals', 'traffic', 'all'])
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--background_weight', type=float)
    args = parser.parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
