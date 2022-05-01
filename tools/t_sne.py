import os
import cv2
import logging
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.utils.visualizer import Visualizer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.engine import launch

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup
from detectron2.evaluation import COCOEvaluator, verify_results
from fsdet.config import get_cfg, set_global_cfg

dataset_root = "/home/eric/mmdetection/data/VOCdevkit/datasets/"
set_num = "set1/" #need change
split_num = "split2/" #need change
# Dataset Root
DATASET_ROOT = dataset_root + set_num + "comparison"
DATASET_ROOT_few = dataset_root + set_num +  "all_60"

ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
ANN_ROOT_few = os.path.join(DATASET_ROOT_few, 'annotations')

TRAINVALTEST_PATH = os.path.join(DATASET_ROOT_few, 'trainvaltest')
TEST_PATH = os.path.join(DATASET_ROOT, 'test')

TRAINVALTEST_JSON = os.path.join(ANN_ROOT_few, 'instances_trainvaltest.json')    
TEST_JSON = os.path.join(ANN_ROOT, 'instances_test.json')  

# 注册数据集和元数据
def plain_register_dataset():
    DatasetCatalog.register("train_tea", lambda: load_coco_json(TRAINVALTEST_JSON, TRAINVALTEST_PATH, "train_tea"))
    MetadataCatalog.get("train_tea").set(thing_classes=['brownblight', 'algal', 'blister', 'sunburn', 'fungi_early', 'roller', 'moth', 'tortrix', 'flushworm', 'caloptilia', 'mosquito_early', 'mosquito_late', 'miner', 'thrips', 'tetrany', 'formosa', 'other'],
                                                    json_file=TRAINVALTEST_JSON,
                                                    image_root=TRAINVALTEST_PATH)

    DatasetCatalog.register("test_tea", lambda: load_coco_json(TEST_JSON, TEST_PATH, "test_tea"))
    MetadataCatalog.get("test_tea").set(thing_classes=['brownblight', 'algal', 'blister', 'sunburn', 'fungi_early', 'roller', 'moth', 'tortrix', 'flushworm', 'caloptilia', 'mosquito_early', 'mosquito_late', 'miner', 'thrips', 'tetrany', 'formosa', 'other'],
                                                json_file=TEST_JSON,
                                                image_root=TEST_PATH)


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=output_folder)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg() # 拷贝default config副本
    #args.config_file = "configs/eric/cascade_rcnn_R_50_FPN.yaml"
    cfg.merge_from_file(args.config_file)   # 从config file 覆盖配置
    cfg.merge_from_list(args.opts)          # 从CLI参数 覆盖配置
    # 更改配置参数
    cfg.DATASETS.TRAIN = ("train_tea",)
    cfg.DATASETS.TEST = ("test_tea"),

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17 # 类别数
    cfg.SOLVER.IMS_PER_BATCH = 3  # batch_size=2; iters_in_one_epoch = dataset_imgs/batch_size  
    ITERS_IN_ONE_EPOCH = int(960 / cfg.SOLVER.IMS_PER_BATCH)
    cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 12) - 1 # epochs # long: 48
    # cfg.SOLVER.BASE_LR = 0.002
    # cfg.SOLVER.MOMENTUM = 0.9
    # cfg.SOLVER.WEIGHT_DECAY = 0.0001
    # cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    cfg.SOLVER.GAMMA = 0.2
    cfg.SOLVER.STEPS = (ITERS_IN_ONE_EPOCH*5, ITERS_IN_ONE_EPOCH*8, ITERS_IN_ONE_EPOCH*10)
    cfg.TEST.EVAL_PERIOD = 3*ITERS_IN_ONE_EPOCH #每三個epoch test 一次
    #cfg.MODEL.WEIGHTS = "checkpoints/coco/faster_rcnn/" + set_num + split_num + "few_60/combined/model_reset_combine.pth"
    cfg.OUTPUT_DIR = "checkpoints/coco/faster_rcnn/" + set_num + split_num + "eval" #change
    # cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    # cfg.SOLVER.WARMUP_ITERS = 1000
    # cfg.SOLVER.WARMUP_METHOD = "linear"
    # cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    print(cfg)

    # 注册数据集
    plain_register_dataset()

    if args.eval_only:
        model = Trainer.build_model(cfg)

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test_tsne(cfg, model)

        #np.save("/home/eric/few-shot-object-detection/temp.npy", features_copy)

        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )