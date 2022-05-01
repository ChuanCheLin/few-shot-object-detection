import os
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
import torch
#data registry
DATASET_ROOT = "/home/eric/mmdetection/data/VOCdevkit/datasets/CocoDataset_15/"
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
TEST_PATH = os.path.join(DATASET_ROOT, 'test')
TEST_JSON = os.path.join(ANN_ROOT, 'instances_test.json') 

PREDEFINED_SPLITS_DATASET = {
    "test_tea": (TEST_PATH, TEST_JSON),
}

DATASET_CATEGORIES = [
    {"name": "algal", "id": 1, "isthing": 1, "color": [220, 20, 60]},
    {"name": "blister", "id": 2, "isthing": 1, "color": [219, 142, 185]},
    {"name": "brownblight", "id": 3, "isthing": 1, "color": [220, 20, 60]},
    {"name": "caloptilia", "id": 4, "isthing": 1, "color": [219, 142, 185]},
    {"name": "flushworm", "id": 5, "isthing": 1, "color": [220, 20, 60]},
    {"name": "formosa", "id": 6, "isthing": 1, "color": [219, 142, 185]},
    {"name": "fungi_early", "id": 7, "isthing": 1, "color": [220, 20, 60]},
    {"name": "miner", "id": 8, "isthing": 1, "color": [219, 142, 185]},
    {"name": "mosquito_early", "id": 9, "isthing": 1, "color": [220, 20, 60]},
    {"name": "mosquito_late", "id": 10, "isthing": 1, "color": [219, 142, 185]},
    {"name": "moth", "id": 11, "isthing": 1, "color": [220, 20, 60]},
    {"name": "other", "id": 12, "isthing": 1, "color": [219, 142, 185]},
    {"name": "roller", "id": 13, "isthing": 1, "color": [220, 20, 60]},
    {"name": "tetrany", "id": 14, "isthing": 1, "color": [219, 142, 185]},
    {"name": "thrips", "id": 15, "isthing": 1, "color": [220, 20, 60]},
    {"name": "tortrix", "id": 16, "isthing": 1, "color": [219, 142, 185]},
]

def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key, 
                                   metadate=get_dataset_instances_meta(), 
                                   json_file=json_file, 
                                   image_root=image_root)


def get_dataset_instances_meta():
    """
    purpose: get metadata of dataset from DATASET_CATEGORIES
    return: dict[metadata]
    """
    thing_ids = [k["id"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_dataset_instances(name, metadate, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file, 
                                  image_root=image_root, 
                                  evaluator_type="coco", 
                                  **metadate)

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

register_dataset()

cfg = get_cfg()
cfg.merge_from_file("/home/eric/few-shot-object-detection/checkpoints/coco/faster_rcnn/15classes/config.yaml")   # load values from config yaml #change

model = build_model(cfg)  # returns a torch.nn.Module

DetectionCheckpointer(model).load("/home/eric/few-shot-object-detection/checkpoints/coco/faster_rcnn/15classes/model_final.pth")

#data_loader = build_detection_test_loader(cfg, "test_tea")

trainer = Trainer(cfg)

model.eval()
with torch.no_grad():
    outputs = model(data_loader)