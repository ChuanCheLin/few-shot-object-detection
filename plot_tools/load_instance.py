import torch
import numpy as np
results = torch.load("/home/eric/few-shot-object-detection/checkpoints/coco/faster_rcnn/15classes/inference/instances_predictions.pth")
img_id = []
for result in results:
    img_id.append(result['image_id'])

print (img_id[0])
