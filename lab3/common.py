import json
import os

import numpy as np
import PIL.Image
import torch

from detection_and_metrics import (
    calc_auc,
    get_cls_model,
    get_detection_model,
    get_detections,
    nms,
)


def calc_detector_auc(img_dir, gt_path, apply_nms=False):
    classifier_model = get_cls_model()
    classifier_model.load_state_dict(
        torch.load(
            "classifier_model.pt",
            weights_only=True,
            map_location="cpu",
        ),
    )
    detection_model = get_detection_model(classifier_model)
    images_detection = read_for_detection(img_dir, gt_path)
    images_detection_no_answer = {}
    images_detection_only_bboxes = {}

    for img_name, data in images_detection.items():
        images_detection_no_answer[img_name] = data[0]
        images_detection_only_bboxes[img_name] = data[1]

    pred = get_detections(detection_model, images_detection_no_answer)
    if apply_nms:
        pred = nms(pred)

    return calc_auc(pred, images_detection_only_bboxes)


def read_for_detection(img_dir, gt_path):
    with open(gt_path) as fp:
        raw_data = json.load(fp)

    data = {}
    for file_name, bboxes in raw_data.items():
        file_path = os.path.join(img_dir, file_name)
        image = np.array(PIL.Image.open(file_path))
        image = image.astype(np.float32) / 255
        data[file_name] = (image, bboxes)
    return data
