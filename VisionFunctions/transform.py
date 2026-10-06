from __future__ import annotations

import numpy as np
import torch
from cv2 import INTER_AREA, boundingRect, drawContours, resize
from torchvision.models import mobilenet_v2

try:
    from torchvision.models import MobileNet_V2_Weights as MobileNetV2_Weights
except ImportError:
    try:
        from torchvision.models import MobileNetV2_Weights
    except ImportError:
        MobileNetV2_Weights = None


# Raspberry Pi 4 has no CUDA GPU, so this always runs on CPU. Pin the thread
# count to the Pi's 4 cores — left unset, PyTorch sometimes over-subscribes
# threads and thrashes rather than speeding things up.
DEVICE = torch.device("cpu")
torch.set_num_threads(4)

weights = MobileNetV2_Weights.DEFAULT if MobileNetV2_Weights is not None else None
model = mobilenet_v2(weights=weights)
model.classifier = torch.nn.Identity()
model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
model.to(DEVICE)
model.eval()

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)


def _extract_object_roi(image, contour):
    x, y, w, h = boundingRect(contour)
    if w <= 0 or h <= 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)

    object_roi = image[y:y + h, x:x + w]
    if object_roi.size == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    contour_in_roi = contour - np.array([x, y], dtype=contour.dtype)
    drawContours(mask, [contour_in_roi], -1, 255, -1)

    masked_roi = np.where(mask[..., None] != 0, object_roi, 0).astype(np.uint8)

    return resize(masked_roi, (224, 224), interpolation=INTER_AREA)


def _preprocess_batch(rois):
    # Build ONE tensor for every object in the frame instead of one tensor
    # (and one model() call) per object. This is the main fix: model-call
    # overhead, not raw FLOPs, is what was costing ~2s/frame when objects
    # were run through sequentially.
    batch = np.stack(rois, axis=0)          # (N, H, W, 3), BGR uint8
    batch = batch[..., ::-1]                # BGR -> RGB
    batch = np.ascontiguousarray(batch)
    tensor = torch.from_numpy(batch).to(DEVICE).float().div_(255.0)
    tensor = tensor.permute(0, 3, 1, 2)     # NHWC -> NCHW
    return (tensor - MEAN) / STD


def transform_objects(image, detected_objects):
    rois = []
    for obj in detected_objects:
        contour = obj.get("contour")
        if contour is None:
            continue
        rois.append(_extract_object_roi(image, contour))

    if not rois:
        return []

    batch = _preprocess_batch(rois)

    with torch.inference_mode():
        features = model(batch).flatten(1).cpu().numpy().astype(np.float32)

    return [features[i] for i in range(features.shape[0])]