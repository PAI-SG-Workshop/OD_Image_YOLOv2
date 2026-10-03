"""
YOLOv2 builders for the three tasks (plan A: original Darknet yolov2.cfg).

Architecture (algorithm-specific, this folder):
    Darknet-19 backbone + passthrough (reorg) neck, ONE detection scale
    (stride 32, 19 x 19 grid at 608), anchors in cfg/.

Detection layers: reused unchanged from detectors/yolov3 (maximum code reuse).
    image   -> Image YOLO layer   (anchor matching over the 5 v2 priors)
    complex -> Complex-YOLO layer (E-RPN im/re angle)      - paper: YOLOv2-based
    yolo3d  -> YOLO3D layer       (direct yaw, z, h)        - paper: YOLOv2-based

Note (image): the original Darknet [region] loss (softmax classes, IoU-rescored
objectness) is replaced by the shared YOLO layer, so only the architecture
differs between Image v2 and Image v3.
"""

from pathlib import Path

from common.darknet import Darknet
from detectors.yolov3.build import image_factory

CFG_DIR = Path(__file__).resolve().parent / "cfg"

SPECS = {
    "image":   ("image_yolov2.cfg",   image_factory,   5),
}


def build_model(task, num_classes):
    cfg_name, factory, num_box_attr = SPECS[task]
    return Darknet(CFG_DIR / cfg_name, factory, num_classes, num_box_attr)
