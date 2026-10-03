"""
YOLOv3 builders for the three tasks.

Everything here is algorithm-specific:
    - which .cfg (backbone + neck + head layout)
    - how [yolo] anchors are parsed
    - which detection layer (loss / target assignment) is used
    - how many box attributes the head regresses

Everything else (Darknet parsing, training, evaluation, demo) is common.
"""


from detectors.yolov3.image_yolo_layer import YoloLayer as ImageYoloLayer


def _floats(text):
    return [float(v) for v in text.split(",")]


def _mask(block):
    return [int(v) for v in block["mask"].split(",")]


def image_factory(block, hyperparams):
    """Image YOLOv3: (w, h) anchors, global 9-anchor matching (legacy)."""
    v = _floats(block["anchors"])
    all_anchors = [(v[i], v[i + 1]) for i in range(0, len(v), 2)]
    mask = _mask(block)
    return ImageYoloLayer(
        [all_anchors[i] for i in mask],
        int(block["classes"]),
        int(hyperparams["height"]),
        all_anchors=all_anchors,
        anchor_mask=mask,
    )
