import torch

# 🟢 2026 ULTIMATE NMS IMPORT FIX
try:
    # Try current standard location
    from ultralytics.utils.ops import non_max_suppression
except ImportError:
    try:
        # Try alternate data utils location
        from ultralytics.data.utils import non_max_suppression
    except ImportError:
        try:
            # Try importing the whole ops module
            import ultralytics.utils.ops as ops
            non_max_suppression = ops.non_max_suppression
        except (ImportError, AttributeError):
            # Final fallback: Look inside the YOLO model's internal predictor
            from ultralytics.models.yolo.detect.predict import DetectionPredictor
            import ultralytics.utils.nms as nms_internal
            non_max_suppression = nms_internal.non_max_suppression

class LegoNMS:
    def __init__(self, conf_thres=0.3, iou_thres=0.45):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def apply(self, prediction):
        return non_max_suppression(
            prediction, 
            conf_thres=self.conf_thres, 
            iou_thres=self.iou_thres
        )