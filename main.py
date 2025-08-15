import os
import time
import argparse
import cv2
import yaml
from ultralytics import YOLO
from drawing import draw_tracks  # Only call this for drawing

DEFAULT_MODEL = "yolov8s.pt"
DEFAULT_CLASSES = [0]  # person, car, bottle
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_TRACKER = "tracker.yaml"
WINDOW_TITLE = "Object Tracking: YOLOv8 + BoT-SORT + ReID"

def ensure_tracker(tracker_path: str) -> str:
    try:
        with open(tracker_path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"⚠️ Could not read {tracker_path}: {e}")
        return tracker_path

    need_reid = bool(cfg.get("with_reid", False))
    weights_path = cfg.get("reid_weights", "")

    if need_reid and (not weights_path or not os.path.exists(weights_path)):
        print("⚠️ ReID weights not found, disabling ReID.")
        cfg["with_reid"] = False
        tmp = os.path.splitext(tracker_path)[0] + "_noreid.yaml"
        try:
            with open(tmp, "w") as f:
                yaml.safe_dump(cfg, f)
            return tmp
        except Exception as e:
            print(f"⚠️ Could not write fallback config: {e}")
    return tracker_path

def parse_args():
    ap = argparse.ArgumentParser(description="YOLOv8 + BoT-SORT tracking with optional ReID fallback")
    ap.add_argument("--source", default="0", help="Webcam index or path to video")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="YOLOv8 model")
    ap.add_argument("--classes", type=int, nargs="*", default=DEFAULT_CLASSES)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--iou", type=float, default=DEFAULT_IOU)
    ap.add_argument("--tracker", default=DEFAULT_TRACKER)
    return ap.parse_args()

def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    tracker_path = ensure_tracker(args.tracker)

    print(f"🔹 Loading YOLOv8 model: {args.model}")
    model = YOLO(args.model)

    # Handle webcam or video stream

    source = 1

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Failed to open video source: {source}")
        return

        print("▶️  Starting tracking. Press 'q' to quit.")
        last_t = time.time()
        fps = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Stream ended.")
                break

            results = model.track(
                source=frame,
                persist=True,
                tracker=tracker_path,
                classes=args.classes or None,
                conf=args.conf,
                iou=args.iou,
                verbose=False
            )

            if results and len(results) > 0:
                draw_tracks(frame, results[0])  # Draw all visual output here

            # Show the processed frame
            cv2.imshow(WINDOW_TITLE, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("👋 Tracking ended.")
    else:
        # If it's a video/image path, let Ultralytics handle streaming
        model.track(
            source=source,
            persist=True,
            tracker=tracker_path,
            classes=args.classes or None,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
            show=True
        )
        print("👋 Done.")

if __name__ == "__main__":
    main()
