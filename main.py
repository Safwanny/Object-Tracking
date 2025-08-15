import os
import time
import cv2
from ultralytics import YOLO
from drawing import draw_tracks

# ========================
# User-tunable parameters
# ========================

# Camera index:
#   0 = built-in webcam
#   1/2 = Camo/iPhone virtual camera
CAMERA_INDEX: int = 1

# Model weights
MODEL_WEIGHTS: str = "yolov8s.pt"

# Classes to track (COCO IDs): 0=person, 2=car, 58=potted plant
SELECTED_CLASS_IDS = [2]

# Detector thresholds and image size
CONF_THRESHOLD: float = 0.65
IOU_THRESHOLD: float = 0.60
IMAGE_SIZE: int = 640

# Tracker config: prefer local file, else use Ultralytics' built-in
LOCAL_TRACKER_YAML: str = "tracker.yaml"
TRACKER_CONFIG: str = LOCAL_TRACKER_YAML if os.path.exists(LOCAL_TRACKER_YAML) else "bytetrack.yaml"

# Verbose debug prints from Ultralytics
VERBOSE: bool = False


def main() -> None:
    """Run YOLOv8 + ByteTrack. Only a single, unified ID is shown (drawing handled in drawing.py)."""
    print("Starting Object Tracking... Press 'q' to quit.")

    # 1) Load detector
    model = YOLO(MODEL_WEIGHTS)

    # 2) Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera index {CAMERA_INDEX}.")
        return

    # 3) ID mapping:
    #    tid_to_unified maps raw tracker ID (TID) -> displayed unified ID (int)
    tid_to_unified: dict[int, int] = {}
    next_unified_id: int = 1

    # Optional: keep track of which unified IDs we've already drawn once
    # (your drawing.py can also manage “seen” coloring internally if you prefer)
    seen_unified_ids: set[int] = set()

    # FPS meter
    last_t = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from camera.")
            break

        # 4) Run tracking on the current frame
        results = model.track(
            source=frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            classes=SELECTED_CLASS_IDS,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMAGE_SIZE,
            verbose=VERBOSE,
        )

        # 5) Build list for drawing: [[x1, y1, x2, y2, unified_id], ...]
        tracks_for_drawing: list[list[int]] = []

        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().tolist()
            tids = boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(xyxy)

            for (x1, y1, x2, y2), tid in zip(xyxy, tids):
                if tid is None:
                    continue

                # Ensure integer coords
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)

                # Map tracker’s TID → unified ID (single ID display)
                if tid not in tid_to_unified:
                    tid_to_unified[tid] = next_unified_id
                    next_unified_id += 1

                unified_id = tid_to_unified[tid]
                tracks_for_drawing.append([x1i, y1i, x2i, y2i, unified_id])

        # 6) Draw overlays (NOTE: pass tid_to_unified so drawing has the IDs)
        frame = draw_tracks(frame, tracks_for_drawing, tid_to_unified)

        # 7) FPS HUD
        now = time.time()
        dt = now - last_t
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else (1.0 / dt)
        last_t = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Object Tracking (single ID)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()