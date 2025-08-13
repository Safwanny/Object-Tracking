from ultralytics import YOLO
import cv2

def main():
    # Load YOLOv8 small model
    model = YOLO("yolov8s.pt")

    # Only potted plant (COCO class ID 58)
    allowed_classes = [0]

    # Track seen IDs
    seen_ids = set()

    # Start webcam (phone camera via Camo)
    cap = cv2.VideoCapture(1)  # Change index if needed

    print("✅ Starting Object Tracking... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run tracking
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.5
        )

        annotated_frame = frame.copy()

        for r in results:
            if not hasattr(r, "boxes"):
                continue

            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in allowed_classes:
                    continue

                track_id = int(box.id[0]) if box.id is not None else -1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Assign colors based on new/known ID
                if track_id not in seen_ids:
                    color = (0, 255, 0)  # Green for new
                    seen_ids.add(track_id)
                else:
                    color = (255, 0, 0)  # Blue for known

                label = f"ID:{track_id} {model.names[cls_id]} {conf:.2f}"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Tracking - Potted Plants", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
