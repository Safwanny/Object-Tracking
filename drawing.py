import cv2

def draw_tracks(frame, tracks):
    for t in tracks:
        x1, y1, x2, y2, track_id = map(int, t[:5])
        color = (0, 255, 0)  # Green for now
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Car ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
