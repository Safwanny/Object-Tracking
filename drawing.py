import cv2

# how long a brand-new ID should stay green (in frames)
NEW_HIGHLIGHT_FRAMES = 5

def draw_tracks(frame, tracks, unified_ids, seen_ids=None,
                color_new=(0, 255, 0), color_known=(255, 0, 0)):
    """
    Draw one unified ID per track with a short green 'new ID' window, then blue.
    - frame: BGR image
    - tracks: list of [x1, y1, x2, y2, unified_id]
    - unified_ids: dict (tid -> unified_id), kept for future flexibility
    - seen_ids: optional set managed by caller (not required)
    """
    # persistent per-run state (age per ID in frames)
    if not hasattr(draw_tracks, "_id_age"):
        draw_tracks._id_age = {}  # uid -> frames seen

    id_age = draw_tracks._id_age

    for x1, y1, x2, y2, uid in tracks:
        uid = int(uid)

        # increment age (starts at 0 on first sight)
        if uid not in id_age:
            id_age[uid] = 0
        else:
            id_age[uid] += 1

        # color logic: green for the first NEW_HIGHLIGHT_FRAMES, then blue
        if id_age[uid] < NEW_HIGHLIGHT_FRAMES:
            color = color_new
        else:
            color = color_known

        # draw box + single ID label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"ID: {uid}", (int(x1), max(0, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame