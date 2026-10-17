from collections import deque
from pathlib import Path

import cv2 as cv


MODEL_PATH = Path(__file__).parent / "runs" / "detect" / "blue_led" / "weights" / "best.pt"
CONFIDENCE_THRESHOLD = 0.25
TRACK_LOSS_FRAMES = 12
TRAIL_LENGTH = 32


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH}. "
            "Run train_model.py after collecting labeled webcam frames."
        )

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is required. Install project dependencies with "
            "'pip install -r requirements.txt'."
        ) from error

    model = YOLO(str(MODEL_PATH))
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam.")

    previous_center = None
    selected_track_id = None
    missed_frames = 0
    trail = deque(maxlen=TRAIL_LENGTH)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam.")

            height, width = frame.shape[:2]
            screen_center = (width // 2, height // 2)
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )

            candidates = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                coordinates = boxes.xyxy.cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                track_ids = (
                    boxes.id.int().cpu().tolist()
                    if boxes.id is not None
                    else [None] * len(coordinates)
                )

                for box, confidence, class_id, track_id in zip(
                    coordinates, confidences, class_ids, track_ids
                ):
                    x1, y1, x2, y2 = (int(round(value)) for value in box)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    candidates.append(
                        (center, (x1, y1, x2, y2), confidence, class_id, track_id)
                    )

            target = None
            if candidates:
                if selected_track_id is not None:
                    target = next(
                        (
                            candidate
                            for candidate in candidates
                            if candidate[4] == selected_track_id
                        ),
                        None,
                    )
                if target is None and previous_center is not None:
                    target = min(
                        candidates,
                        key=lambda candidate: (
                            (candidate[0][0] - previous_center[0]) ** 2
                            + (candidate[0][1] - previous_center[1]) ** 2
                        ),
                    )
                if target is None:
                    target = max(candidates, key=lambda candidate: candidate[2])

            cv.drawMarker(
                frame,
                screen_center,
                (255, 255, 255),
                cv.MARKER_CROSS,
                20,
                2,
            )

            if target is not None:
                center, (x1, y1, x2, y2), confidence, class_id, track_id = target
                previous_center = center
                selected_track_id = track_id
                missed_frames = 0
                trail.append(center)

                class_name = model.names[int(class_id)]
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.circle(frame, center, 5, (0, 0, 255), -1)
                for start, end in zip(trail, list(trail)[1:]):
                    cv.line(frame, start, end, (0, 255, 255), 2)

                offset_x = center[0] - screen_center[0]
                offset_y = center[1] - screen_center[1]
                status = f"{class_name} {confidence:.0%}  X: {center[0]}  Y: {center[1]}"
                offset = f"Offset from center: ({offset_x:+d}, {offset_y:+d})"
                status_color = (0, 255, 0)
            else:
                missed_frames += 1
                status = "Searching for blue LED..."
                offset = ""
                status_color = (0, 165, 255)
                if missed_frames >= TRACK_LOSS_FRAMES:
                    previous_center = None
                    selected_track_id = None
                    trail.clear()

            cv.putText(
                frame,
                status,
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )
            if offset:
                cv.putText(
                    frame,
                    offset,
                    (10, 60),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
            cv.putText(
                frame,
                "ESC: quit",
                (10, height - 15),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv.imshow("Blue LED YOLO tracking", frame)

            if cv.waitKey(5) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
