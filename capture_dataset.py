from datetime import datetime
from pathlib import Path
import random
import shutil

import cv2 as cv
import numpy as np


DATASET_DIR = Path(__file__).with_name("blue_led_dataset")
STAGING_DIR = DATASET_DIR / ".staging"
LOWER_BLUE = np.array([90, 100, 170])
UPPER_BLUE = np.array([140, 255, 255])
MIN_COMPONENT_AREA = 3
BOX_PADDING = 4
CAPTURE_EVERY_N_FRAMES = 3
VALIDATION_FRACTION = 0.2


def find_blue_led_boxes(frame):
    height, width = frame.shape[:2]
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    count, _, stats, _ = cv.connectedComponentsWithStats(mask)

    boxes = []
    for index in range(1, count):
        x, y, box_width, box_height, area = stats[index]
        if area < MIN_COMPONENT_AREA:
            continue

        x1 = max(0, int(x) - BOX_PADDING)
        y1 = max(0, int(y) - BOX_PADDING)
        x2 = min(width, int(x + box_width) + BOX_PADDING)
        y2 = min(height, int(y + box_height) + BOX_PADDING)
        boxes.append((x1, y1, x2, y2))

    return mask, boxes


def write_yolo_labels(label_path, boxes, image_width, image_height):
    rows = []
    for x1, y1, x2, y2 in boxes:
        center_x = (x1 + x2) / (2 * image_width)
        center_y = (y1 + y2) / (2 * image_height)
        box_width = (x2 - x1) / image_width
        box_height = (y2 - y1) / image_height
        rows.append(
            f"0 {center_x:.6f} {center_y:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )
    label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def finish_session(session_dir, session_frames):
    if not session_frames:
        shutil.rmtree(session_dir)
        print("No bright-blue LED frames were captured.")
        return

    random.shuffle(session_frames)
    validation_count = (
        max(1, round(len(session_frames) * VALIDATION_FRACTION))
        if len(session_frames) > 1
        else 0
    )
    validation_frames = set(session_frames[:validation_count])

    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    for filename in session_frames:
        split = "val" if filename in validation_frames else "train"
        shutil.move(
            str(session_dir / "images" / filename),
            DATASET_DIR / "images" / split / filename,
        )
        shutil.move(
            str(session_dir / "labels" / f"{Path(filename).stem}.txt"),
            DATASET_DIR / "labels" / split / f"{Path(filename).stem}.txt",
        )

    shutil.rmtree(session_dir)
    print(
        f"Saved {len(session_frames)} labeled frames "
        f"({len(session_frames) - validation_count} train, "
        f"{validation_count} validation) in {DATASET_DIR}"
    )


def write_dataset_config():
    dataset_path = str(DATASET_DIR.resolve()).replace("\\", "/")
    config = (
        f"path: '{dataset_path}'\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: blue_led\n"
    )
    (DATASET_DIR / "data.yaml").write_text(config, encoding="utf-8")


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam.")

    recording = False
    session_dir = None
    session_id = None
    session_frames = []
    frame_number = 0
    total_saved = 0
    r_was_down = False

    try:
        while True:
            success, frame = cap.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam.")

            height, width = frame.shape[:2]
            mask, boxes = find_blue_led_boxes(frame)
            frame_number += 1

            if (
                recording
                and boxes
                and frame_number % CAPTURE_EVERY_N_FRAMES == 0
            ):
                filename = f"{session_id}_frame_{len(session_frames):06d}.jpg"
                image_path = session_dir / "images" / filename
                label_path = session_dir / "labels" / f"{Path(filename).stem}.txt"
                if not cv.imwrite(str(image_path), frame):
                    raise RuntimeError(f"Could not save captured frame: {image_path}")
                write_yolo_labels(label_path, boxes, width, height)
                session_frames.append(filename)
                total_saved += 1

            display = frame.copy()
            for x1, y1, x2, y2 in boxes:
                cv.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            recording_text = "RECORDING" if recording else "READY"
            recording_color = (0, 0, 255) if recording else (0, 255, 0)
            light_text = "Blue light found" if boxes else "No bright blue light"
            cv.putText(
                display,
                f"{recording_text} | {light_text} | Saved this session: "
                f"{len(session_frames)}",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                recording_color,
                2,
            )
            cv.putText(
                display,
                "R: start/stop recording | ESC: quit",
                (10, 60),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv.imshow("Blue LED dataset capture", display)
            cv.imshow("Blue LED filter", mask)

            key = cv.waitKey(1) & 0xFF
            if key == 27:
                break
            r_is_down = key in (ord("r"), ord("R"))
            if r_is_down and not r_was_down:
                recording = not recording
                if recording:
                    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    session_dir = STAGING_DIR / session_id
                    (session_dir / "images").mkdir(parents=True, exist_ok=False)
                    (session_dir / "labels").mkdir()
                    session_frames = []
                    frame_number = 0
                    print("Recording started. Only detected bright-blue frames are saved.")
                else:
                    finish_session(session_dir, session_frames)
                    session_dir = None
                    session_frames = []
                    if total_saved:
                        write_dataset_config()
            r_was_down = r_is_down
    finally:
        if recording and session_dir is not None:
            finish_session(session_dir, session_frames)
            if total_saved:
                write_dataset_config()
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
