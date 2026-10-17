from pathlib import Path
import time

import cv2 as cv
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "detect" / "blue_led" / "weights" / "best.pt"
CONFIDENCE_THRESHOLD = 0.25
INFERENCE_SIZE = 640
SHAPES = ("Square", "Triangle", "Circle")
TIME_PER_SHAPE_SECONDS = 20
TRACE_TOLERANCE_FRACTION = 0.08
CURSOR_TRAIL_LENGTH = 500
WINDOW_NAME = "Blue LED Shape Tracing"


def sample_segment(start, end, samples=40):
    return np.linspace(start, end, samples, endpoint=False)


def create_target_shape(shape_name, frame_width, frame_height):
    center_x = frame_width / 2
    center_y = frame_height / 2
    size = min(frame_width, frame_height) * 0.34

    if shape_name == "Square":
        corners = [
            (center_x - size, center_y - size),
            (center_x + size, center_y - size),
            (center_x + size, center_y + size),
            (center_x - size, center_y + size),
        ]
        vertices = np.array(corners + [corners[0]], dtype=np.float32)
        segments = [
            sample_segment(start, end)
            for start, end in zip(vertices, vertices[1:])
        ]
        contour = np.vstack(segments)
        display_points = np.rint(vertices).astype(np.int32).reshape((-1, 1, 2))
    elif shape_name == "Triangle":
        vertices = np.array(
            [
                (center_x, center_y - size),
                (center_x + size * 0.9, center_y + size * 0.7),
                (center_x - size * 0.9, center_y + size * 0.7),
                (center_x, center_y - size),
            ],
            dtype=np.float32,
        )
        contour = np.vstack(
            [
                sample_segment(start, end)
                for start, end in zip(vertices, vertices[1:])
            ]
        )
        display_points = np.rint(vertices).astype(np.int32).reshape((-1, 1, 2))
    elif shape_name == "Circle":
        angles = np.linspace(0, 2 * np.pi, 240, endpoint=False)
        contour = np.column_stack(
            (
                center_x + size * np.cos(angles),
                center_y + size * np.sin(angles),
            )
        )
        display_points = np.rint(contour).astype(np.int32).reshape((-1, 1, 2))
    else:
        raise ValueError(f"Unknown shape: {shape_name}")

    return contour.astype(np.float32), display_points


def evaluate_trace(user_points, target_points, tolerance):
    if not user_points:
        return 0.0, float("inf"), 0.0

    user = np.asarray(user_points, dtype=np.float32)
    target = np.asarray(target_points, dtype=np.float32)
    distances = np.linalg.norm(user[:, None, :] - target[None, :, :], axis=2)
    closest_target_distance = distances.min(axis=1)
    closest_user_distance = distances.min(axis=0)
    average_error = float(closest_target_distance.mean())
    closeness = max(0.0, 1.0 - average_error / tolerance)
    coverage = float(np.mean(closest_user_distance <= tolerance))
    accuracy = 100.0 * closeness * coverage
    return accuracy, average_error, coverage


def grade_for_score(score):
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def detect_led(model, frame):
    results = model.predict(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=INFERENCE_SIZE,
        device=0,
        verbose=False,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes
    class_ids = boxes.cls.int().cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    coordinates = boxes.xyxy.cpu().tolist()
    led_candidates = [
        (confidence, box)
        for confidence, class_id, box in zip(confidences, class_ids, coordinates)
        if class_id == 0
    ]
    if not led_candidates:
        return None

    _, box = max(led_candidates, key=lambda candidate: candidate[0])
    x1, y1, x2, y2 = box
    return (int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2)))


def put_centered_text(image, text, y, scale=0.8, color=(255, 255, 255)):
    (text_width, _), _ = cv.getTextSize(
        text, cv.FONT_HERSHEY_SIMPLEX, scale, 2
    )
    x = max(10, (image.shape[1] - text_width) // 2)
    cv.putText(
        image,
        text,
        (x, y),
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv.LINE_AA,
    )


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained blue LED model not found: {MODEL_PATH}"
        )

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is required. Install it with "
            "'pip install -r requirements.txt'."
        ) from error

    model = YOLO(str(MODEL_PATH))
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam.")

    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
    cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    shape_index = 0
    phase = "ready"
    phase_started = 0.0
    target_contour = None
    target_display = None
    trace_points = []
    live_trail = []
    results_by_shape = []
    final_results_started = 0.0

    try:
        while True:
            success, raw_frame = cap.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam.")

            frame = cv.flip(raw_frame, 1)
            height, width = frame.shape[:2]
            display = frame.copy()
            now = time.monotonic()
            cursor = detect_led(model, frame)

            if phase == "playing":
                if cursor is not None:
                    trace_points.append(cursor)
                    live_trail.append(cursor)
                    live_trail = live_trail[-CURSOR_TRAIL_LENGTH:]

                cv.polylines(
                    display,
                    [target_display],
                    True,
                    (255, 180, 0),
                    2,
                    cv.LINE_AA,
                )
                if len(live_trail) > 1:
                    for start, end in zip(live_trail, live_trail[1:]):
                        if np.linalg.norm(np.asarray(end) - start) < 80:
                            cv.line(display, start, end, (0, 255, 255), 3, cv.LINE_AA)
                if cursor is not None:
                    cv.circle(display, cursor, 8, (0, 0, 255), -1, cv.LINE_AA)

                remaining = max(
                    0.0,
                    TIME_PER_SHAPE_SECONDS - (now - phase_started),
                )
                put_centered_text(
                    display,
                    f"{SHAPES[shape_index]}  |  {remaining:04.1f}s",
                    35,
                    0.8,
                    (0, 255, 255),
                )
                cv.putText(
                    display,
                    "Trace the blue outline with the LED",
                    (15, height - 18),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv.LINE_AA,
                )

                if remaining <= 0:
                    tolerance = min(width, height) * TRACE_TOLERANCE_FRACTION
                    accuracy, average_error, coverage = evaluate_trace(
                        trace_points,
                        target_contour,
                        tolerance,
                    )
                    results_by_shape.append(
                        (SHAPES[shape_index], accuracy, average_error, coverage)
                    )
                    shape_index += 1
                    live_trail.clear()
                    trace_points.clear()
                    if shape_index >= len(SHAPES):
                        phase = "results"
                        final_results_started = now
                    else:
                        phase = "ready"
            elif phase == "ready":
                put_centered_text(
                    display,
                    f"Next shape: {SHAPES[shape_index]}",
                    height // 2 - 25,
                    1.0,
                    (0, 255, 255),
                )
                put_centered_text(
                    display,
                    f"Press SPACE to start its {TIME_PER_SHAPE_SECONDS}-second round",
                    height // 2 + 25,
                    0.7,
                )
                put_centered_text(
                    display,
                    "Follow the blue outline with your blue LED",
                    height // 2 + 65,
                    0.6,
                )
            elif phase == "results":
                put_centered_text(display, "GAME OVER", 70, 1.1, (0, 255, 255))
                for index, (name, score, error, coverage) in enumerate(
                    results_by_shape
                ):
                    grade = grade_for_score(score)
                    error_text = (
                        f"{error:.1f}px" if np.isfinite(error) else "No trace"
                    )
                    message = (
                        f"{name}: {grade}  |  Accuracy {score:.2f}%  |  "
                        f"Line covered {coverage:.0%}  |  Error {error_text}"
                    )
                    put_centered_text(
                        display,
                        message,
                        145 + index * 42,
                        0.62,
                    )
                mean_score = float(np.mean([result[1] for result in results_by_shape]))
                put_centered_text(
                    display,
                    f"Final grade: {grade_for_score(mean_score)} "
                    f"({mean_score:.0f}%)",
                    height - 65,
                    0.9,
                    (0, 255, 0),
                )
                if now - final_results_started > 2:
                    put_centered_text(
                        display,
                        "Press R to play again or ESC to quit",
                        height - 25,
                        0.55,
                    )

            if phase != "playing":
                if cursor is not None:
                    cv.circle(display, cursor, 7, (0, 0, 255), -1, cv.LINE_AA)
                if phase == "ready":
                    put_centered_text(
                        display,
                        f"LED {'FOUND' if cursor is not None else 'NOT FOUND'}",
                        35,
                        0.6,
                        (0, 255, 0) if cursor is not None else (0, 165, 255),
                    )

            cv.imshow(WINDOW_NAME, display)
            key = cv.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break
            if phase == "ready" and key == 32:
                target_contour, target_display = create_target_shape(
                    SHAPES[shape_index],
                    width,
                    height,
                )
                trace_points.clear()
                live_trail.clear()
                phase_started = time.monotonic()
                phase = "playing"
            elif phase == "results" and key in (ord("r"), ord("R")):
                shape_index = 0
                phase = "ready"
                results_by_shape.clear()
    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
