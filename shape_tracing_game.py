from collections import deque
from pathlib import Path
import sysconfig
import time

import cv2 as cv
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "detect" / "blue_led" / "weights" / "best.pt"
INSTALLED_MODEL_PATH = (
    Path(sysconfig.get_path("data"))
    / "share"
    / "hp-shape-tracing"
    / "best.pt"
)
CONFIDENCE_THRESHOLD = 0.25
INFERENCE_SIZE = 640
SHAPES = (
    "Square",
    "Triangle",
    "Circle",
    "Deathly Hallows",
    "Star",
    "Golden Snitch",
)
TIME_PER_SHAPE_SECONDS = 20
TRACE_TOLERANCE_FRACTION = 0.1
CURSOR_TRAIL_LENGTH = 500
MIN_TRACE_STEP = 3
CLOSE_PATH_RADIUS = 12
CLOSE_PATH_FRACTION = 0.65
WINDOW_NAME = "Wizarding World Shape Tracing"


def sample_segment(start, end, samples=40):
    return np.linspace(start, end, samples, endpoint=False)


def create_polygon_path(vertices, closed=True, samples_per_segment=40):
    vertices = np.asarray(vertices, dtype=np.float32)
    segment_ends = np.roll(vertices, -1, axis=0) if closed else vertices[1:]
    segment_starts = vertices if closed else vertices[:-1]
    contour = np.vstack(
        [
            sample_segment(start, end, samples_per_segment)
            for start, end in zip(segment_starts, segment_ends)
        ]
    )
    display_points = np.rint(vertices).astype(np.int32).reshape((-1, 1, 2))
    return contour, display_points, closed


def create_target_shape(shape_name, frame_width, frame_height):
    center_x = frame_width / 2
    center_y = frame_height / 2
    size = min(frame_width, frame_height) * 0.34

    if shape_name == "Square":
        vertices = [
            (center_x - size, center_y - size),
            (center_x + size, center_y - size),
            (center_x + size, center_y + size),
            (center_x - size, center_y + size),
        ]
        paths = [create_polygon_path(vertices)]
    elif shape_name == "Triangle":
        vertices = [
            (center_x, center_y - size),
            (center_x + size * 0.9, center_y + size * 0.7),
            (center_x - size * 0.9, center_y + size * 0.7),
        ]
        paths = [create_polygon_path(vertices)]
    elif shape_name == "Circle":
        angles = np.linspace(0, 2 * np.pi, 240, endpoint=False)
        vertices = np.column_stack(
            (
                center_x + size * np.cos(angles),
                center_y + size * np.sin(angles),
            )
        )
        paths = [create_polygon_path(vertices, samples_per_segment=1)]
    elif shape_name == "Deathly Hallows":
        triangle = [
            (center_x, center_y - size),
            (center_x + size * 0.9, center_y + size * 0.7),
            (center_x - size * 0.9, center_y + size * 0.7),
        ]
        circle_angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        circle = np.column_stack(
            (
                center_x + size * 0.43 * np.cos(circle_angles),
                center_y + size * 0.25 + size * 0.43 * np.sin(circle_angles),
            )
        )
        wand_line = [
            (center_x, center_y - size),
            (center_x, center_y + size * 0.7),
        ]
        paths = [
            create_polygon_path(triangle),
            create_polygon_path(circle, samples_per_segment=1),
            create_polygon_path(wand_line, closed=False),
        ]
    elif shape_name == "Star":
        star_vertices = []
        for index in range(10):
            angle = -np.pi / 2 + index * np.pi / 5
            radius = size if index % 2 == 0 else size * 0.42
            star_vertices.append(
                (
                    center_x + radius * np.cos(angle),
                    center_y + radius * np.sin(angle),
                )
            )
        paths = [create_polygon_path(star_vertices)]
    elif shape_name == "Golden Snitch":
        wing_offset = size * 0.22
        left_wing = [
            (center_x - wing_offset, center_y - size * 0.08),
            (center_x - size * 0.48, center_y - size * 0.25),
            (center_x - size, center_y - size * 0.62),
            (center_x - size * 0.86, center_y - size * 0.08),
            (center_x - size, center_y + size * 0.42),
            (center_x - size * 0.48, center_y + size * 0.24),
            (center_x - wing_offset, center_y + size * 0.08),
        ]
        right_wing = [(2 * center_x - x, y) for x, y in left_wing]
        ball_angles = np.linspace(0, 2 * np.pi, 160, endpoint=False)
        ball = np.column_stack(
            (
                center_x + size * 0.24 * np.cos(ball_angles),
                center_y + size * 0.24 * np.sin(ball_angles),
            )
        )
        paths = [
            create_polygon_path(left_wing),
            create_polygon_path(right_wing),
            create_polygon_path(ball, samples_per_segment=1),
        ]
    else:
        raise ValueError(f"Unknown shape: {shape_name}")

    contour = np.vstack([path[0] for path in paths]).astype(np.float32)
    display_paths = [(path[1], path[2]) for path in paths]
    return contour, display_paths


def evaluate_trace(user_points, target_points, tolerance):
    if not user_points:
        return 0.0, float("inf"), 0.0

    target = np.asarray(target_points, dtype=np.float32)
    user_samples = []
    points = list(user_points)
    for start, end in zip(points, points[1:]):
        distance = float(np.linalg.norm(np.asarray(end) - start))
        sample_count = max(1, int(np.ceil(distance / 4)))
        user_samples.extend(sample_segment(start, end, sample_count))
    user_samples.append(np.asarray(points[-1], dtype=np.float32))
    user = np.asarray(user_samples, dtype=np.float32)
    distances = np.linalg.norm(user[:, None, :] - target[None, :, :], axis=2)
    closest_target_distance = distances.min(axis=1)
    closest_user_distance = distances.min(axis=0)
    average_error = float(closest_target_distance.mean())
    closeness = max(0.0, 1.0 - average_error / tolerance)
    coverage = float(np.mean(closest_user_distance <= tolerance))
    accuracy = 100.0 * closeness * coverage
    return accuracy, average_error, coverage


def trace_length(points):
    points = list(points)
    return sum(
        float(np.linalg.norm(np.asarray(end) - start))
        for start, end in zip(points, points[1:])
    )


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


def detect_wand(model, frame):
    results = model.predict(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=INFERENCE_SIZE,
        device="cpu",
        verbose=False,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes
    class_ids = boxes.cls.int().cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    coordinates = boxes.xyxy.cpu().tolist()
    wand_candidates = [
        (confidence, box)
        for confidence, class_id, box in zip(confidences, class_ids, coordinates)
        if class_id == 0
    ]
    if not wand_candidates:
        return None

    _, box = max(wand_candidates, key=lambda candidate: candidate[0])
    x1, y1, x2, y2 = box
    return (int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2)))


def draw_wand_cursor(image, cursor):
    tip_x, tip_y = cursor
    handle = (tip_x - 24, tip_y + 30)
    cv.line(image, handle, cursor, (180, 80, 220), 5, cv.LINE_AA)
    cv.line(image, handle, cursor, (0, 215, 255), 2, cv.LINE_AA)
    cv.circle(image, cursor, 5, (255, 255, 255), -1, cv.LINE_AA)


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
    model_path = next(
        (
            path
            for path in (MODEL_PATH, INSTALLED_MODEL_PATH)
            if path.is_file()
        ),
        None,
    )
    if model_path is None:
        raise FileNotFoundError(
            "Magic wand tracking model not found. Install the project with "
            "'pip install .' or restore runs/detect/blue_led/weights/best.pt."
        )

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is required. Install it with "
            "'pip install -r requirements.txt'."
        ) from error

    model = YOLO(str(model_path))
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
    trace_points = deque()
    live_trail = []
    results_by_shape = []
    final_results_started = 0.0
    round_notice = ""
    notice_until = 0.0

    try:
        while True:
            success, raw_frame = cap.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam.")

            frame = cv.flip(raw_frame, 1)
            height, width = frame.shape[:2]
            display = frame.copy()
            now = time.monotonic()
            cursor = detect_wand(model, frame)

            if phase == "playing":
                if target_contour is None or target_display is None:
                    raise RuntimeError("The active shape was not initialized.")
                round_end_reason = None
                if cursor is not None:
                    live_trail.append(cursor)
                    if len(live_trail) > CURSOR_TRAIL_LENGTH:
                        del live_trail[:-CURSOR_TRAIL_LENGTH]
                    if (
                        not trace_points
                        or np.linalg.norm(
                            np.asarray(cursor) - trace_points[-1]
                        )
                        >= MIN_TRACE_STEP
                    ):
                        trace_points.append(cursor)
                        if len(trace_points) > 1:
                            outline_length = sum(
                                cv.arcLength(path, closed)
                                for path, closed in target_display
                            )
                            returned_to_start = (
                                np.linalg.norm(
                                    np.asarray(cursor) - trace_points[0]
                                )
                                <= CLOSE_PATH_RADIUS
                                and trace_length(trace_points)
                                >= outline_length * CLOSE_PATH_FRACTION
                            )
                            if returned_to_start:
                                round_end_reason = "Closed the shape"

                for path, closed in target_display:
                    cv.polylines(
                        display,
                        [path],
                        closed,
                        (0, 215, 255),
                        2,
                        cv.LINE_AA,
                    )
                if trace_points:
                    cv.circle(
                        display,
                        trace_points[0],
                        9,
                        (255, 0, 255),
                        2,
                        cv.LINE_AA,
                    )
                if len(live_trail) > 1:
                    for start, end in zip(live_trail, live_trail[1:]):
                        if np.linalg.norm(np.asarray(end) - start) < 80:
                            cv.line(display, start, end, (0, 255, 255), 3, cv.LINE_AA)
                if cursor is not None:
                    draw_wand_cursor(display, cursor)

                remaining = max(
                    0.0,
                    TIME_PER_SHAPE_SECONDS - (now - phase_started),
                )
                put_centered_text(
                    display,
                    f"{SHAPES[shape_index]}  |  {remaining:04.1f}s",
                    35,
                    0.8,
                    (0, 215, 255),
                )
                cv.putText(
                    display,
                    "Trace the golden outline with your magic wand",
                    (15, height - 40),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv.LINE_AA,
                )
                cv.putText(
                    display,
                    "Close at your start to finish early",
                    (15, height - 16),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv.LINE_AA,
                )

                if remaining <= 0 and round_end_reason is None:
                    round_end_reason = "Time expired"

                if round_end_reason is not None:
                    tolerance = min(width, height) * TRACE_TOLERANCE_FRACTION
                    accuracy, _, coverage = evaluate_trace(
                        trace_points,
                        target_contour,
                        tolerance,
                    )
                    score = accuracy
                    results_by_shape.append(
                        (
                            SHAPES[shape_index],
                            score,
                            accuracy,
                            coverage,
                            round_end_reason,
                        )
                    )
                    round_notice = (
                        f"{SHAPES[shape_index]} ended: {round_end_reason} "
                        f"| {grade_for_score(score)} ({score:.0f}%)"
                    )
                    notice_until = now + 2.0
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
                    (0, 215, 255),
                )
                put_centered_text(
                    display,
                    f"Press SPACE to start its {TIME_PER_SHAPE_SECONDS}-second round",
                    height // 2 + 25,
                    0.7,
                )
                put_centered_text(
                    display,
                    "Follow the golden outline with your magic wand",
                    height // 2 + 65,
                    0.6,
                )
                if now < notice_until:
                    put_centered_text(
                        display,
                        round_notice,
                        height // 2 + 115,
                        0.58,
                        (0, 165, 255),
                    )
            elif phase == "results":
                put_centered_text(
                    display, "GAME COMPLETE", 70, 1.1, (0, 215, 255)
                )
                for index, result in enumerate(results_by_shape):
                    name, score, accuracy = result[:3]
                    coverage, reason = result[3:]
                    grade = grade_for_score(score)
                    message = (
                        f"{name}: {grade}  |  Score {score:.0f}%  |  "
                        f"Trace {accuracy:.0f}%  |  Covered {coverage:.0%}  |  "
                        f"{reason}"
                    )
                    put_centered_text(
                        display,
                        message,
                        145 + index * 42,
                        0.5,
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
                    draw_wand_cursor(display, cursor)
                if phase == "ready":
                    put_centered_text(
                        display,
                        f"WAND {'READY' if cursor is not None else 'NOT FOUND'}",
                        35,
                        0.6,
                        (0, 215, 255) if cursor is not None else (0, 165, 255),
                    )

            cv.imshow(WINDOW_NAME, display)
            key = cv.waitKeyEx(1)
            if key >= 0:
                key &= 0xFF

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
