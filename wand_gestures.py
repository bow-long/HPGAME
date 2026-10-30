"""
Wand gesture recognition by bitmap template matching (no ML at match time).

Flow
    wand detected -> smoothed (One Euro) -> stroke starts when it moves,
    stroke ends when the wand is lifted (lost) or held still
    -> trace squashed into a GRID x GRID 0/1 array -> matched against TEMPLATES

Add templates with:  python record_gesture.py "Circle"
and paste the copied line into TEMPLATES below.
"""

import math
import sysconfig
import threading
import time
from pathlib import Path

import cv2 as cv
import numpy as np

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "runs" / "detect" / "blue_led" / "weights" / "best.pt"

#MODEL_PATH = BASE_DIR / "best_openvino_model"

INSTALLED_MODEL_PATH = (
    Path(sysconfig.get_path("data")) / "share" / "hp-shape-tracing" / "best.pt"
)

WINDOW_NAME = "Wand Gesture Recognition"

# Camera (smaller frames = faster everything)
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Detection. Sizes must be multiples of 32.
CONFIDENCE_THRESHOLD = 0.25
FULL_INFERENCE_SIZE = 416     # used only to (re)acquire the wand
CROP_INFERENCE_SIZE = 224     # used while tracking
CROP_MIN_SIDE = 200           # px, crop is never smaller than this
CROP_SCALE = 10               # crop side = CROP_SCALE * wand box size
DEBUG_DRAW_CROP = False

# Cursor smoothing (One Euro filter)
# lower MIN_CUTOFF = smoother when slow; higher BETA = less lag when fast
SMOOTH_MIN_CUTOFF = 1.2
SMOOTH_BETA = 0.015

# Stroke start / auto finish
START_MOVE_PX = 25            # wand must move this far to start a stroke
ANCHOR_REFRESH_SECONDS = 1.0  # slow drift while idle never starts a stroke
MIN_TRACE_STEP = 3            # px between stored trace points
LOST_FRAMES_TO_FINISH = 8     # wand not detected for this many frames = pen lifted
DWELL_SECONDS = 0.6           # wand held still this long = pen lifted
DWELL_RADIUS_PX = 12

MIN_TRACE_POINTS = 5
MIN_TRACE_LENGTH = 60

# Template matching
GRID = 16                     # GRID*GRID must be divisible by 8
GRID_MARGIN = 1
GRID_STROKE = 1               # stroke thickness in cells
TOLERANCE = 3                 # dilation kernel (cells), forgives wobble
MATCH_THRESHOLD = 0.50
MIN_MARGIN = 0.03             # best must beat the runner-up gesture by this

# --------------------------------------------------
# Gesture dictionary
# Paste lines from record_gesture.py here.
#   "Name": ["hex", "hex"],     (several variants per gesture is best)
# --------------------------------------------------

TEMPLATES = {
    # "Vertical Line": ["..."],
    # "Horizontal Line": ["..."],
    # "Triangle": ["..."],
    # "Square": ["..."],
    # "Circle": ["..."],
        
    "circle": ["000003e00c301818300c200460064002400240024002400430041e7801c00000","000000f003080608180420042004400240042004200410081018082007c00000"],
     "v": ["0000000000004002400620042008101018300c20044003800180000000000000","0000000200020004000400080008000800100010402020202040184007800000"],
    "Hline": ["000000000000000000000000000060fe1f800000000000000000000000000000","0000000000000000000000000000001e7fe00000000000000000000000000000"],
    "heart": ["000000701c9016881308130810081008180808100820046004c0058002000000"],
    "u": ["0000000000000180034002400440082008201010200820084006000000000000"],
    "square": ["000000083ffc20042004200420042004200420042004100410041fc4103c0000", "000000007ffc40044004400440024002400240024002400231fe2e0000000000"],
    "Triangle": ["000002000400050004800840086008201010101020082008200460061ff80000"],
    "Vline": ["0000010001000100010001000100008000800080008000800080008000800000"],

        
}

# --------------------------------------------------
# Camera (background thread, always the newest frame)
# --------------------------------------------------


class ThreadedCamera:
    def __init__(self, index=CAMERA_INDEX):
        self.cap = cv.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

        self._frame = None
        self._lock = threading.Lock()
        self._fresh = threading.Event()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
            self._fresh.set()

    def read(self):
        if not self._fresh.wait(1.0):
            return None
        self._fresh.clear()
        with self._lock:
            return self._frame

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


# --------------------------------------------------
# Tracking (crop around last position, full frame only to reacquire)
# --------------------------------------------------


def load_model():
    model_path = next(
        (p for p in (MODEL_PATH, INSTALLED_MODEL_PATH) if p.is_file()),
        None,
    )
    if model_path is None:
        raise FileNotFoundError("YOLO wand model not found.")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("Install ultralytics first.")

    model = YOLO(str(model_path))

    # warm-up so the first real frame isn't slow
    model.predict(
        np.zeros((CROP_MIN_SIDE, CROP_MIN_SIDE, 3), np.uint8),
        imgsz=CROP_INFERENCE_SIZE,
        device="cpu",
        verbose=False,
    )
    return model


class WandTracker:
    def __init__(self, model):
        self.model = model
        self.box = None        # (cx, cy, w, h) in full-frame pixels
        self.last_crop = None  # (x0, y0, side) for debugging

    def _predict(self, image, imgsz):
        results = self.model.predict(
            image,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=imgsz,
            device="cpu",
            verbose=False,
        )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes
        best = None
        for conf, cls, xyxy in zip(
            boxes.conf.cpu().tolist(),
            boxes.cls.int().cpu().tolist(),
            boxes.xyxy.cpu().tolist(),
        ):
            if cls == 0 and (best is None or conf > best[0]):
                best = (conf, xyxy)

        if best is None:
            return None

        x1, y1, x2, y2 = best[1]
        return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1

    def detect(self, frame):
        """Return the wand center (x, y) in full-frame pixels, or None."""
        height, width = frame.shape[:2]
        self.last_crop = None

        if self.box is not None:
            cx, cy, bw, bh = self.box
            side = int(min(max(CROP_MIN_SIDE, CROP_SCALE * max(bw, bh)), width, height))
            x0 = int(np.clip(cx - side / 2, 0, width - side))
            y0 = int(np.clip(cy - side / 2, 0, height - side))
            self.last_crop = (x0, y0, side)

            crop = np.ascontiguousarray(frame[y0 : y0 + side, x0 : x0 + side])
            det = self._predict(crop, CROP_INFERENCE_SIZE)

            if det is not None:
                self.box = (det[0] + x0, det[1] + y0, det[2], det[3])
                return int(round(self.box[0])), int(round(self.box[1]))

        # lost (or first frame): search the whole frame
        det = self._predict(frame, FULL_INFERENCE_SIZE)
        self.box = det
        if det is None:
            return None
        return int(round(det[0])), int(round(det[1]))


# --------------------------------------------------
# Smoothing
# --------------------------------------------------


class OneEuro:
    """One Euro filter: steady when slow, responsive when fast."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self.x = None
        self.dx = None
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, point, t):
        p = np.asarray(point, dtype=np.float32)

        if self.x is None:
            self.x, self.dx, self.t = p, np.zeros(2, np.float32), t
            return self.x

        dt = max(t - self.t, 1e-3)
        raw_velocity = (p - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * raw_velocity + (1 - a_d) * self.dx

        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        a = self._alpha(cutoff, dt)
        self.x = a * p + (1 - a) * self.x
        self.t = t
        return self.x


# --------------------------------------------------
# Stroke detection (auto start / auto finish)
# --------------------------------------------------


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class StrokeDetector:
    def __init__(self):
        self.filter = OneEuro(SMOOTH_MIN_CUTOFF, SMOOTH_BETA)
        self.drawing = False
        self.trace = []
        self._lost = 0
        self._anchor = None
        self._anchor_time = 0.0
        self._still_anchor = None
        self._still_time = 0.0
        self._still_index = 0

    def start(self, point, now):
        self.drawing = True
        self.trace = [point]
        self._still_anchor = point
        self._still_time = now
        self._still_index = 1

    def finish(self, trim_to=None):
        trace = list(self.trace[:trim_to] if trim_to else self.trace)
        self.drawing = False
        self.trace = []
        self._anchor = None  # re-anchor on the next sample
        return trace

    def update(self, raw, now):
        """Feed the raw wand position (or None).

        Returns (smoothed_cursor | None, started: bool, finished_trace | None)
        """
        started = False
        finished = None

        if raw is None:
            self._lost += 1
            if self._lost >= LOST_FRAMES_TO_FINISH:
                if self.drawing:
                    finished = self.finish()
                self.filter.reset()
                self._anchor = None
            return None, started, finished

        self._lost = 0
        smooth = self.filter(raw, now)
        cur = (int(round(float(smooth[0]))), int(round(float(smooth[1]))))

        if not self.drawing:
            if self._anchor is None or now - self._anchor_time > ANCHOR_REFRESH_SECONDS:
                self._anchor = cur
                self._anchor_time = now
            elif _dist(cur, self._anchor) >= START_MOVE_PX:
                self.start(self._anchor, now)
                started = True

        if self.drawing:
            if _dist(cur, self.trace[-1]) >= MIN_TRACE_STEP:
                self.trace.append(cur)

            if _dist(cur, self._still_anchor) > DWELL_RADIUS_PX:
                self._still_anchor = cur
                self._still_time = now
                self._still_index = len(self.trace)
            elif now - self._still_time >= DWELL_SECONDS:
                finished = self.finish(trim_to=self._still_index)

        return cur, started, finished


# --------------------------------------------------
# Bitmap signature + template matching
# --------------------------------------------------

_NBYTES = GRID * GRID // 8
_KERNEL = np.ones((TOLERANCE, TOLERANCE), np.uint8)


def path_length(points):
    p = np.asarray(points, dtype=np.float32)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def is_valid_trace(points):
    return len(points) >= MIN_TRACE_POINTS and path_length(points) >= MIN_TRACE_LENGTH


def trace_to_grid(points):
    """Trace -> GRID x GRID array of 0/1 (1 = stroke touched the cell).

    Uniform scale so a vertical line stays a thin vertical line.
    """
    pts = np.asarray(points, dtype=np.float32)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = max(float((hi - lo).max()), 1.0)

    usable = GRID - 1 - 2 * GRID_MARGIN
    scaled = (pts - (lo + hi) / 2) / span * usable + (GRID - 1) / 2

    poly = np.round(scaled).astype(np.int32).reshape(-1, 1, 2)
    grid = np.zeros((GRID, GRID), np.uint8)
    cv.polylines(grid, [poly], False, 1, GRID_STROKE)
    return grid


def grid_to_int(grid):
    return int.from_bytes(np.packbits(grid.ravel()).tobytes(), "big")


def int_to_grid(value):
    raw = np.frombuffer(value.to_bytes(_NBYTES, "big"), np.uint8)
    return np.unpackbits(raw).reshape(GRID, GRID)


def grid_to_hex(grid):
    return f"{grid_to_int(grid):0{_NBYTES * 2}x}"


def grid_to_ascii(grid):
    return "\n".join("".join("#" if c else "." for c in row) for row in grid)


def _dilated_int(grid):
    return grid_to_int(cv.dilate(grid, _KERNEL))


def compile_templates(templates):
    compiled = []
    for name, sigs in templates.items():
        if isinstance(sigs, str):
            sigs = [sigs]
        for h in sigs:
            bits = int(h, 16)
            compiled.append(
                (name, bits, bits.bit_count(), _dilated_int(int_to_grid(bits)))
            )
    return compiled


COMPILED_TEMPLATES = compile_templates(TEMPLATES)


def _score(u_bits, u_dil, u_n, t_bits, t_n, t_dil):
    """Tolerant F-score using only AND + popcount."""
    if u_n == 0 or t_n == 0:
        return 0.0
    precision = (u_bits & t_dil).bit_count() / u_n   # my stroke lies on the template
    recall = (t_bits & u_dil).bit_count() / t_n      # the template is covered by my stroke
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def match_grid(grid):
    """Return (best_name | None, best_score, runner_up_score)."""
    if not COMPILED_TEMPLATES:
        return None, 0.0, 0.0

    u_bits = grid_to_int(grid)
    u_n = u_bits.bit_count()
    u_dil = _dilated_int(grid)

    best = {}
    for name, t_bits, t_n, t_dil in COMPILED_TEMPLATES:
        s = _score(u_bits, u_dil, u_n, t_bits, t_n, t_dil)
        if s > best.get(name, 0.0):
            best[name] = s

    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    name, score = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0.0
    return name, score, runner


def recognize_gesture(trace):
    """Return None (ignore: too short) or (label | None, score)."""
    if not is_valid_trace(trace):
        return None

    name, score, runner = match_grid(trace_to_grid(trace))

    if name is None or score < MATCH_THRESHOLD or score - runner < MIN_MARGIN:
        return None, score

    return name, score


# --------------------------------------------------
# Drawing
# --------------------------------------------------


def draw_wand_cursor(image, cursor):
    tip_x, tip_y = cursor
    handle = (tip_x - 24, tip_y + 30)

    cv.line(image, handle, cursor, (180, 80, 220), 5, cv.LINE_AA)
    cv.line(image, handle, cursor, (0, 215, 255), 2, cv.LINE_AA)
    cv.circle(image, cursor, 5, (255, 255, 255), -1, cv.LINE_AA)


def put_centered_text(image, text, y, scale=0.7, color=(255, 255, 255)):
    (width, _), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = max(10, (image.shape[1] - width) // 2)
    cv.putText(image, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv.LINE_AA)


# --------------------------------------------------
# Main
# --------------------------------------------------


def main():
    model = load_model()
    tracker = WandTracker(model)
    stroke = StrokeDetector()
    camera = ThreadedCamera()

    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)

    trail_mask = None   # persistent 1-channel canvas, segments added incrementally
    last_point = None
    result_text = None
    result_color = (0, 255, 0)
    fps = 0.0
    prev_time = time.monotonic()

    def handle_finished(trace):
        nonlocal result_text, result_color, trail_mask
        result = recognize_gesture(trace)
        if result is None:           # too short: accidental twitch
            trail_mask[:] = 0
            result_text = None
            return
        label, score = result
        if label:
            result_text, result_color = f"Detected: {label} ({score:.2f})", (0, 255, 0)
        else:
            result_text, result_color = f"Unrecognized ({score:.2f})", (0, 140, 255)

    try:
        while True:
            frame = camera.read()
            if frame is None:
                break

            frame = cv.flip(frame, 1)
            now = time.monotonic()

            if trail_mask is None or trail_mask.shape != frame.shape[:2]:
                trail_mask = np.zeros(frame.shape[:2], np.uint8)

            raw = tracker.detect(frame)
            cursor, started, finished = stroke.update(raw, now)

            if started:
                trail_mask[:] = 0
                last_point = stroke.trace[0]
                result_text = None

            if stroke.drawing and cursor is not None:
                if last_point is not None and last_point != cursor:
                    cv.line(trail_mask, last_point, cursor, 255, 3, cv.LINE_AA)
                last_point = cursor

            if finished is not None:
                handle_finished(finished)
                last_point = None

            display = frame
            display[trail_mask > 0] = (0, 255, 255)

            if cursor is not None:
                draw_wand_cursor(display, cursor)

            if DEBUG_DRAW_CROP and tracker.last_crop:
                x0, y0, side = tracker.last_crop
                cv.rectangle(display, (x0, y0), (x0 + side, y0 + side), (255, 0, 0), 1)

            if stroke.drawing:
                status = "DRAWING"
            elif cursor is None:
                status = "NO WAND"
            else:
                status = "READY"

            dt = max(now - prev_time, 1e-6)
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
            prev_time = now

            put_centered_text(display, f"Status: {status}   {fps:4.1f} fps", 30)
            put_centered_text(display, "Move to draw. Pause or hide the wand to finish.", 60, 0.55)
            put_centered_text(display, "SPACE=manual start/finish  C=clear  ESC=quit", 88, 0.55)

            if not COMPILED_TEMPLATES:
                put_centered_text(
                    display, "No templates - run record_gesture.py", 120, 0.6, (0, 0, 255)
                )

            if result_text:
                put_centered_text(display, result_text, 150, 0.9, result_color)

            cv.imshow(WINDOW_NAME, display)

            key = cv.waitKeyEx(1)
            if key >= 0:
                key &= 0xFF

            if key == 27:
                break

            if key in (ord("c"), ord("C")):
                if stroke.drawing:
                    stroke.finish()
                trail_mask[:] = 0
                last_point = None
                result_text = None

            elif key == 32:
                if stroke.drawing:
                    handle_finished(stroke.finish())
                    last_point = None
                elif cursor is not None:
                    stroke.start(cursor, now)
                    trail_mask[:] = 0
                    last_point = cursor
                    result_text = None

    finally:
        camera.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()