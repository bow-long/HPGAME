"""
Record hand-drawn gesture templates and copy the dictionary line to the clipboard.

    python record_gesture.py "Circle"            # track the wand
    python record_gesture.py "Circle" --mouse    # draw with the mouse (hold left button)

Draw the shape, then lift the wand / release the mouse (or hold still).
The line   "Circle": ["<hex>"],   is copied automatically and printed.
Paste it into TEMPLATES in wand_gestures.py. To add another variant of
the same gesture, add its hex string to that gesture's list.

Keys:  ENTER = copy again   C = clear   ESC = quit
"""

import argparse
import time

import cv2 as cv
import numpy as np

import wand_gestures as wg

PREVIEW_SCALE = 16


def copy_to_clipboard(text):
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        root.destroy()
        return True
    except Exception:
        return False


def show_preview(grid):
    big = cv.resize(
        grid * 255,
        (wg.GRID * PREVIEW_SCALE, wg.GRID * PREVIEW_SCALE),
        interpolation=cv.INTER_NEAREST,
    )
    big = cv.cvtColor(big, cv.COLOR_GRAY2BGR)
    for i in range(1, wg.GRID):
        p = i * PREVIEW_SCALE
        cv.line(big, (p, 0), (p, big.shape[0]), (60, 60, 60), 1)
        cv.line(big, (0, p), (big.shape[1], p), (60, 60, 60), 1)
    cv.imshow("Signature preview", big)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help='gesture name, e.g. "Circle"')
    ap.add_argument("--mouse", action="store_true", help="draw with the mouse")
    args = ap.parse_args()

    mouse = {"pos": None, "down": False}
    camera = tracker = None

    win = f"Record: {args.name}"
    cv.namedWindow(win, cv.WINDOW_NORMAL)

    if args.mouse:
        def on_mouse(event, x, y, flags, param):
            mouse["pos"] = (x, y)
            if event == cv.EVENT_LBUTTONDOWN:
                mouse["down"] = True
            elif event == cv.EVENT_LBUTTONUP:
                mouse["down"] = False

        cv.setMouseCallback(win, on_mouse)
    else:
        tracker = wg.WandTracker(wg.load_model())
        camera = wg.ThreadedCamera()

    stroke = wg.StrokeDetector()
    last_trace = []
    line = None
    message = ""

    def handle_finished(trace):
        nonlocal last_trace, line, message
        if not wg.is_valid_trace(trace):
            message = "Too short - try again"
            print(message)
            return

        last_trace = trace
        grid = wg.trace_to_grid(trace)
        line = f'    "{args.name}": ["{wg.grid_to_hex(grid)}"],'
        show_preview(grid)

        name, score, _ = wg.match_grid(grid)
        match = f"{name} {score:.2f}" if name else "no templates yet"
        copied = copy_to_clipboard(line)
        message = ("Copied. " if copied else "Clipboard unavailable. ") + f"Currently matches: {match}"

        print("\n" + wg.grid_to_ascii(grid))
        print(line)
        print(message)

    try:
        while True:
            now = time.monotonic()

            if camera is not None:
                frame = camera.read()
                if frame is None:
                    break
                frame = cv.flip(frame, 1)
                raw = tracker.detect(frame)
            else:
                frame = np.zeros((480, 640, 3), np.uint8)
                raw = mouse["pos"] if mouse["down"] else None

            cursor, started, finished = stroke.update(raw, now)
            if started:
                message = ""
            if finished is not None:
                handle_finished(finished)

            shown = stroke.trace if stroke.drawing else last_trace
            if len(shown) > 1:
                cv.polylines(
                    frame,
                    [np.asarray(shown, np.int32).reshape(-1, 1, 2)],
                    False,
                    (0, 255, 255),
                    3,
                    cv.LINE_AA,
                )
            if cursor is not None:
                wg.draw_wand_cursor(frame, cursor)

            status = "DRAWING" if stroke.drawing else ("READY" if cursor else "NO WAND")
            wg.put_centered_text(frame, f"{args.name} | {status}", 30)
            wg.put_centered_text(frame, "Draw, then lift/hold still. ENTER=copy C=clear ESC=quit", 58, 0.5)
            if message:
                wg.put_centered_text(frame, message, 90, 0.6, (0, 255, 0))

            cv.imshow(win, frame)

            key = cv.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key in (ord("c"), ord("C")):
                if stroke.drawing:
                    stroke.finish()
                last_trace, line, message = [], None, ""
            elif key in (13, 10) and line:
                message = "Copied." if copy_to_clipboard(line) else "Clipboard unavailable - see terminal."
                print(line)
    finally:
        if camera is not None:
            camera.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()