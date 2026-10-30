import copy
import math
import time

import cv2 as cv
import numpy as np

from wand_gestures import (
    load_model,
    WandTracker,
    StrokeDetector,
    recognize_gesture,
    ThreadedCamera,
    draw_wand_cursor,
)




WINDOW_NAME = "Dementor Rush"

PLAYER_IMAGE = "C:\\Users\\Admin\\Documents\\School\\Fall 2026\\HPGAME\\assests\\player.png"
DEMENTOR_IMAGE = "C:\\Users\\Admin\\Documents\\School\\Fall 2026\\HPGAME\\assests\\dementor.png"
BOSS_IMAGE = "C:\\Users\\Admin\\Documents\\School\\Fall 2026\\HPGAME\\assests\\dementor.png"

player_img = cv.imread(PLAYER_IMAGE, 0)
ghost_img = cv.imread(DEMENTOR_IMAGE, cv.IMREAD_UNCHANGED)
boss_img = cv.imread(BOSS_IMAGE, cv.IMREAD_UNCHANGED)


PLAYER_RADIUS = 40
PLAYER_MAX_HP = 10

GHOST_RADIUS = 28
GHOST_SPEED = 100

WAVE_DELAY = 2.0


# =====================================================
# GHOST
# =====================================================

class Ghost:
    def __init__(self, pos, symbols, speed, image=None, boss=False):
        self.pos = np.array(pos,dtype=np.float32)
        self.symbols = symbols
        self.speed = speed

        self.image = image
        self.boss = boss

        self.alive = True
        
    @property
    def current_symbol(self):
        if not self.symbols:
            return None
        return self.symbols[0]

    def update(self, center, dt):
        if not self.alive:
            return

        direction = center - self.pos
        dist = np.linalg.norm(direction)

        if dist > 1:
            direction = direction / dist
            self.pos += direction * self.speed * dt


# =====================================================
# WAVES
# =====================================================


SYMBOLS = {
    "circle": "○",
    "square": "□",
    "triangle": "△",
    "heart": "♥",
    "v": "\\/",
    "u": "/\\",
    "Hline": "—",
    "Vline": "|",
    "UpArrow": "↑",
}


WAVE_DATA = [
    [
        ((-80, 120), ["Hline"], 25),
        ((750, 250), ["v"], 30),
         ((-80, 400), ["u","v","Hline"], 30),
    ],
    [
        ((-80, 150), ["circle", "u"], 30),
        ((750, 120), ["Hline","Vline"], 30),
        ((320, -80), ["v","u","v"], 50),
    ],
    [
        ((-80, 240), ["square", "circle"], 25),
        ((750, 320), ["Hline"], 25),
        ((750, 120), ["square"], 25),
        ((320, 560), ["circle"], 25),
    ],
    [
        ((-80, 120), ["circle"], 40),
        ((750, 350), ["Hline", "Hline","v"], 15),
        ((320, -80), ["circle", "circle"], 15),
        ((320, 560), ["triangle",], 15),
    ],
    [
            ((-80, 120), ["circle", "Vline", "Vline"], 25),
            ((750, 350), ["Hline", "Hline","v"], 30),
            ((320, -80), ["v", "v"], 35),
            ((320, 560), ["square","square","triangle"], 25),
    ],
]


def create_wave(index):
    ghosts = []

    if index >= len(WAVE_DATA):
        return ghosts

    for pos, symbols, speed in WAVE_DATA[index]:
        ghosts.append(Ghost(pos, symbols, speed))

    return ghosts


# =====================================================
# GAME STATE
# =====================================================

class GameState:

    def __init__(self):
        self.restart()

    @property
    def current_wave(self):
        return self.wave_index + 1

    @property
    def final_wave_index(self):
        return len(WAVE_DATA) - 1


    @property
    def boss_wave(self):
        return self.wave_index == self.final_wave_index
    @property
    def enemies_remaining(self):
        return sum(
            1
            for g in self.ghosts
            if g.alive
        )
    def start_next_wave(self):

        self.wave_index += 1

        if self.wave_index >= len(WAVE_DATA):

            self.game_over = True
            self.victory = True
            return

        self.ghosts = create_wave(self.wave_index)

        self.wave_intro_start = time.monotonic()

        self.show_wave_intro = True
    def restart(self):

        self.hp = PLAYER_MAX_HP

        self.score = 0

        self.combo = 0
        self.multiplier = 1

        self.wave_index = -1

        self.ghosts = []

        self.game_over = False
        self.victory = False

        self.last_spell = ""

        self.wave_intro_duration = 3.0

        self.wave_intro_start = time.monotonic()

        self.show_wave_intro = True

        self.start_next_wave()


    def cast_spell(self, symbol):

        matched = 0

        for ghost in self.ghosts:

            if not ghost.alive:
                continue

            if not ghost.symbols:
                continue

            if ghost.symbols[0].lower() != symbol.lower():
                continue

            ghost.symbols.pop(0)

            

            matched += 1


            # Destroy the ghost when its gesture queue is empty.
            if not ghost.symbols:
                ghost.alive = False
                self.score += 100 * self.multiplier

        if matched:

            self.combo += matched

            self.multiplier = min(
                5,
                1 + self.combo // 10
            )

        else:

            self.combo = 0
            self.multiplier = 1

    def update(self, center, dt):

        if self.game_over:
            return

        if self.show_wave_intro:

            if (
                time.monotonic()
                - self.wave_intro_start
            ) >= self.wave_intro_duration:

                self.show_wave_intro = False

            return

        alive_count = 0

        for ghost in self.ghosts:

            if not ghost.alive:
                continue

            alive_count += 1

            ghost.update(center, dt)

            if np.linalg.norm(
                center - ghost.pos
            ) < PLAYER_RADIUS:

                ghost.alive = False

                self.hp -= 1

        if self.hp <= 0:

            self.game_over = True
            return

        if alive_count == 0:

            self.start_next_wave()


# =====================================================
# DRAWING
# =====================================================


def draw_png(frame, png, x, y, size):
    if png is None:
        return

    img = cv.resize(png, (size, size))

    # Convert grayscale images to BGR.
    if img.ndim == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    h, w = img.shape[:2]

    # Calculate image bounds, centered at (x, y).
    x1, y1 = x - w // 2, y - h // 2
    x2, y2 = x1 + w, y1 + h

    # Clip bounds to the frame.
    fx1, fy1 = max(0, x1), max(0, y1)
    fx2, fy2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    # Nothing visible if completely outside the frame.
    if fx1 >= fx2 or fy1 >= fy2:
        return

    # Corresponding region of the image.
    img = img[fy1-y1:fy2-y1, fx1-x1:fx2-x1]
    roi = frame[fy1:fy2, fx1:fx2]

    # Blend using alpha if the image has transparency.
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        foreground = img[:, :, :3].astype(np.float32)

        roi[:] = (
            roi.astype(np.float32) * (1 - alpha)
            + foreground * alpha
        ).astype(np.uint8)
    else:
        roi[:] = img[:, :, :3]

def symbol_string(symbols):
    return " ".join(
        SYMBOLS.get(x, x)
        for x in symbols
    )

def draw_player(frame, center):

    x, y = int(center[0]), int(center[1])

    draw_png(frame, player_img, x, y, PLAYER_RADIUS * 2)

    cv.putText(
        frame,
        "Wizard",
        (x - 34, y + 5),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )


def draw_ghost(frame, ghost):

    if not ghost.alive:
        return

    x, y = int(ghost.pos[0]), int(ghost.pos[1])

    draw_png(frame, ghost_img, x, y, GHOST_RADIUS * 2)

    if not ghost.boss:
        symbols = ghost.symbols[:5]
    else:
        symbols = ghost.symbols[:10]

    queue_text = symbol_string(symbols)

    

    if queue_text:
        cv.putText(
                frame,
                queue_text,
                (x - 50, y - 40),
                cv.FONT_HERSHEY_DUPLEX,
                0.7,
                (0,255,255),
                2,
                cv.LINE_AA,
            )

        


def draw_ui(frame, state):

    cv.rectangle(frame, (20, 20), (220, 50), (50, 50, 50), -1)

    hp_width = int((state.hp / PLAYER_MAX_HP) * 200)

    cv.rectangle(frame, (20, 20), (20 + hp_width, 50), (0, 255, 0), -1)

    cv.putText(
        frame,
        f"HP {state.hp}/{PLAYER_MAX_HP}",
        (25, 45),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )

    cv.putText(
        frame,
        f"Wave {state.wave_index + 1}",
        (20, 80),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv.putText(
        frame,
        f"Score {state.score}",
        (20, 110),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv.putText(
        frame,
        f"Spell: {state.last_spell}",
        (20, 140),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )


def draw_game_over(frame, state):

    overlay = frame.copy()

    cv.rectangle(
        overlay,
        (0, 0),
        (frame.shape[1], frame.shape[0]),
        (0, 0, 0),
        -1,
    )

    frame[:] = cv.addWeighted(overlay, 0.7, frame, 0.3, 0)

    text = "VICTORY!" if state.victory else "GAME OVER"

    cv.putText(
        frame,
        text,
        (180, 180),
        cv.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 255),
        4,
    )

    cv.putText(
        frame,
        f"Score: {state.score}",
        (240, 250),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    cv.putText(
        frame,
        "Press R to Restart",
        (180, 320),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    cv.putText(
        frame,
        "ESC to Quit",
        (250, 380),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
def draw_wave_intro(frame, state : GameState):
    title = (
    "FINAL BOSS"
    if state.boss_wave
    else f"WAVE {state.current_wave}"
    )

    cv.putText(
        frame,
        title,
        (120,120),
        cv.FONT_HERSHEY_DUPLEX,
        2,
        (0,255,255),
        4,
    )


# =====================================================
# MAIN
# =====================================================

def main():

    model = load_model()

    tracker = WandTracker(model)
    detector = StrokeDetector()

    camera = ThreadedCamera()

    state = GameState()

    trail_mask = None
    last_point = None

    last_time = time.monotonic()

    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
    cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    try:

        while True:

            frame = camera.read()

            if frame is None:
                break

            frame = cv.flip(frame, 1)

            now = time.monotonic()
            dt = now - last_time
            last_time = now

            if trail_mask is None:
                trail_mask = np.zeros(frame.shape[:2], np.uint8)

            raw = tracker.detect(frame)

            cursor, started, finished = detector.update(raw, now)

            if started:
                trail_mask[:] = 0
                last_point = None

            if detector.drawing and cursor is not None:

                if last_point is not None:
                    cv.line(
                        trail_mask,
                        last_point,
                        cursor,
                        255,
                        3,
                        cv.LINE_AA,
                    )

                last_point = cursor

            if finished is not None:

                result = recognize_gesture(finished)

                if result is not None:

                    label, score = result

                    if label:
                        state.cast_spell(label)

                trail_mask[:] = 0
                last_point = None

            center = np.array(
                [frame.shape[1] // 2, frame.shape[0] // 2],
                dtype=np.float32,
            )

            state.update(center, dt)

            frame[trail_mask > 0] = (0, 255, 255)

            draw_player(frame, center)

            for ghost in state.ghosts:
                draw_ghost(frame, ghost)

            if cursor is not None:
                draw_wand_cursor(frame, cursor)

            draw_ui(frame, state)

            if state.game_over:
                draw_game_over(frame, state)

            cv.namedWindow(WINDOW_NAME)

            cv.setWindowProperty(
                WINDOW_NAME,
                cv.WND_PROP_FULLSCREEN,
                cv.WINDOW_FULLSCREEN
            )

            cv.imshow(WINDOW_NAME, frame)

            key = cv.waitKey(1) & 0xFF

            if key == 27:
                break

            if key in (ord("r"), ord("R")):
                if state.game_over:
                    state.restart()
                    trail_mask[:] = 0

    finally:
        camera.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
