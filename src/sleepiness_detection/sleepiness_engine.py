
from collections import deque
import time
import numpy as np


class SleepinessDetectionEngine:

    def __init__(
        self,
        eye_threshold=0.21,
        blink_consecutive_frames=3,
        long_closure_seconds=1.5,
        yawn_threshold=0.45,
        yawn_min_duration=1.0,
        perclos_window=30
    ):

        self.eye_threshold = eye_threshold
        self.blink_consecutive_frames = blink_consecutive_frames

        self.long_closure_seconds = long_closure_seconds
        self.yawn_threshold = yawn_threshold
        self.yawn_min_duration = yawn_min_duration
        self.perclos_window = perclos_window

        # Eye state
        self.eye_closed = False
        self.closed_frames = 0
        self.closure_start = None

        # Blink tracking
        self.blink_count = 0
        self.blink_timestamps = deque()
        self.blink_durations = deque(maxlen=30)

        # PERCLOS tracking
        self.eye_history = deque()

        # Yawning tracking
        self.yawning = False
        self.yawn_start = None
        self.yawn_count = 0
        self.yawn_timestamps = deque()

        self.current_closure_duration = 0.0
        self.current_yawn_duration = 0.0

    # --------------------------------------------------
    # Eye analysis
    # --------------------------------------------------

    def update_eye(self, ear, current_time):

        eye_status = (
            "CLOSED" if ear < self.eye_threshold else "OPEN"
        )

        if eye_status == "CLOSED":

            self.closed_frames += 1

            if not self.eye_closed:

                self.eye_closed = True
                self.closure_start = current_time

            self.current_closure_duration = (
                current_time - self.closure_start
            )

        else:

            if self.eye_closed:

                duration = (
                    current_time - self.closure_start
                )

                if self.closed_frames >= self.blink_consecutive_frames:

                    self.blink_count += 1
                    self.blink_timestamps.append(current_time)
                    self.blink_durations.append(duration)

            self.closed_frames = 0
            self.eye_closed = False
            self.closure_start = None
            self.current_closure_duration = 0.0

        return eye_status

    # --------------------------------------------------
    # Time-based PERCLOS
    # --------------------------------------------------

    def update_perclos(self, current_time, eye_closed):

        self.eye_history.append(
            (current_time, eye_closed)
        )

        cutoff = current_time - self.perclos_window

        # Preserve one sample before cutoff if available
        while (
            len(self.eye_history) > 1
            and self.eye_history[1][0] <= cutoff
        ):

            self.eye_history.popleft()

        observed_duration = 0.0
        closed_duration = 0.0

        samples = list(self.eye_history)

        for index in range(len(samples)):

            start_time, closed = samples[index]

            if index + 1 < len(samples):
                end_time = samples[index + 1][0]
            else:
                end_time = current_time

            interval_start = max(start_time, cutoff)
            interval_end = min(end_time, current_time)

            duration = max(
                0.0,
                interval_end - interval_start
            )

            # Ignore unusually large gaps caused by lost frames
            if duration > 0.5:
                continue

            observed_duration += duration

            if closed:
                closed_duration += duration

        if observed_duration <= 0:
            return 0.0

        return (
            closed_duration / observed_duration
        ) * 100.0

    # --------------------------------------------------
    # Blink rate
    # --------------------------------------------------

    def calculate_blink_rate(self, current_time):

        window_seconds = 60

        while (
            self.blink_timestamps
            and current_time - self.blink_timestamps[0] > window_seconds
        ):

            self.blink_timestamps.popleft()

        return len(self.blink_timestamps)

    # --------------------------------------------------
    # Yawning detection
    # --------------------------------------------------

    def update_yawning(self, mouth_ratio, current_time):

        mouth_open = mouth_ratio >= self.yawn_threshold

        if mouth_open:

            if not self.yawning:

                self.yawning = True
                self.yawn_start = current_time

            self.current_yawn_duration = (
                current_time - self.yawn_start
            )

        else:

            if self.yawning and self.yawn_start is not None:

                duration = current_time - self.yawn_start

                if duration >= self.yawn_min_duration:

                    self.yawn_count += 1
                    self.yawn_timestamps.append(current_time)

            self.yawning = False
            self.yawn_start = None
            self.current_yawn_duration = 0.0

        return mouth_open

    # --------------------------------------------------
    # Sleepiness indicators
    # --------------------------------------------------

    def get_indicators(self, current_time, perclos):

        blink_rate = self.calculate_blink_rate(current_time)

        long_closure = (
            self.current_closure_duration
            >= self.long_closure_seconds
        )

        recent_yawns = sum(
            1 for timestamp in self.yawn_timestamps
            if current_time - timestamp <= 300
        )

        while (
            self.yawn_timestamps
            and current_time - self.yawn_timestamps[0] > 300
        ):
            self.yawn_timestamps.popleft()

        indicators = []

        if long_closure:
            indicators.append("PROLONGED EYE CLOSURE")

        if perclos >= 20:
            indicators.append("ELEVATED PERCLOS")

        if recent_yawns >= 2:
            indicators.append("REPEATED YAWNING")

        if len(indicators) >= 2:
            status = "MULTIPLE FATIGUE INDICATORS"
            color = (0, 0, 255)

        elif len(indicators) == 1:
            status = "FATIGUE INDICATOR OBSERVED"
            color = (0, 165, 255)

        else:
            status = "NO ELEVATED VISUAL INDICATORS"
            color = (0, 255, 0)

        average_blink_duration = (
            float(np.mean(self.blink_durations))
            if self.blink_durations
            else 0.0
        )

        return {
            "blink_count": self.blink_count,
            "blink_rate": blink_rate,
            "average_blink_duration": average_blink_duration,
            "perclos": perclos,
            "closure_duration": self.current_closure_duration,
            "yawn_count": self.yawn_count,
            "recent_yawns": recent_yawns,
            "yawn_duration": self.current_yawn_duration,
            "indicators": indicators,
            "status": status,
            "color": color
        }
