
import cv2
import mediapipe as mp
import numpy as np
import time

from pathlib import Path

from src.sleepiness_detection.sleepiness_engine import (
    SleepinessDetectionEngine
)

# ============================================================
# AI HUMAN EYE ANALYSIS SYSTEM
# ADVANCED SLEEPINESS & FATIGUE MODULE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "src" / "models" / "face_landmarker.task"

CAMERA_ID = 0

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# MediaPipe Face Landmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


# ============================================================
# DISTANCE CALCULATION
# ============================================================

def calculate_distance(point1, point2):

    return np.linalg.norm(
        np.array(point1) - np.array(point2)
    )


# ============================================================
# EYE ASPECT RATIO
# ============================================================

def calculate_ear(landmarks, eye_indices, width, height):

    points = []

    for index in eye_indices:

        landmark = landmarks[index]

        points.append((
            landmark.x * width,
            landmark.y * height
        ))

    vertical_1 = calculate_distance(points[1], points[5])
    vertical_2 = calculate_distance(points[2], points[4])

    horizontal = calculate_distance(points[0], points[3])

    if horizontal < 1e-6:
        return 0.0

    return (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)


# ============================================================
# MOUTH ASPECT RATIO
# ============================================================

def calculate_mar(landmarks, width, height):

    # MediaPipe Face Mesh mouth landmarks
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]

    left_corner = landmarks[78]
    right_corner = landmarks[308]

    vertical = calculate_distance(
        (upper_lip.x * width, upper_lip.y * height),
        (lower_lip.x * width, lower_lip.y * height)
    )

    horizontal = calculate_distance(
        (left_corner.x * width, left_corner.y * height),
        (right_corner.x * width, right_corner.y * height)
    )

    if horizontal < 1e-6:
        return 0.0

    return vertical / horizontal


# ============================================================
# DRAW HELPERS
# ============================================================

def draw_text(frame, text, position, color=(255, 255, 255),
              scale=0.62, thickness=2):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_eye_points(frame, landmarks, indices, width, height):

    for index in indices:

        landmark = landmarks[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(
            frame,
            (x, y),
            3,
            (0, 255, 0),
            -1
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    print("=" * 60)
    print("AI HUMAN EYE ANALYSIS SYSTEM")
    print("ADVANCED SLEEPINESS & FATIGUE DETECTION")
    print("=" * 60)

    if not MODEL_PATH.exists():

        print("ERROR: Face Landmarker model not found!")
        print("Expected:", MODEL_PATH)

        return

    camera = cv2.VideoCapture(CAMERA_ID)

    if not camera.isOpened():

        print("ERROR: Cannot open webcam!")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    options = FaceLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),

        running_mode=RunningMode.VIDEO,

        num_faces=1,

        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    engine = SleepinessDetectionEngine()

    start_time = time.perf_counter()
    previous_time = start_time

    fps = 0.0

    try:

        with FaceLandmarker.create_from_options(options) as landmarker:

            print("\nWebcam started!")
            print("Sleepiness & Fatigue Analysis running...")
            print("Press Q or ESC to exit.")

            while camera.isOpened():

                success, frame = camera.read()

                if not success:
                    print("Camera frame unavailable.")
                    break

                frame = cv2.flip(frame, 1)

                height, width, _ = frame.shape

                current_time = time.perf_counter()

                timestamp = int(
                    (current_time - start_time) * 1000
                )

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                rgb_frame = np.ascontiguousarray(rgb_frame)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp
                )

                # ========================================
                # FACE DETECTED
                # ========================================

                if result.face_landmarks:

                    landmarks = result.face_landmarks[0]

                    # Eye measurements
                    left_ear = calculate_ear(
                        landmarks,
                        LEFT_EYE,
                        width,
                        height
                    )

                    right_ear = calculate_ear(
                        landmarks,
                        RIGHT_EYE,
                        width,
                        height
                    )

                    average_ear = (
                        left_ear + right_ear
                    ) / 2.0

                    # Mouth measurement
                    mouth_ratio = calculate_mar(
                        landmarks,
                        width,
                        height
                    )

                    # Update eye engine
                    eye_status = engine.update_eye(
                        average_ear,
                        current_time
                    )

                    # Update PERCLOS
                    perclos = engine.update_perclos(
                        current_time,
                        eye_status == "CLOSED"
                    )

                    # Update yawning analysis
                    engine.update_yawning(
                        mouth_ratio,
                        current_time
                    )

                    # Final indicators
                    analysis = engine.get_indicators(
                        current_time,
                        perclos
                    )

                    # Draw landmarks
                    draw_eye_points(
                        frame,
                        landmarks,
                        LEFT_EYE,
                        width,
                        height
                    )

                    draw_eye_points(
                        frame,
                        landmarks,
                        RIGHT_EYE,
                        width,
                        height
                    )

                    # Face status
                    draw_text(
                        frame,
                        "FACE DETECTED",
                        (20, 35),
                        (0, 255, 0)
                    )

                    # Eye status
                    eye_color = (
                        (0, 0, 255)
                        if eye_status == "CLOSED"
                        else (0, 255, 0)
                    )

                    draw_text(
                        frame,
                        f"EYES: {eye_status}",
                        (20, 75),
                        eye_color
                    )

                    # Eye measurements
                    draw_text(
                        frame,
                        f"LEFT EAR: {left_ear:.3f}",
                        (20, 115),
                        (255, 255, 0)
                    )

                    draw_text(
                        frame,
                        f"RIGHT EAR: {right_ear:.3f}",
                        (20, 150),
                        (255, 255, 0)
                    )

                    draw_text(
                        frame,
                        f"AVERAGE EAR: {average_ear:.3f}",
                        (20, 185),
                        (255, 255, 0)
                    )

                    # Blink analysis
                    draw_text(
                        frame,
                        f"BLINK COUNT: {analysis['blink_count']}",
                        (20, 230),
                        (0, 255, 255)
                    )

                    draw_text(
                        frame,
                        f"BLINK RATE: {analysis['blink_rate']} / MIN",
                        (20, 265),
                        (0, 255, 255)
                    )

                    draw_text(
                        frame,
                        f"AVG BLINK DURATION: {analysis['average_blink_duration']:.2f}s",
                        (20, 300),
                        (0, 255, 255)
                    )

                    # PERCLOS
                    draw_text(
                        frame,
                        f"PERCLOS: {analysis['perclos']:.1f}%",
                        (20, 345),
                        (255, 200, 0)
                    )

                    draw_text(
                        frame,
                        f"EYE CLOSURE: {analysis['closure_duration']:.2f}s",
                        (20, 380),
                        (255, 200, 0)
                    )

                    # Yawning
                    draw_text(
                        frame,
                        f"MOUTH RATIO: {mouth_ratio:.3f}",
                        (20, 425),
                        (255, 150, 0)
                    )

                    draw_text(
                        frame,
                        f"YAWN COUNT: {analysis['yawn_count']}",
                        (20, 460),
                        (255, 150, 0)
                    )

                    draw_text(
                        frame,
                        f"YAWN DURATION: {analysis['yawn_duration']:.2f}s",
                        (20, 495),
                        (255, 150, 0)
                    )

                    # Sleepiness indicators
                    draw_text(
                        frame,
                        "VISUAL INDICATORS:",
                        (20, 545),
                        (255, 255, 255)
                    )

                    draw_text(
                        frame,
                        analysis["status"],
                        (20, 580),
                        analysis["color"],
                        0.65
                    )

                # ========================================
                # NO FACE DETECTED
                # ========================================

                else:

                    draw_text(
                        frame,
                        "NO FACE DETECTED",
                        (20, 40),
                        (0, 0, 255),
                        0.8
                    )

                    draw_text(
                        frame,
                        "Please face the camera",
                        (20, 80),
                        (255, 255, 255)
                    )

                # ========================================
                # FPS
                # ========================================

                elapsed = current_time - previous_time

                if elapsed > 0:

                    instant_fps = 1.0 / elapsed

                    fps = (
                        instant_fps
                        if fps == 0
                        else 0.9 * fps + 0.1 * instant_fps
                    )

                previous_time = current_time

                draw_text(
                    frame,
                    f"FPS: {fps:.1f}",
                    (width - 160, 35)
                )

                cv2.imshow(
                    "AI Human Analysis - Sleepiness & Fatigue",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    break

    except Exception as error:

        print("\nSYSTEM ERROR:", error)

    finally:

        camera.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 50)
        print("ANALYSIS STOPPED")
        print("Total Blinks:", engine.blink_count)
        print("Total Yawns:", engine.yawn_count)
        print("Webcam Released Successfully.")
        print("=" * 50)


if __name__ == "__main__":
    main()
