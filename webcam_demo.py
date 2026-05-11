import argparse
import io
import os
import queue
import re
import shutil
import subprocess
import threading
import tkinter as tk
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageTk


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "zombienet_best.keras"
DATASET_TRAIN_DIR = PROJECT_DIR / "dataset" / "train"
IMAGE_SIZE = (224, 224)
DEFAULT_CAPTURE_SIZE = (640, 480)
DEFAULT_CAMERA_FPS = 30
DEFAULT_INFERENCE_FPS = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="ZombieNet live webcam demo for macOS using ffmpeg + Tkinter."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index from `ffmpeg -f avfoundation -list_devices true -i \"\"`."
    )
    parser.add_argument(
        "--capture-fps",
        type=int,
        default=DEFAULT_CAMERA_FPS,
        help="Camera capture FPS. FaceTime HD typically supports 15 or 30."
    )
    parser.add_argument(
        "--infer-fps",
        type=int,
        default=DEFAULT_INFERENCE_FPS,
        help="How many frames per second to send to the model."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_CAPTURE_SIZE[0],
        help="Capture width."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_CAPTURE_SIZE[1],
        help="Capture height."
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List AVFoundation camera devices and exit."
    )
    return parser.parse_args()


def resolve_ffmpeg():
    ffmpeg_binary = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if not os.path.exists(ffmpeg_binary):
        raise FileNotFoundError(
            "ffmpeg was not found. Install it with `brew install ffmpeg` or adjust the path."
        )
    return ffmpeg_binary


def unique_preserve_order(items):
    unique_items = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        unique_items.append(item)
        seen.add(item)
    return unique_items


def list_avfoundation_cameras(ffmpeg_binary):
    probe_command = [ffmpeg_binary, "-f", "avfoundation", "-list_devices", "true", "-i", ""]
    probe_result = subprocess.run(
        probe_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False
    )

    cameras = []
    in_video_section = False
    for line in probe_result.stderr.splitlines():
        if "AVFoundation video devices" in line:
            in_video_section = True
            continue
        if "AVFoundation audio devices" in line:
            in_video_section = False
            continue
        if not in_video_section:
            continue

        match = re.search(r"\[(\d+)\]\s+(.+)$", line)
        if match:
            index = int(match.group(1))
            name = match.group(2).strip()
            cameras.append((index, name))

    return cameras


def choose_camera(cameras, requested_index=None):
    if requested_index is not None:
        for index, name in cameras:
            if index == requested_index:
                return index, name
        raise ValueError(f"Camera index {requested_index} was not found.")

    for index, name in cameras:
        if "Capture screen" not in name:
            return index, name

    raise RuntimeError("No physical camera was found in the AVFoundation device list.")


def load_class_names():
    if not DATASET_TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_TRAIN_DIR}"
        )

    class_names = sorted(
        entry.name
        for entry in DATASET_TRAIN_DIR.iterdir()
        if entry.is_dir()
    )
    if not class_names:
        raise RuntimeError("No class folders were found under dataset/train.")
    return class_names


def ordered_display_labels(class_names):
    preferred_order = ["human", "zombie", "vampire", "skeleton"]
    ordered = [label for label in preferred_order if label in class_names]
    ordered.extend(label for label in class_names if label not in ordered)
    return ordered


def preprocess_frame(frame_image):
    resized = frame_image.resize(IMAGE_SIZE)
    frame_array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(frame_array, axis=0)


def annotate_frame(frame_image, probabilities, class_names, display_labels):
    palette = {
        "human": (46, 204, 113),
        "zombie": (231, 76, 60),
        "vampire": (155, 89, 182),
        "skeleton": (243, 156, 18),
    }

    top_index = int(np.argmax(probabilities))
    top_label = class_names[top_index]
    top_score = float(probabilities[top_index]) * 100.0

    score_lookup = {
        class_name: float(probabilities[class_names.index(class_name)]) * 100.0
        for class_name in class_names
    }

    draw = ImageDraw.Draw(frame_image)
    draw.rounded_rectangle((14, 14, 390, 184), radius=16, fill=(14, 18, 24))
    draw.text((28, 28), "ZombieNet Live CCTV Scan", fill=(255, 255, 255))
    draw.text(
        (28, 54),
        f"Top detection: {top_label.capitalize()} ({top_score:.1f}%)",
        fill=palette.get(top_label, (255, 255, 255)),
    )
    draw.text((28, 78), "Press Q or close the window to stop.", fill=(215, 215, 215))

    for row_index, class_name in enumerate(display_labels):
        score = score_lookup[class_name]
        color = palette.get(class_name, (255, 255, 255))
        y = 106 + row_index * 18
        bar_left = 165
        bar_right = 365
        filled_right = bar_left + int((bar_right - bar_left) * min(score, 100.0) / 100.0)

        draw.text((28, y), f"{class_name.capitalize():<9} {score:5.1f}%", fill=color)
        draw.rectangle((bar_left, y + 3, bar_right, y + 13), outline=(225, 225, 225), width=1)
        draw.rectangle((bar_left, y + 3, filled_right, y + 13), fill=color)

    border_color = palette.get(top_label, (255, 255, 255))
    draw.rectangle((0, 0, frame_image.width - 1, frame_image.height - 1), outline=border_color, width=6)
    return frame_image


def build_ffmpeg_command(ffmpeg_binary, input_spec, width, height, capture_fps, infer_fps):
    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "avfoundation",
    ]
    if capture_fps is not None:
        command.extend(["-framerate", str(capture_fps)])
    command.extend(
        [
            "-video_size",
            f"{width}x{height}",
            "-i",
            input_spec,
            "-vf",
            f"fps={infer_fps}",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-q:v",
            "7",
            "-"
        ]
    )
    return command


def build_ffmpeg_commands(ffmpeg_binary, camera_index, width, height, capture_fps, infer_fps):
    capture_sizes = unique_preserve_order(
        [
            (width, height),
            (640, 480),
            (1280, 720),
            (1920, 1080),
        ]
    )
    fps_candidates = unique_preserve_order([capture_fps, 30, 15, None])
    input_specs = unique_preserve_order([str(camera_index), f"{camera_index}:none"])

    commands = []
    for input_spec in input_specs:
        for current_width, current_height in capture_sizes:
            for current_fps in fps_candidates:
                commands.append(
                    {
                        "description": (
                            f"input={input_spec}, size={current_width}x{current_height}, "
                            f"fps={current_fps if current_fps is not None else 'auto'}"
                        ),
                        "command": build_ffmpeg_command(
                            ffmpeg_binary=ffmpeg_binary,
                            input_spec=input_spec,
                            width=current_width,
                            height=current_height,
                            capture_fps=current_fps,
                            infer_fps=infer_fps,
                        ),
                    }
                )
    return commands


class WebcamApp:
    def __init__(self, model, class_names, display_labels, ffmpeg_commands, window_title):
        self.model = model
        self.class_names = class_names
        self.display_labels = display_labels
        self.ffmpeg_commands = ffmpeg_commands
        self.window_title = window_title

        self.root = tk.Tk()
        self.root.title(self.window_title)
        self.root.configure(bg="#10141a")

        self.image_label = tk.Label(self.root, bg="#10141a")
        self.image_label.pack(padx=12, pady=(12, 8))

        self.status_var = tk.StringVar(value="Starting webcam stream...")
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            fg="#d9d9d9",
            bg="#10141a",
            anchor="w"
        )
        self.status_label.pack(fill="x", padx=12, pady=(0, 12))

        self.root.bind("<q>", lambda _event: self.shutdown())
        self.root.bind("<Escape>", lambda _event: self.shutdown())
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)

        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.capture_process = None
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.current_photo = None
        self.latest_error = None
        self.last_attempt_description = None

    def start(self):
        self.capture_thread.start()
        self.root.after(40, self.refresh_gui)
        self.root.mainloop()

    def refresh_gui(self):
        if self.latest_error and self.frame_queue.empty():
            self.status_var.set(self.latest_error)

        try:
            frame_image = self.frame_queue.get_nowait()
        except queue.Empty:
            frame_image = None

        if frame_image is not None:
            self.current_photo = ImageTk.PhotoImage(frame_image)
            self.image_label.configure(image=self.current_photo)
            self.status_var.set("Live scan active. Press Q or close the window to stop.")

        if not self.stop_event.is_set():
            self.root.after(40, self.refresh_gui)

    def stream_from_current_process(self):
        buffer = bytearray()
        decoded_any_frame = False

        try:
            while not self.stop_event.is_set():
                chunk = self.capture_process.stdout.read(4096)
                if not chunk:
                    break
                buffer.extend(chunk)

                while True:
                    start = buffer.find(b"\xff\xd8")
                    end = buffer.find(b"\xff\xd9", start + 2 if start != -1 else 0)
                    if start == -1 or end == -1:
                        if start > 0:
                            del buffer[:start]
                        break

                    jpeg_bytes = bytes(buffer[start:end + 2])
                    del buffer[:end + 2]

                    try:
                        frame_image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
                    except Exception:
                        continue

                    decoded_any_frame = True
                    model_input = preprocess_frame(frame_image)
                    probabilities = self.model(model_input, training=False).numpy()[0]
                    annotated = annotate_frame(
                        frame_image,
                        probabilities,
                        self.class_names,
                        self.display_labels
                    )

                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put_nowait(annotated)
        finally:
            stderr_output = ""
            if self.capture_process and self.capture_process.stderr is not None:
                stderr_output = self.capture_process.stderr.read().decode("utf-8", errors="replace").strip()
            self.terminate_process()
        return decoded_any_frame, stderr_output

    def capture_loop(self):
        last_error = None
        for attempt in self.ffmpeg_commands:
            if self.stop_event.is_set():
                return

            self.last_attempt_description = attempt["description"]
            self.capture_process = subprocess.Popen(
                attempt["command"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )

            decoded_any_frame, stderr_output = self.stream_from_current_process()
            if decoded_any_frame:
                if not self.stop_event.is_set():
                    self.latest_error = (
                        "Camera stream stopped after starting. "
                        "Close apps already using the camera, then try again."
                    )
                return

            if stderr_output:
                last_error = stderr_output

        if not self.stop_event.is_set():
            if last_error:
                self.latest_error = (
                    "Camera stream could not start with the tested macOS capture modes.\n"
                    f"Last ffmpeg error:\n{last_error}"
                )
            else:
                self.latest_error = (
                    "Camera stream could not start. Try another camera index with "
                    "`--camera 1` or `--camera 2`."
                )

    def terminate_process(self):
        if self.capture_process and self.capture_process.poll() is None:
            self.capture_process.terminate()
            try:
                self.capture_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.capture_process.kill()

    def shutdown(self):
        self.stop_event.set()
        self.terminate_process()
        if self.root.winfo_exists():
            self.root.quit()
            self.root.destroy()


def main():
    if os.name != "posix" or not sys_platform_is_macos():
        raise RuntimeError("This script is configured for macOS AVFoundation capture.")

    args = parse_args()
    ffmpeg_binary = resolve_ffmpeg()
    cameras = list_avfoundation_cameras(ffmpeg_binary)

    if args.list_cameras:
        print("Available cameras:")
        for index, name in cameras:
            print(f"  [{index}] {name}")
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Run the training notebook first so `zombienet_best.keras` exists."
        )

    class_names = load_class_names()
    display_labels = ordered_display_labels(class_names)

    tf.keras.utils.set_random_seed(42)
    model = tf.keras.models.load_model(MODEL_PATH)

    camera_index, camera_name = choose_camera(cameras, requested_index=args.camera)
    ffmpeg_commands = build_ffmpeg_commands(
        ffmpeg_binary=ffmpeg_binary,
        camera_index=camera_index,
        width=args.width,
        height=args.height,
        capture_fps=args.capture_fps,
        infer_fps=args.infer_fps,
    )

    print(f"Launching ZombieNet live scan on camera [{camera_index}] {camera_name}")
    print("If the window stays black, try `--camera 1` or `--camera 2`.")

    app = WebcamApp(
        model=model,
        class_names=class_names,
        display_labels=display_labels,
        ffmpeg_commands=ffmpeg_commands,
        window_title=f"ZombieNet Live Demo — {camera_name}",
    )
    app.start()


def sys_platform_is_macos():
    import sys
    return sys.platform == "darwin"


if __name__ == "__main__":
    main()
