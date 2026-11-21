"""
SMART MIRROR - smoother version
Install: python -m pip install opencv-python==4.11.0.86 deepface
Run:     python mirror1.py
Press Q or ESC to quit.

DeepFace analysis runs in a background thread so it won't block the webcam loop.
Age and expression results are approximate predictions. Webcam frames are not saved.
"""
import cv2
import time
import threading
import queue

CAMERA_INDEX = 0
AI_INTERVAL_SECONDS = 2.0
WINDOW_NAME = "Smart Mirror"


class FaceAnalyzer:
    def __init__(self):
        self.jobs = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.result = None
        self.status = "Loading AI..."
        self.deepface = None
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            from deepface import DeepFace
            self.deepface = DeepFace
            with self.lock:
                self.status = "AI ready"
        except Exception as exc:
            with self.lock:
                self.status = f"AI unavailable: {type(exc).__name__}"
            print("DeepFace import failed:", repr(exc))
            return

        while True:
            crop = self.jobs.get()
            try:
                h, w = crop.shape[:2]
                if max(h, w) > 320:
                    scale = 320 / max(h, w)
                    crop = cv2.resize(crop, (int(w * scale), int(h * scale)))

                output = self.deepface.analyze(
                    img_path=crop,
                    actions=["age", "emotion"],
                    enforce_detection=False,
                    detector_backend="skip",
                    silent=True,
                )
                item = output[0] if isinstance(output, list) else output
                age = item.get("age")
                emotions = item.get("emotion") or {}
                expression = max(emotions, key=emotions.get) if emotions else "unknown"
                with self.lock:
                    self.result = {
                        "age": int(round(age)) if age is not None else None,
                        "expression": str(expression),
                    }
                    self.status = "AI ready"
            except Exception as exc:
                with self.lock:
                    self.status = f"AI error: {type(exc).__name__}"
                print("DeepFace analysis failed:", repr(exc))
            finally:
                self.jobs.task_done()

    def submit(self, crop):
        if self.deepface is None:
            return
        try:
            self.jobs.put_nowait(crop.copy())
        except queue.Full:
            pass

    def get_result(self):
        with self.lock:
            return self.result, self.status


def main():
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        raise RuntimeError("Could not load OpenCV face detector.")

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Close other camera apps or try CAMERA_INDEX = 1.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    analyzer = FaceAnalyzer()
    last_submit = 0.0
    latest_result = None
    ai_status = "Loading AI..."
    fps = 0.0
    previous_time = time.time()

    print("Smart Mirror running. Press Q or ESC to exit.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=5, minSize=(65, 65)
            )

            if len(faces):
                # Analyze the largest face; outline every detected face.
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                now = time.time()
                if now - last_submit >= AI_INTERVAL_SECONDS:
                    px, py = int(w * 0.15), int(h * 0.15)
                    x1, y1 = max(0, x - px), max(0, y - py)
                    x2, y2 = min(frame.shape[1], x + w + px), min(frame.shape[0], y + h + py)
                    analyzer.submit(frame[y1:y2, x1:x2])
                    last_submit = now

            latest_result, ai_status = analyzer.get_result()

            green = (60, 230, 120)
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), green, 2)
                c = max(12, min(w, h) // 7)
                t = 4
                cv2.line(frame, (x, y), (x + c, y), green, t)
                cv2.line(frame, (x, y), (x, y + c), green, t)
                cv2.line(frame, (x + w, y), (x + w - c, y), green, t)
                cv2.line(frame, (x + w, y), (x + w, y + c), green, t)
                cv2.line(frame, (x, y + h), (x + c, y + h), green, t)
                cv2.line(frame, (x, y + h), (x, y + h - c), green, t)
                cv2.line(frame, (x + w, y + h), (x + w - c, y + h), green, t)
                cv2.line(frame, (x + w, y + h), (x + w, y + h - c), green, t)

            if len(faces) and latest_result:
                age = latest_result.get("age")
                expression = latest_result.get("expression", "unknown")
                label = f"Approx age: {age if age is not None else '?'}  |  Expression: {expression}"
                cv2.putText(frame, label, (20, 132), cv2.FONT_HERSHEY_SIMPLEX,
                            0.58, green, 2, cv2.LINE_AA)

            now = time.time()
            dt = now - previous_time
            if dt > 0:
                current_fps = 1.0 / dt
                fps = current_fps if fps == 0 else 0.9 * fps + 0.1 * current_fps
            previous_time = now

            cv2.putText(frame, "SMART MIRROR", (20, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, green, 2, cv2.LINE_AA)
            cv2.putText(frame, f"FACES: {len(faces)}  |  FPS: {fps:.1f}", (20, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(frame, ai_status[:65], (20, 102), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (190, 190, 190), 1, cv2.LINE_AA)
            cv2.putText(frame, "Q / ESC - EXIT", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Smart Mirror closed.")


if __name__ == "__main__":
    main()
