
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import os
import pickle
import csv
import time
import threading
from datetime import datetime
from PIL import Image, ImageTk
import pyttsx3


# ─────────────────────────────────────────────
#  VOICE
# ─────────────────────────────────────────────

_speak_queue = []
_speak_running = False

def speak(text):
    global _speak_running
    _speak_queue.append(text)
    if not _speak_running:
        _speak_running = True
        threading.Thread(target=_speak_loop, daemon=True).start()

def _speak_loop():
    global _speak_running
    try:
        engine = pyttsx3.init()
        while _speak_queue:
            engine.say(_speak_queue.pop(0))
            engine.runAndWait()
    except Exception:
        pass
    _speak_running = False


# ─────────────────────────────────────────────
#  FOLDERS & CASCADE
# ─────────────────────────────────────────────

for d in ("data", "Attendance", "faces"):
    os.makedirs(d, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    print("ERROR: Haar cascade not found")
    exit(1)

# ── recognition settings ──────────────────────
FACE_SIZE    = 100          # pixels (square)
KNN_K        = 3            # neighbours to vote
# Distance threshold — raise if too many "Unknown", lower if wrong names shown
DIST_THRESH  = 6000


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def preprocess(bgr_crop):
    """BGR crop → equalised grayscale 100×100 flat float64 vector."""
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, (FACE_SIZE, FACE_SIZE))
    return gray.flatten().astype(np.float64)


def knn_predict(vec, faces_db, labels, k=KNN_K, threshold=DIST_THRESH):
    """
    Simple k-NN on Euclidean distances.
    Returns the majority-vote label, or 'Unknown' if best distance > threshold.
    """
    dists = np.linalg.norm(faces_db - vec, axis=1)
    top_k = np.argsort(dists)[:k]

    # If the closest neighbour is too far, reject
    if dists[top_k[0]] > threshold:
        return "Unknown", dists[top_k[0]]

    # Majority vote
    votes = {}
    for idx in top_k:
        name = labels[idx]
        votes[name] = votes.get(name, 0) + 1

    winner = max(votes, key=votes.get)
    return winner, dists[top_k[0]]


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────

class FaceAttendanceApp:

    def __init__(self,root):
        self.root = root
        self.root.title("Face Attendance System")
        self.root.geometry("1100x620")
        self.root.resizable(False, False)
        self.root.configure(bg="#0f0f1a")

        # camera
        self.video             = None
        self.current_frame     = None
        self._after_id         = None
        self.attendance_running = False
        self._frame_lock       = threading.Lock()

        # capture buffer (current session only)
        self.session_faces  = []   # flat float64 vectors
        self.session_count  = 0    # images saved to disk this session

        # loaded training data
        self.db_faces  = None
        self.db_labels = []

        # attendance cooldown
        self.last_mark_time = {}

        # ── UI ───────────────────────────────────────────────────────────────

        left = tk.Frame(root, bg="#12121f", width=300)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        right = tk.Frame(root, bg="#0a0a14")
        right.pack(side="right", fill="both", expand=True)

        # title
        tk.Label(left, text="Face\nAttendance\nSystem",
                 font=("Courier", 20, "bold"), fg="#e0e0ff",
                 bg="#12121f", justify="center").pack(pady=(28, 10))

        tk.Frame(left, bg="#2a2a4a", height=1).pack(fill="x", padx=20, pady=6)

        # name entry
        tk.Label(left, text="Student / Staff Name",
                 font=("Courier", 10), fg="#6060a0",
                 bg="#12121f").pack(pady=(10, 2))

        self.name_entry = tk.Entry(
            left, font=("Courier", 13), width=22,
            relief="flat", bg="#1e1e38", fg="#e0e0ff",
            insertbackground="#e0e0ff", justify="center"
        )
        self.name_entry.pack(pady=(0, 16), ipady=6)

        tk.Frame(left, bg="#2a2a4a", height=1).pack(fill="x", padx=20, pady=2)

        # buttons
        for label, cmd, bg, abg in [
            ("Start Camera",     self.start_camera,     "#1d6ef5", "#1558d0"),
            ("Capture Face",     self.capture_face,     "#059669", "#047857"),
            ("Train System",     self.train_data,       "#7c3aed", "#5b21b6"),
            ("Start Attendance", self.start_attendance, "#d97706", "#b45309"),
            ("Exit",             self.exit_app,         "#dc2626", "#991b1b"),
        ]:
            tk.Button(left, text=label, font=("Courier", 12, "bold"),
                      fg="white", bg=bg, activebackground=abg,
                      activeforeground="white", relief="flat",
                      cursor="hand2", command=cmd,
                      width=22, pady=10).pack(pady=6, padx=18)

        tk.Frame(left, bg="#2a2a4a", height=1).pack(fill="x", padx=20, pady=10)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(left, textvariable=self.status_var,
                 font=("Courier", 9), fg="#5050a0", bg="#12121f",
                 wraplength=260, justify="center").pack(pady=(4, 20), padx=10)

        # camera area
        self.camera_label = tk.Label(right, bg="#0a0a14")
        self.camera_label.pack(fill="both", expand=True)

        self.placeholder = tk.Label(
            right,
            text="Camera is off\n\nPress  Start Camera\nto begin",
            font=("Courier", 15), fg="#2a2a4a",
            bg="#0a0a14", justify="center"
        )
        self.placeholder.place(relx=0.5, rely=0.5, anchor="center")

        speak("Application started")


    # ── CAMERA ───────────────────────────────────────────────────────────────

    def start_camera(self):
        if self.video is not None:
            return
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera.")
            return
        self.video = cap
        self.attendance_running = False
        self.placeholder.place_forget()
        self._preview_loop()
        speak("Camera started")
        self.status_var.set("Camera running")


    def _preview_loop(self):
        if self.attendance_running or self.video is None:
            return
        ret, frame = self._read_frame()
        if ret and frame is not None:
            try:
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 100), 2)
            except Exception:
                pass
            self._show(frame)
        self._after_id = self.root.after(20, self._preview_loop)


    def _read_frame(self):
        try:
            ret, frame = self.video.read()
            if ret and frame is not None:
                with self._frame_lock:
                    self.current_frame = frame.copy()
            return ret, frame
        except Exception as e:
            print(f"[read] {e}")
            return False, None


    def _show(self, frame):
        try:
            img   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(img).resize((800, 620))
            imgtk = ImageTk.PhotoImage(img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)
        except Exception as e:
            print(f"[show] {e}")


    # ── CAPTURE ──────────────────────────────────────────────────────────────

    def capture_face(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Enter a name first.")
            return

        if self.video is None:
            self.start_camera()
            self.root.after(700, self.capture_face)
            return

        with self._frame_lock:
            frame = self.current_frame.copy() if self.current_frame is not None else None

        if frame is None:
            messagebox.showerror("Error", "Camera not ready — wait a moment.")
            return

        # detect
        try:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray  = cv2.equalizeHist(gray)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed:\n{e}")
            return

        if len(faces) == 0:
            messagebox.showwarning("No Face",
                "No face detected.\nMake sure your face is visible and well-lit.")
            return

        x, y, w, h = faces[0]
        crop = frame[y:y+h, x:x+w]

        if crop.size == 0:
            messagebox.showerror("Error", "Face crop is empty. Try again.")
            return

        # save to disk
        folder = f"faces/{name}"
        os.makedirs(folder, exist_ok=True)
        path = f"{folder}/{self.session_count}.jpg"
        cv2.imwrite(path, crop)
        self.session_count += 1

        # store vector in session buffer
        vec = preprocess(crop)
        self.session_faces.append(vec)

        count = len(self.session_faces)
        self.status_var.set(f"Captured {count} image(s) for '{name}'" +
            (f"  — {max(0, 4-count)} more needed" if count < 4 else "  ✔ Ready to train"))
        speak(f"Photo {count} saved")

        needed = max(0, 4 - count)
        if needed > 0:
            messagebox.showinfo("Captured",
                f"Image {count} saved.\nCapture {needed} more image(s) for '{name}'.")
        else:
            messagebox.showinfo("Captured",
                f"Image {count} saved for '{name}'.\n"
                f"You can capture more or click Train System.")


    # ── TRAIN ────────────────────────────────────────────────────────────────

    def train_data(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Enter a name.")
            return
        if len(self.session_faces) < 4:
            messagebox.showerror("Error",
                f"Capture at least 4 images before training.\nYou have {len(self.session_faces)} so far.")
            return

        try:
            names_path = "data/names.pkl"
            faces_path = "data/faces_data.pkl"

            # load existing
            if os.path.exists(names_path) and os.path.exists(faces_path):
                with open(names_path, "rb") as f:
                    all_labels = pickle.load(f)
                with open(faces_path, "rb") as f:
                    all_faces  = pickle.load(f)

                new_vecs = np.array(self.session_faces, dtype=np.float64)

                # shape guard
                if all_faces.ndim == 2 and all_faces.shape[1] == new_vecs.shape[1]:
                    all_faces  = np.vstack([all_faces, new_vecs])
                    all_labels = all_labels + [name] * len(self.session_faces)
                else:
                    # incompatible old data — start fresh
                    all_faces  = new_vecs
                    all_labels = [name] * len(self.session_faces)
            else:
                all_faces  = np.array(self.session_faces, dtype=np.float64)
                all_labels = [name] * len(self.session_faces)

            with open(names_path, "wb") as f:
                pickle.dump(all_labels, f)
            with open(faces_path, "wb") as f:
                pickle.dump(all_faces, f)

        except Exception as e:
            messagebox.showerror("Training Error", f"Failed to save data:\n{e}")
            return

        self.session_faces.clear()
        self.session_count = 0

        people = sorted(set(all_labels))
        speak("Training complete")
        self.status_var.set("Trained: " + ", ".join(people))
        messagebox.showinfo("Done",
            f"Training successful!\nPeople in system: {', '.join(people)}")


    # ── START ATTENDANCE ──────────────────────────────────────────────────────

    def start_attendance(self):
        names_path = "data/names.pkl"
        faces_path = "data/faces_data.pkl"

        if not os.path.exists(names_path) or not os.path.exists(faces_path):
            messagebox.showerror("Error",
                "No trained data found.\nCapture faces and click Train System first.")
            return

        if self.video is None:
            self.start_camera()
            self.root.after(800, self.start_attendance)
            return

        try:
            with open(names_path, "rb") as f:
                self.db_labels = pickle.load(f)
            with open(faces_path, "rb") as f:
                self.db_faces  = pickle.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load training data:\n{e}")
            return

        if self.db_faces is None or len(self.db_faces) == 0:
            messagebox.showerror("Error", "Training data is empty. Please retrain.")
            return

        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None

        self.attendance_running = True
        speak("Attendance started")
        self.status_var.set("Attendance mode — scanning…")
        messagebox.showinfo("Attendance", "Face recognition is now running.")
        self._attendance_loop()


    def _attendance_loop(self):
        if not self.attendance_running or self.video is None:
            return

        ret, frame = self._read_frame()
        if not ret or frame is None:
            self._after_id = self.root.after(30, self._attendance_loop)
            return

        try:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray  = cv2.equalizeHist(gray)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
        except Exception as e:
            print(f"[attendance detect] {e}")
            self._after_id = self.root.after(30, self._attendance_loop)
            return

        for (x, y, w, h) in faces:
            try:
                crop = frame[y:y+h, x:x+w]
                if crop.size == 0:
                    continue

                vec  = preprocess(crop)

                # shape guard
                if (self.db_faces.ndim != 2 or
                        self.db_faces.shape[1] != vec.shape[0]):
                    name, dist = "Unknown", 99999
                else:
                    name, dist = knn_predict(vec, self.db_faces, self.db_labels)

                color = (0, 220, 80) if name != "Unknown" else (30, 30, 220)
                label = f"{name} ({dist:.0f})"

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                self._log_attendance(name)

            except Exception as e:
                print(f"[attendance face] {e}")
                continue

        self._show(frame)
        self._after_id = self.root.after(50, self._attendance_loop)


    # ── LOG ───────────────────────────────────────────────────────────────────

    def _log_attendance(self, name):
        if not name or name == "Unknown":
            return
        now = time.time()
        if now - self.last_mark_time.get(name, 0) < 30:
            return
        self.last_mark_time[name] = now

        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            time_str = datetime.now().strftime("%H:%M:%S")
            path     = f"Attendance/attendance_{date_str}.csv"
            exists   = os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if not exists:
                    w.writerow(["Name", "Date", "Time"])
                w.writerow([name, date_str, time_str])
            self.status_var.set(f"Marked: {name} at {time_str}")
            speak(f"Attendance marked for {name}")
        except Exception as e:
            print(f"[log] {e}")


    # ── EXIT ──────────────────────────────────────────────────────────────────

    def exit_app(self):
        self.attendance_running = False
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        if self.video:
            self.video.release()
            self.video = None
        speak("Application closed")
        self.root.after(400, self.root.destroy)


# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    FaceAttendanceApp(root)
    root.mainloop()
