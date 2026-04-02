"""
face_process.py
---------------
Sirf face recognition + DeepFace analysis karta hai.
Detection queue se data leta hai, face result queue mein dalta hai.
 
FIXES:
- recognize_face threshold 700 (50x50 flattened pixel distance sahi range)
- DeepFace age/gender/emotion — last known values cache kiye jate hain
  taaki har frame pe "Unknown" na aaye Telegram pe
- reload_queue se 'r' key ka signal receive karta hai
- names.pkl fix — faces.py ka duplicate entry bug bhi yahan handle
"""
import cv2
import pickle
import numpy as np
import os
import time
from multiprocessing import Queue
 
 
def load_face_data():
    faces_data, known_names = [], []
    try:
        if os.path.exists("data/faces_data.pkl") and os.path.exists("data/names.pkl"):
            with open("data/faces_data.pkl", "rb") as f:
                faces_data = pickle.load(f)
            with open("data/names.pkl", "rb") as f:
                known_names = pickle.load(f)
 
            # BUG FIX: names.pkl mein faces_data se zyada entries ho sakti hain
            # (faces.py mein `[name] * len(faces_data)` ne cumulative data pe
            # multiply kiya tha). Sirf utne names rakho jitne face rows hain.
            if len(known_names) > len(faces_data):
                known_names = known_names[-len(faces_data):]
 
            print(f"[Face] Loaded {len(faces_data)} face samples, {len(set(known_names))} unique persons")
        else:
            print("[Face] No face data found. Run faces.py first.")
    except Exception as e:
        print(f"[Face] Load error: {e}")
    return faces_data, known_names
 
 
def recognize_face(face_img, faces_data, known_names, threshold=700):
    """
    threshold=700 — 50x50 flattened image (7500 values) ke liye
    Euclidean distance ~300-400 same person, ~800+ different person.
    Pehle 70 tha jo kabhi match nahi karta tha.
    """
    if len(faces_data) == 0 or face_img.size == 0:
        return "Unknown"
    try:
        face_flattened = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY).reshape(-1).astype(np.float32)
        distances = [np.linalg.norm(face_flattened - kf.astype(np.float32)) for kf in faces_data]
        min_dist = min(distances)
        if min_dist < threshold:
            return known_names[distances.index(min_dist)]
    except Exception as e:
        print(f"[Face] recognize error: {e}")
    return "Unknown"
 
 
def face_worker(detection_queue: Queue, result_queue: Queue, reload_queue: Queue, stop_event):
    facedetect = None
    try:
        facedetect = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
        if facedetect.empty():
            print("[Face] Haar cascade not found!")
            facedetect = None
    except Exception as e:
        print(f"[Face] Cascade error: {e}")
 
    faces_data, known_names = load_face_data()
 
    # DeepFace cache — last known values store karo
    # Taaki Telegram pe hamesha latest info jaaye, "Unknown" nahi
    deepface_cache = {}   # key: bbox tuple -> {"age", "gender", "emotion", "ts"}
    deepface_cooldown = 0
    DEEPFACE_CACHE_TTL = 60   # 60 seconds tak cached values valid
 
    frame_count = 0
    print("[Face] Started")
 
    while not stop_event.is_set():
 
        # 'r' key reload signal check karo
        try:
            reload_queue.get_nowait()
            faces_data, known_names = load_face_data()
            deepface_cache.clear()
            print(f"[Face] Reloaded — {len(set(known_names))} persons")
        except Exception:
            pass
 
        try:
            data = detection_queue.get(timeout=1.0)
        except Exception:
            continue
 
        frame = data["frame"]
        detections = data["detections"]
        frame_count += 1
 
        enriched = []
        now = time.time()
 
        for det in detections:
            label = det["label"]
            x1, y1, x2, y2 = det["bbox"]
            face_name = "Unknown"
            age = gender = emotion = "Unknown"
            face_bbox_abs = None
 
            if label == "person" and (x2 - x1) > 80 and (y2 - y1) > 80:
                try:
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size > 0 and facedetect is not None:
                        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                        faces = facedetect.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
 
                        if len(faces) > 0:
                            fx, fy, fw, fh = faces[0]
                            if fw > 20 and fh > 20:
                                face_crop = person_crop[fy:fy + fh, fx:fx + fw]
                                if face_crop.size > 0:
                                    # Face recognition
                                    resized = cv2.resize(face_crop, (50, 50))
                                    face_name = recognize_face(resized, faces_data, known_names)
 
                                    # DeepFace cache key — approximate bbox position
                                    cache_key = (x1 // 40, y1 // 40)
 
                                    # Cache check — fresh data hai?
                                    cached = deepface_cache.get(cache_key)
                                    if cached and (now - cached["ts"]) < DEEPFACE_CACHE_TTL:
                                        age    = cached["age"]
                                        gender = cached["gender"]
                                        emotion = cached["emotion"]
                                    
                                    # DeepFace — throttled + cooldown
                                    if frame_count % 60 == 0 and now > deepface_cooldown:
                                        try:
                                            from deepface import DeepFace
                                            analysis = DeepFace.analyze(
                                                face_crop,
                                                actions=["age", "gender", "emotion"],
                                                enforce_detection=False,
                                                detector_backend="skip"
                                            )
                                            if isinstance(analysis, list):
                                                analysis = analysis[0]
                                            age    = str(int(analysis.get("age", 0)))
                                            gender = analysis.get("dominant_gender", "Unknown")
                                            emotion = analysis.get("dominant_emotion", "Unknown")
 
                                            # Cache mein save karo
                                            deepface_cache[cache_key] = {
                                                "age": age, "gender": gender,
                                                "emotion": emotion, "ts": now
                                            }
                                            deepface_cooldown = now + 10
                                            print(f"[Face] DeepFace: {face_name} | {age}yo {gender} {emotion}")
                                        except Exception as e:
                                            print(f"[Face] DeepFace error: {e}")
 
                                    face_bbox_abs = (x1 + fx, y1 + fy, fw, fh)
                except Exception as e:
                    print(f"[Face] person analysis error: {e}")
 
            entry = {
                **det,
                "face_name": face_name,
                "age":       age,
                "gender":    gender,
                "emotion":   emotion,
            }
            if face_bbox_abs:
                entry["face_bbox"] = face_bbox_abs
 
            enriched.append(entry)
 
        result_queue.put({
            "frame":      frame,
            "detections": enriched
        })
 
    print("[Face] Stopped")
 