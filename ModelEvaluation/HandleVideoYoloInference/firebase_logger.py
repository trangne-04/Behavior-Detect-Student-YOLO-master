from google.cloud import firestore
import datetime
# Tải service account key từ Firebase Console
cred = credentials.Certificate("/hdd2/minhnv/CodingYOLOv12/Behavior-Detect-Student-YOLO/ModelEvaluation/HandleVideoYoloInference/google-services.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
# ========== Lớp quản lý Firebase ==========
class FirebaseLogger:
    def __init__(self):
        self.batch = None
        self.batch_size = 0
        self.MAX_BATCH_SIZE = 50

    def start_session(self):
        global current_session_id
        session_ref = db.collection('sessions').document()
        current_session_id = session_ref.id
        session_ref.set({
            'start_time': datetime.now(),
            'status': 'recording',
            # 'video_path': VIDEO_PATHS[current_video_index]
        })
        self._init_batch()

    def _init_batch(self):
        self.batch = db.batch()
        self.batch_size = 0

    def log_behavior(self, behavior_type, confidence):
        if self.batch_size >= self.MAX_BATCH_SIZE:
            self._commit_batch()

        log_ref = db.collection('sessions').document(current_session_id)\
                    .collection('behaviors').document()
        self.batch.set(log_ref, {
            'type': behavior_type,
            'confidence': confidence,
            'timestamp': datetime.now()
        })
        self.batch_size += 1

    def _commit_batch(self):
        if self.batch_size > 0:
            self.batch.commit()
            self._init_batch()

    def end_session(self):
        self._commit_batch()
        session_ref = db.collection('sessions').document(current_session_id)
        session_ref.update({
            'end_time': datetime.now(),
            'status': 'completed'
        })