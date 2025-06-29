# video_pipeline.py
"""
Real-Time Video Pipeline with Face Recognition
Author: AI Meeting Recorder Team
Date: 2024

Video Input → Frame Capture → Frame Queue → Face Detection → 
Face Recognition → Identity Tracking → Results Aggregation → Output
"""

import cv2
import numpy as np
import face_recognition
import threading
import queue
import time
import json
import pickle
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from mtcnn import MTCNN

# Google Drive API imports
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.http import MediaIoBaseDownload
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DetectedPerson:
    """Data structure for detected person information"""
    person_id: str
    display_name: str
    is_known: bool
    confidence: float
    face_location: Tuple[int, int, int, int]  # (top, right, bottom, left)
    face_encoding: Optional[np.ndarray] = None
    first_seen_frame: int = 0
    consecutive_frames: int = 0
    total_appearances: int = 0
    last_seen_timestamp: float = 0.0

@dataclass
class FrameResult:
    """Data structure for frame processing results"""
    timestamp: str
    frame_number: int
    processing_time_ms: float
    detected_persons: List[DetectedPerson]
    frame_shape: Tuple[int, int, int]
    statistics: Dict

class GoogleDriveManager:
    """Handles Google Drive integration for employee photos"""
    
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    
    def __init__(self, credentials_path: str = 'credentials.json', token_path: str = 'token.json'):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Drive API"""
        creds = None
        
        # Load existing token
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, self.SCOPES)
        
        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_path):
                    logger.error(f"Google Drive credentials file not found: {self.credentials_path}")
                    logger.info("Please download credentials.json from Google Cloud Console")
                    return
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('drive', 'v3', credentials=creds)
        logger.info("Google Drive API authenticated successfully")
    
    def download_employee_photos(self, folder_name: str = "employee_photos", 
                                local_dir: str = "./employee_photos"):
        """Download employee photos from Google Drive folder"""
        if not self.service:
            logger.error("Google Drive service not authenticated")
            return False
        
        try:
            # Find the employee photos folder
            folder_query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
            folder_results = self.service.files().list(q=folder_query).execute()
            folders = folder_results.get('files', [])
            
            if not folders:
                logger.error(f"Folder '{folder_name}' not found in Google Drive")
                return False
            
            employee_folder_id = folders[0]['id']
            logger.info(f"Found employee photos folder: {folder_name}")
            
            # Create local directory
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            
            # Get all person folders inside employee_photos
            person_folders_query = f"'{employee_folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
            person_results = self.service.files().list(q=person_folders_query).execute()
            person_folders = person_results.get('files', [])
            
            downloaded_count = 0
            
            for person_folder in person_folders:
                person_name = person_folder['name']
                person_folder_id = person_folder['id']
                
                # Create local person directory
                person_dir = os.path.join(local_dir, person_name)
                Path(person_dir).mkdir(parents=True, exist_ok=True)
                
                # Get all images in person folder
                images_query = f"'{person_folder_id}' in parents and (mimeType contains 'image/')"
                image_results = self.service.files().list(q=images_query).execute()
                images = image_results.get('files', [])
                
                for image in images:
                    image_name = image['name']
                    image_id = image['id']
                    
                    # Download image
                    local_image_path = os.path.join(person_dir, image_name)
                    
                    if not os.path.exists(local_image_path):
                        request = self.service.files().get_media(fileId=image_id)
                        fh = io.BytesIO()
                        downloader = MediaIoBaseDownload(fh, request)
                        
                        done = False
                        while done is False:
                            status, done = downloader.next_chunk()
                        
                        # Save to local file
                        with open(local_image_path, 'wb') as f:
                            f.write(fh.getvalue())
                        
                        downloaded_count += 1
                        logger.info(f"Downloaded: {person_name}/{image_name}")
            
            logger.info(f"Successfully downloaded {downloaded_count} employee photos")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading employee photos: {str(e)}")
            return False

class VideoCapture:
    """Handles video input and frame capture"""
    
    def __init__(self, source=0, width=1280, height=720, fps=30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=30)
        self.running = False
        self.thread = None
        self.frame_count = 0
        
    def start(self):
        """Start video capture thread"""
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.thread.start()
        logger.info(f"Video capture started: {self.width}x{self.height} @ {self.fps}fps")
    
    def _capture_frames(self):
        """Capture frames in separate thread"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame from video source")
                continue
            
            self.frame_count += 1
            
            # Add frame to queue (drop old frames if queue is full)
            try:
                self.frame_queue.put_nowait((self.frame_count, frame, time.time()))
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()  # Remove old frame
                    self.frame_queue.put_nowait((self.frame_count, frame, time.time()))
                except queue.Empty:
                    pass
    
    def get_frame(self):
        """Get latest frame from queue"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop video capture"""
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
        logger.info("Video capture stopped")

class FaceDetector:
    """Handles face detection using MTCNN"""
    
    def __init__(self, min_face_size=40, scale_factor=0.709, steps_threshold=None):
        if steps_threshold is None:
            steps_threshold = [0.6, 0.7, 0.7]
        
        self.detector = MTCNN(
            min_face_size=min_face_size,
            scale_factor=scale_factor,
            steps_threshold=steps_threshold
        )
        self.detection_queue = queue.Queue(maxsize=10)
        self.running = False
        self.thread = None
        
    def start(self, frame_queue):
        """Start face detection thread"""
        self.frame_queue = frame_queue
        self.running = True
        self.thread = threading.Thread(target=self._detect_faces, daemon=True)
        self.thread.start()
        logger.info("Face detector started")
    
    def _detect_faces(self):
        """Detect faces in frames"""
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=1.0)
                if frame_data is None:
                    continue
                
                frame_number, frame, timestamp = frame_data
                
                # Convert BGR to RGB for MTCNN
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                detections = self.detector.detect_faces(rgb_frame)
                
                face_locations = []
                confidences = []
                
                for detection in detections:
                    confidence = detection['confidence']
                    if confidence > 0.8:  # Confidence threshold
                        box = detection['box']
                        # Convert to face_recognition format (top, right, bottom, left)
                        x, y, w, h = box
                        face_location = (y, x + w, y + h, x)
                        face_locations.append(face_location)
                        confidences.append(confidence)
                
                # Add to detection queue
                detection_result = {
                    'frame_number': frame_number,
                    'frame': frame,
                    'timestamp': timestamp,
                    'face_locations': face_locations,
                    'confidences': confidences
                }
                
                try:
                    self.detection_queue.put_nowait(detection_result)
                except queue.Full:
                    # Remove old detection if queue is full
                    try:
                        self.detection_queue.get_nowait()
                        self.detection_queue.put_nowait(detection_result)
                    except queue.Empty:
                        pass
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Face detection error: {str(e)}")
    
    def get_detections(self):
        """Get face detection results"""
        try:
            return self.detection_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop face detection"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Face detector stopped")

class FaceRecognizer:
    """Handles face recognition and known person database"""
    
    def __init__(self, tolerance=0.6, unknown_threshold=0.4):
        self.tolerance = tolerance
        self.unknown_threshold = unknown_threshold
        self.known_encodings = []
        self.known_names = []
        self.recognition_queue = queue.Queue(maxsize=10)
        self.running = False
        self.thread = None
        
    def load_employee_database(self, database_path: str):
        """Load known faces from employee photos directory"""
        if not os.path.exists(database_path):
            logger.warning(f"Employee database directory not found: {database_path}")
            return 0
        
        self.known_encodings = []
        self.known_names = []
        loaded_count = 0
        
        for person_folder in os.listdir(database_path):
            person_path = os.path.join(database_path, person_folder)
            if not os.path.isdir(person_path):
                continue
            
            person_name = person_folder.replace('_', ' ').title()
            person_encodings = []
            
            for image_file in os.listdir(person_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(person_path, image_file)
                    
                    try:
                        # Load and encode face
                        image = face_recognition.load_image_file(image_path)
                        encodings = face_recognition.face_encodings(image)
                        
                        if encodings:
                            person_encodings.extend(encodings)
                            loaded_count += 1
                            logger.debug(f"Loaded encoding for {person_name} from {image_file}")
                    
                    except Exception as e:
                        logger.warning(f"Failed to load {image_path}: {str(e)}")
            
            # Add all encodings for this person
            for encoding in person_encodings:
                self.known_encodings.append(encoding)
                self.known_names.append(person_name)
        
        logger.info(f"Loaded {loaded_count} face encodings for {len(set(self.known_names))} people")
        return loaded_count
    
    def start(self, detection_queue):
        """Start face recognition thread"""
        self.detection_queue = detection_queue
        self.running = True
        self.thread = threading.Thread(target=self._recognize_faces, daemon=True)
        self.thread.start()
        logger.info("Face recognizer started")
    
    def _recognize_faces(self):
        """Recognize faces in detected face crops"""
        while self.running:
            try:
                detection_data = self.detection_queue.get(timeout=1.0)
                if detection_data is None:
                    continue
                
                frame = detection_data['frame']
                face_locations = detection_data['face_locations']
                
                if not face_locations:
                    # No faces detected, pass through
                    recognition_result = {
                        **detection_data,
                        'face_encodings': [],
                        'face_names': [],
                        'face_confidences': []
                    }
                    
                    try:
                        self.recognition_queue.put_nowait(recognition_result)
                    except queue.Full:
                        try:
                            self.recognition_queue.get_nowait()
                            self.recognition_queue.put_nowait(recognition_result)
                        except queue.Empty:
                            pass
                    continue
                
                # Convert BGR to RGB for face_recognition
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Get face encodings
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                face_names = []
                face_confidences = []
                
                for encoding in face_encodings:
                    name = "Unknown"
                    confidence = 0.0
                    
                    if len(self.known_encodings) > 0:
                        # Compare with known faces
                        distances = face_recognition.face_distance(self.known_encodings, encoding)
                        best_match_index = np.argmin(distances)
                        
                        if distances[best_match_index] <= self.tolerance:
                            name = self.known_names[best_match_index]
                            confidence = 1.0 - distances[best_match_index]
                        else:
                            confidence = max(0.0, self.unknown_threshold - distances[best_match_index])
                    
                    face_names.append(name)
                    face_confidences.append(confidence)
                
                # Create recognition result
                recognition_result = {
                    **detection_data,
                    'face_encodings': face_encodings,
                    'face_names': face_names,
                    'face_confidences': face_confidences
                }
                
                try:
                    self.recognition_queue.put_nowait(recognition_result)
                except queue.Full:
                    try:
                        self.recognition_queue.get_nowait()
                        self.recognition_queue.put_nowait(recognition_result)
                    except queue.Empty:
                        pass
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Face recognition error: {str(e)}")
    
    def get_recognitions(self):
        """Get face recognition results"""
        try:
            return self.recognition_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop face recognition"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Face recognizer stopped")

class IdentityTracker:
    """Tracks person identities across frames"""
    
    def __init__(self, max_missing_frames=30, similarity_threshold=0.7):
        self.max_missing_frames = max_missing_frames
        self.similarity_threshold = similarity_threshold
        self.active_persons = {}  # person_id -> DetectedPerson
        self.next_subject_id = 1
        self.tracking_queue = queue.Queue(maxsize=10)
        self.running = False
        self.thread = None
        
    def start(self, recognition_queue):
        """Start identity tracking thread"""
        self.recognition_queue = recognition_queue
        self.running = True
        self.thread = threading.Thread(target=self._track_identities, daemon=True)
        self.thread.start()
        logger.info("Identity tracker started")
    
    def _calculate_similarity(self, encoding1, encoding2):
        """Calculate similarity between two face encodings"""
        if encoding1 is None or encoding2 is None:
            return 0.0
        distance = face_recognition.face_distance([encoding1], encoding2)[0]
        return 1.0 - distance
    
    def _find_matching_person(self, face_encoding, face_location):
        """Find matching person from active persons"""
        best_match = None
        best_similarity = 0.0
        
        for person_id, person in self.active_persons.items():
            if person.face_encoding is None:
                continue
            
            similarity = self._calculate_similarity(person.face_encoding, face_encoding)
            
            # Also consider location proximity
            if face_location and person.face_location:
                loc_distance = np.sqrt(
                    (face_location[0] - person.face_location[0])**2 +
                    (face_location[1] - person.face_location[1])**2
                )
                # Normalize location distance (assuming max distance of 200 pixels)
                loc_similarity = max(0.0, 1.0 - (loc_distance / 200.0))
                
                # Combine face and location similarity
                combined_similarity = (similarity * 0.8) + (loc_similarity * 0.2)
            else:
                combined_similarity = similarity
            
            if combined_similarity > best_similarity and combined_similarity > self.similarity_threshold:
                best_similarity = combined_similarity
                best_match = person_id
        
        return best_match, best_similarity
    
    def _track_identities(self):
        """Track person identities across frames"""
        while self.running:
            try:
                recognition_data = self.recognition_queue.get(timeout=1.0)
                if recognition_data is None:
                    continue
                
                frame_number = recognition_data['frame_number']
                timestamp = recognition_data['timestamp']
                face_locations = recognition_data['face_locations']
                face_encodings = recognition_data['face_encodings']
                face_names = recognition_data['face_names']
                face_confidences = recognition_data['face_confidences']
                
                current_frame_persons = []
                used_person_ids = set()
                
                # Process each detected face
                for i, (location, encoding, name, confidence) in enumerate(
                    zip(face_locations, face_encodings, face_names, face_confidences)
                ):
                    # Try to match with existing person
                    matched_person_id, similarity = self._find_matching_person(encoding, location)
                    
                    if matched_person_id and matched_person_id not in used_person_ids:
                        # Update existing person
                        person = self.active_persons[matched_person_id]
                        person.face_location = location
                        person.face_encoding = encoding
                        person.confidence = confidence
                        person.consecutive_frames += 1
                        person.total_appearances += 1
                        person.last_seen_timestamp = timestamp
                        
                        # Update name if recognized and was unknown
                        if name != "Unknown" and not person.is_known:
                            person.display_name = name
                            person.is_known = True
                            person.person_id = name.lower().replace(' ', '_')
                        
                        used_person_ids.add(matched_person_id)
                    else:
                        # Create new person
                        if name == "Unknown":
                            person_id = f"subject_{self.next_subject_id}"
                            display_name = f"Subject {self.next_subject_id}"
                            is_known = False
                            self.next_subject_id += 1
                        else:
                            person_id = name.lower().replace(' ', '_')
                            display_name = name
                            is_known = True
                        
                        person = DetectedPerson(
                            person_id=person_id,
                            display_name=display_name,
                            is_known=is_known,
                            confidence=confidence,
                            face_location=location,
                            face_encoding=encoding,
                            first_seen_frame=frame_number,
                            consecutive_frames=1,
                            total_appearances=1,
                            last_seen_timestamp=timestamp
                        )
                        
                        self.active_persons[person_id] = person
                        used_person_ids.add(person_id)
                    
                    current_frame_persons.append(self.active_persons[list(used_person_ids)[-1]])
                
                # Update consecutive frames for persons not seen in this frame
                persons_to_remove = []
                for person_id, person in self.active_persons.items():
                    if person_id not in used_person_ids:
                        person.consecutive_frames = 0
                        # Remove persons not seen for too long
                        frames_missing = frame_number - person.first_seen_frame - person.total_appearances
                        if frames_missing > self.max_missing_frames:
                            persons_to_remove.append(person_id)
                
                # Remove old persons
                for person_id in persons_to_remove:
                    del self.active_persons[person_id]
                    logger.debug(f"Removed inactive person: {person_id}")
                
                # Create tracking result
                tracking_result = {
                    **recognition_data,
                    'tracked_persons': current_frame_persons,
                    'active_persons_count': len(self.active_persons)
                }
                
                try:
                    self.tracking_queue.put_nowait(tracking_result)
                except queue.Full:
                    try:
                        self.tracking_queue.get_nowait()
                        self.tracking_queue.put_nowait(tracking_result)
                    except queue.Empty:
                        pass
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Identity tracking error: {str(e)}")
    
    def get_tracking_results(self):
        """Get identity tracking results"""
        try:
            return self.tracking_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop identity tracking"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Identity tracker stopped")

class ResultsAggregator:
    """Aggregates and formats final results"""
    
    def __init__(self):
        self.results_queue = queue.Queue(maxsize=50)
        self.running = False
        self.thread = None
        self.statistics = {
            'total_frames_processed': 0,
            'total_persons_detected': 0,
            'known_persons_seen': set(),
            'unknown_persons_count': 0,
            'average_processing_time': 0.0,
            'fps': 0.0
        }
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        
    def start(self, tracking_queue):
        """Start results aggregation thread"""
        self.tracking_queue = tracking_queue
        self.running = True
        self.thread = threading.Thread(target=self._aggregate_results, daemon=True)
        self.thread.start()
        logger.info("Results aggregator started")
    
    def _aggregate_results(self):
        """Aggregate tracking results into final output format"""
        while self.running:
            try:
                tracking_data = self.tracking_queue.get(timeout=1.0)
                if tracking_data is None:
                    continue
                
                start_time = time.time()
                
                frame_number = tracking_data['frame_number']
                timestamp = tracking_data['timestamp']
                tracked_persons = tracking_data['tracked_persons']
                
                # Update statistics
                self.statistics['total_frames_processed'] += 1
                self.statistics['total_persons_detected'] = len(tracked_persons)
                
                known_count = 0
                unknown_count = 0
                
                # Process tracked persons
                processed_persons = []
                for person in tracked_persons:
                    if person.is_known:
                        known_count += 1
                        self.statistics['known_persons_seen'].add(person.person_id)
                    else:
                        unknown_count += 1
                    
                    # Create serializable person data (excluding numpy arrays)
                    person_data = {
                        'person_id': person.person_id,
                        'display_name': person.display_name,
                        'is_known': person.is_known,
                        'confidence': float(person.confidence),
                        'face_location': {
                            'top': int(person.face_location[0]),
                            'right': int(person.face_location[1]),
                            'bottom': int(person.face_location[2]),
                            'left': int(person.face_location[3])
                        },
                        'tracking_info': {
                            'first_seen_frame': person.first_seen_frame,
                            'consecutive_frames': person.consecutive_frames,
                            'total_appearances': person.total_appearances,
                            'last_seen_timestamp': person.last_seen_timestamp
                        }
                    }
                    processed_persons.append(person_data)
                
                self.statistics['unknown_persons_count'] = unknown_count
                
                # Calculate FPS
                self.fps_frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.statistics['fps'] = self.fps_frame_count / (current_time - self.last_fps_time)
                    self.fps_frame_count = 0
                    self.last_fps_time = current_time
                
                # Calculate processing time
                processing_time_ms = (time.time() - start_time) * 1000
                
                # Update average processing time
                total_frames = self.statistics['total_frames_processed']
                avg_time = self.statistics['average_processing_time']
                self.statistics['average_processing_time'] = (
                    (avg_time * (total_frames - 1) + processing_time_ms) / total_frames
                )
                
                # Create final result
                frame_result = FrameResult(
                    timestamp=datetime.fromtimestamp(timestamp).isoformat(),
                    frame_number=frame_number,
                    processing_time_ms=processing_time_ms,
                    detected_persons=processed_persons,
                    frame_shape=tracking_data['frame'].shape,
                    statistics={
                        'total_persons': len(processed_persons),
                        'known_persons': known_count,
                        'unknown_persons': unknown_count,
                        'average_confidence': np.mean([p['confidence'] for p in processed_persons]) if processed_persons else 0.0,
                        'fps': self.statistics['fps'],
                        'unique_known_persons': len(self.statistics['known_persons_seen'])
                    }
                )
                
                # Add frame for display
                result_with_frame = {
                    'result': frame_result,
                    'frame': tracking_data['frame'],
                    'raw_tracking_data': tracking_data
                }
                
                try:
                    self.results_queue.put_nowait(result_with_frame)
                except queue.Full:
                    try:
                        self.results_queue.get_nowait()
                        self.results_queue.put_nowait(result_with_frame)
                    except queue.Empty:
                        pass
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Results aggregation error: {str(e)}")
    
    def get_results(self):
        """Get aggregated results"""
        try:
            return self.results_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_statistics(self):
        """Get current statistics"""
        stats = self.statistics.copy()
        stats['known_persons_seen'] = list(stats['known_persons_seen'])
        return stats
    
    def stop(self):
        """Stop results aggregation"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Results aggregator stopped")

class DisplayManager:
    """Handles visual output and display"""
    
    def __init__(self, window_name="Video Pipeline", show_confidence=True, 
                 show_bounding_boxes=True, show_statistics=True):
        self.window_name = window_name
        self.show_confidence = show_confidence
        self.show_bounding_boxes = show_bounding_boxes
        self.show_statistics = show_statistics
        self.colors = {
            'known': (0, 255, 0),      # Green for known persons
            'unknown': (0, 0, 255),    # Red for unknown persons
            'text': (255, 255, 255),   # White for text
            'background': (0, 0, 0)    # Black for text background
        }
        
    def draw_results(self, frame, frame_result):
        """Draw detection results on frame"""
        display_frame = frame.copy()
        
        if self.show_bounding_boxes:
            for person in frame_result.detected_persons:
                # Get face location
                face_loc = person['face_location']
                top, right, bottom, left = face_loc['top'], face_loc['right'], face_loc['bottom'], face_loc['left']
                
                # Choose color based on known/unknown status
                color = self.colors['known'] if person['is_known'] else self.colors['unknown']
                
                # Draw bounding box
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                
                # Prepare label text
                label = person['display_name']
                if self.show_confidence:
                    label += f" ({person['confidence']:.2f})"
                
                # Calculate text size and position
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                thickness = 1
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw text background
                text_y = top - 10 if top > 30 else bottom + 25
                cv2.rectangle(display_frame, 
                            (left, text_y - text_height - 5), 
                            (left + text_width + 10, text_y + 5), 
                            color, -1)
                
                # Draw text
                cv2.putText(display_frame, label, (left + 5, text_y - 5), 
                          font, font_scale, self.colors['text'], thickness)
        
        if self.show_statistics:
            self._draw_statistics(display_frame, frame_result)
        
        return display_frame
    
    def _draw_statistics(self, frame, frame_result):
        """Draw statistics overlay on frame"""
        stats = frame_result.statistics
        
        # Prepare statistics text
        stats_text = [
            f"Frame: {frame_result.frame_number}",
            f"Total Persons: {stats['total_persons']}",
            f"Known: {stats['known_persons']} | Unknown: {stats['unknown_persons']}",
            f"FPS: {stats['fps']:.1f}",
            f"Processing: {frame_result.processing_time_ms:.1f}ms"
        ]
        
        # Draw statistics background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Calculate background size
        max_width = 0
        total_height = 0
        line_height = 30
        
        for text in stats_text:
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            max_width = max(max_width, text_width)
            total_height += line_height
        
        # Draw background rectangle
        cv2.rectangle(frame, (10, 10), (max_width + 30, total_height + 20), 
                     (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (max_width + 30, total_height + 20), 
                     (255, 255, 255), 2)
        
        # Draw statistics text
        y_offset = 35
        for text in stats_text:
            cv2.putText(frame, text, (20, y_offset), font, font_scale, 
                       self.colors['text'], thickness)
            y_offset += line_height
    
    def show_frame(self, frame):
        """Display frame in window"""
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(1) & 0xFF

class VideoPipeline:
    """Main video pipeline orchestrator"""
    
    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.gdrive_manager = None
        self.video_capture = None
        self.face_detector = None
        self.face_recognizer = None
        self.identity_tracker = None
        self.results_aggregator = None
        self.display_manager = None
        
        # Pipeline state
        self.running = False
        self.results_callback = None
        
        # Statistics
        self.start_time = None
        self.total_frames = 0
        
    def _load_config(self, config_path):
        """Load configuration from file or use defaults"""
        default_config = {
            "video": {
                "source": 0,
                "width": 1280,
                "height": 720,
                "fps": 30
            },
            "detection": {
                "min_face_size": 40,
                "scale_factor": 0.709,
                "confidence_threshold": 0.8
            },
            "recognition": {
                "tolerance": 0.6,
                "unknown_threshold": 0.4
            },
            "tracking": {
                "max_missing_frames": 30,
                "similarity_threshold": 0.7
            },
            "gdrive": {
                "credentials_path": "credentials.json",
                "token_path": "token.json",
                "employee_folder": "employee_photos",
                "local_dir": "./employee_photos",
                "auto_download": True
            },
            "display": {
                "window_name": "Video Pipeline",
                "show_confidence": True,
                "show_bounding_boxes": True,
                "show_statistics": True
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    # Merge with defaults
                    self._deep_update(default_config, file_config)
                logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {str(e)}")
                logger.info("Using default configuration")
        else:
            logger.info("Using default configuration")
        
        return default_config
    
    def _deep_update(self, base_dict, update_dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def setup_gdrive_integration(self):
        """Setup Google Drive integration"""
        try:
            gdrive_config = self.config['gdrive']
            self.gdrive_manager = GoogleDriveManager(
                credentials_path=gdrive_config['credentials_path'],
                token_path=gdrive_config['token_path']
            )
            
            if gdrive_config['auto_download']:
                logger.info("Downloading employee photos from Google Drive...")
                success = self.gdrive_manager.download_employee_photos(
                    folder_name=gdrive_config['employee_folder'],
                    local_dir=gdrive_config['local_dir']
                )
                if success:
                    logger.info("Employee photos downloaded successfully")
                else:
                    logger.warning("Failed to download employee photos")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Google Drive integration: {str(e)}")
            return False
    
    def load_employee_database(self, folder_path=None):
        """Load employee database from local folder"""
        if folder_path is None:
            folder_path = self.config['gdrive']['local_dir']
        
        if not self.face_recognizer:
            recognition_config = self.config['recognition']
            self.face_recognizer = FaceRecognizer(
                tolerance=recognition_config['tolerance'],
                unknown_threshold=recognition_config['unknown_threshold']
            )
        
        return self.face_recognizer.load_employee_database(folder_path)
    
    def start(self, display=True, gdrive_sync=True):
        """Start the video pipeline"""
        logger.info("Starting video pipeline...")
        self.start_time = time.time()
        
        try:
            # Setup Google Drive integration
            if gdrive_sync:
                self.setup_gdrive_integration()
            
            # Load employee database
            loaded_faces = self.load_employee_database()
            logger.info(f"Loaded {loaded_faces} face encodings")
            
            # Initialize components
            video_config = self.config['video']
            self.video_capture = VideoCapture(
                source=video_config['source'],
                width=video_config['width'],
                height=video_config['height'],
                fps=video_config['fps']
            )
            
            detection_config = self.config['detection']
            self.face_detector = FaceDetector(
                min_face_size=detection_config['min_face_size'],
                scale_factor=detection_config['scale_factor']
            )
            
            tracking_config = self.config['tracking']
            self.identity_tracker = IdentityTracker(
                max_missing_frames=tracking_config['max_missing_frames'],
                similarity_threshold=tracking_config['similarity_threshold']
            )
            
            self.results_aggregator = ResultsAggregator()
            
            if display:
                display_config = self.config['display']
                self.display_manager = DisplayManager(
                    window_name=display_config['window_name'],
                    show_confidence=display_config['show_confidence'],
                    show_bounding_boxes=display_config['show_bounding_boxes'],
                    show_statistics=display_config['show_statistics']
                )
            
            # Start pipeline components
            self.video_capture.start()
            self.face_detector.start(self.video_capture.frame_queue)
            self.face_recognizer.start(self.face_detector.detection_queue)
            self.identity_tracker.start(self.face_recognizer.recognition_queue)
            self.results_aggregator.start(self.identity_tracker.tracking_queue)
            
            self.running = True
            logger.info("Video pipeline started successfully")
            
            # Main processing loop
            self._run_pipeline(display)
            
        except Exception as e:
            logger.error(f"Failed to start video pipeline: {str(e)}")
            self.stop()
            raise
    
    def _run_pipeline(self, display=True):
        """Main pipeline processing loop"""
        logger.info("Pipeline processing started. Press 'q' to quit.")
        
        try:
            while self.running:
                # Get results from aggregator
                result_data = self.results_aggregator.get_results()
                
                if result_data:
                    frame_result = result_data['result']
                    frame = result_data['frame']
                    
                    self.total_frames += 1
                    
                    # Call results callback if set
                    if self.results_callback:
                        self.results_callback(frame_result)
                    
                    # Display results
                    if display and self.display_manager:
                        display_frame = self.display_manager.draw_results(frame, frame_result)
                        key = self.display_manager.show_frame(display_frame)
                        
                        # Handle key presses
                        if key == ord('q'):
                            logger.info("Quit key pressed")
                            break
                        elif key == ord('s'):
                            # Save screenshot
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"screenshot_{timestamp}.jpg"
                            cv2.imwrite(filename, display_frame)
                            logger.info(f"Screenshot saved: {filename}")
                        elif key == ord('p'):
                            # Print statistics
                            stats = self.results_aggregator.get_statistics()
                            logger.info(f"Pipeline Statistics: {json.dumps(stats, indent=2)}")
                
                else:
                    # No results available, small delay
                    time.sleep(0.01)
                    
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
        except Exception as e:
            logger.error(f"Pipeline processing error: {str(e)}")
        finally:
            self.stop()
    
    def set_results_callback(self, callback):
        """Set callback function for processing results"""
        self.results_callback = callback
    
    def get_statistics(self):
        """Get pipeline statistics"""
        if self.results_aggregator:
            stats = self.results_aggregator.get_statistics()
            
            # Add pipeline-level statistics
            if self.start_time:
                runtime = time.time() - self.start_time
                stats['pipeline_runtime_seconds'] = runtime
                stats['total_frames_captured'] = self.total_frames
                stats['average_fps'] = self.total_frames / runtime if runtime > 0 else 0
            
            return stats
        return {}
    
    def add_person_to_database(self, person_id, name, face_encoding):
        """Add a new person to the recognition database"""
        if self.face_recognizer:
            self.face_recognizer.known_encodings.append(face_encoding)
            self.face_recognizer.known_names.append(name)
            logger.info(f"Added {name} to face recognition database")
            return True
        return False
    
    def save_database(self, filepath="face_database.pkl"):
        """Save current face database to file"""
        if self.face_recognizer:
            database = {
                'encodings': self.face_recognizer.known_encodings,
                'names': self.face_recognizer.known_names,
                'saved_at': datetime.now().isoformat()
            }
            
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(database, f)
                logger.info(f"Face database saved to {filepath}")
                return True
            except Exception as e:
                logger.error(f"Failed to save database: {str(e)}")
        return False
    
    def load_database(self, filepath="face_database.pkl"):
        """Load face database from file"""
        if not os.path.exists(filepath):
            logger.warning(f"Database file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                database = pickle.load(f)
            
            if self.face_recognizer:
                self.face_recognizer.known_encodings = database['encodings']
                self.face_recognizer.known_names = database['names']
                logger.info(f"Face database loaded from {filepath}")
                logger.info(f"Loaded {len(database['encodings'])} face encodings")
                return True
            
        except Exception as e:
            logger.error(f"Failed to load database: {str(e)}")
        return False
    
    def stop(self):
        """Stop the video pipeline"""
        logger.info("Stopping video pipeline...")
        self.running = False
        
        # Stop all components
        if self.results_aggregator:
            self.results_aggregator.stop()
        if self.identity_tracker:
            self.identity_tracker.stop()
        if self.face_recognizer:
            self.face_recognizer.stop()
        if self.face_detector:
            self.face_detector.stop()
        if self.video_capture:
            self.video_capture.stop()
        
        # Close display windows
        cv2.destroyAllWindows()
        
        # Print final statistics
        stats = self.get_statistics()
        if stats:
            logger.info("Final Pipeline Statistics:")
            logger.info(json.dumps(stats, indent=2, default=str))
        
        logger.info("Video pipeline stopped")

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-Time Video Pipeline with Face Recognition')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--input', type=str, help='Video input source (default: webcam)')
    parser.add_argument('--no-display', action='store_true', help='Run without display')
    parser.add_argument('--no-gdrive', action='store_true', help='Skip Google Drive sync')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--save-db', type=str, help='Save face database to file')
    parser.add_argument('--load-db', type=str, help='Load face database from file')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create pipeline
    pipeline = VideoPipeline(config_path=args.config)
    
    # Override video source if specified
    if args.input:
        if args.input.isdigit():
            pipeline.config['video']['source'] = int(args.input)
        else:
            pipeline.config['video']['source'] = args.input
    
    # Load database if specified
    if args.load_db:
        pipeline.load_database(args.load_db)
    
    try:
        # Start pipeline
        pipeline.start(
            display=not args.no_display,
            gdrive_sync=not args.no_gdrive
        )
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
    finally:
        # Save database if specified
        if args.save_db:
            pipeline.save_database(args.save_db)
        
        pipeline.stop()

if __name__ == "__main__":
    main()
