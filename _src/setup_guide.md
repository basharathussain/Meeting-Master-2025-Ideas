
# Video Pipeline Setup Guide

## 🚀 Quick Setup Instructions

### 1. System Prerequisites

**Python Requirements:**
- Python 3.9 or higher
- pip package manager

**System Requirements:**
- Webcam or video input device
- 4GB RAM minimum, 8GB recommended
- Multi-core CPU (Intel i5 or AMD Ryzen 5 equivalent)

### 2. Installation Steps

```bash
# 1. Clone or download the project
mkdir video_pipeline_mvp
cd video_pipeline_mvp

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import cv2, face_recognition, mtcnn; print('✅ All libraries installed successfully!')"
```

### 3. Google Drive API Setup

**Step 1: Create Google Cloud Project**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Google Drive API

**Step 2: Create Credentials**
1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth 2.0 Client IDs"
3. Choose "Desktop application"
4. Download the JSON file and rename it to `credentials.json`
5. Place it in your project directory

**Step 3: Setup Google Drive Folder Structure**

Create this folder structure in your Google Drive:

```
Google Drive/
└── employee_photos/
    ├── john_smith/
    │   ├── john_1.jpg
    │   ├── john_2.jpg
    │   └── john_3.jpg
    ├── sarah_johnson/
    │   ├── sarah_profile.jpg
    │   └── sarah_meeting.jpg
    └── mike_chen/
        ├── mike_formal.jpg
        └── mike_casual.jpg
```

**Photo Guidelines:**
- **Format**: JPG, PNG, or BMP
- **Resolution**: 300x300 pixels minimum
- **Quality**: Clear, well-lit face photos
- **Quantity**: 2-5 photos per person
- **Naming**: Folder name = person's name (use underscores for spaces)

### 4. Configuration

Edit `config.json` to match your setup:

```json
{
  "video": {
    "source": 0,           // 0 for webcam, or path to video file
    "width": 1280,
    "height": 720,
    "fps": 30
  },
  "gdrive": {
    "employee_folder": "employee_photos",  // Your Google Drive folder name
    "local_dir": "./employee_photos",      // Local download directory
    "auto_download": true                  // Auto-sync on startup
  }
}
```

### 5. First Run

```bash
# Run with all features
python video_pipeline.py

# Run with custom configuration
python video_pipeline.py --config config.json

# Run without Google Drive sync
python video_pipeline.py --no-gdrive

# Run without display (headless)
python video_pipeline.py --no-display

# Debug mode
python video_pipeline.py --debug
```

## 🎮 Usage Instructions

### Keyboard Controls
- **'q'**: Quit the application
- **'s'**: Save screenshot with annotations
- **'p'**: Print current statistics to console

### Real-time Output
The system will display:
- **Green boxes**: Known employees with names
- **Red boxes**: Unknown persons labeled as "Subject 1", "Subject 2", etc.
- **Statistics overlay**: Frame count, person count, FPS, processing time

### Managing the Database

**Add New Person:**
```python
from video_pipeline import VideoPipeline

pipeline = VideoPipeline()
# When you see "Subject 1" and want to add them as "Alice Cooper"
pipeline.add_person_to_database("subject_1", "Alice Cooper", face_encoding)
pipeline.save_database("updated_database.pkl")
```

## 🔧 Troubleshooting

### Common Issues

**1. Camera Not Detected**
```bash
# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera available:', cap.isOpened()); cap.release()"

# Try different camera indices
python video_pipeline.py --input 1  # Try camera index 1
python video_pipeline.py --input 2  # Try camera index 2
```

**2. Google Drive Authentication Issues**
- Ensure `credentials.json` is in the project directory
- Delete `token.json` and re-authenticate
- Check Google Cloud Console for API quotas
- Verify Google Drive API is enabled

**3. Face Recognition Errors**
```bash
# Reinstall face_recognition with proper dependencies
pip uninstall face-recognition dlib
pip install cmake
pip install dlib --verbose
pip install face-recognition
```

**4. Performance Issues**
```bash
# Reduce video resolution
python video_pipeline.py --config config_low_res.json

# Use GPU acceleration (if available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**5. Memory Issues**
- Reduce buffer sizes in config.json
- Lower video resolution
- Close other applications

### Debug Mode
```bash
# Enable detailed logging
python video_pipeline.py --debug

# Check log file
tail -f video_pipeline.log
```

## 📊 Performance Optimization

### Hardware Recommendations

**Minimum Setup:**
- Intel i5-8400 / AMD Ryzen 5 2600
- 8GB RAM
- Integrated graphics
- USB 2.0 webcam
- Expected: 15-20 FPS, 80ms latency

**Recommended Setup:**
- Intel i7-10700K / AMD Ryzen 7 3700X
- 16GB RAM
- NVIDIA GTX 1660 or better
- USB 3.0 high-quality webcam
- Expected: 30-45 FPS, 40ms latency

**High-Performance Setup:**
- Intel i9-11900K / AMD Ryzen 9 5900X
- 32GB RAM
- NVIDIA RTX 3070 or better
- Professional camera setup
- Expected: 60+ FPS, <25ms latency

### Software Optimization

**Config Tuning for Performance:**
```json
{
  "video": {
    "width": 640,      // Reduce from 1280
    "height": 480,     // Reduce from 720
    "fps": 15          // Reduce from 30
  },
  "detection": {
    "min_face_size": 60,     // Increase from 40
    "scale_factor": 0.8      // Increase from 0.709
  },
  "performance": {
    "frame_skip": 2,         // Process every 2nd frame
    "detection_scale": 0.5,  // Resize for detection
    "batch_processing": true // Enable batching
  }
}
```

## 🔒 Security & Privacy

### Data Protection
- All processing happens locally
- Employee photos stored locally after download
- No data sent to external services (except Google Drive sync)
- Face encodings encrypted at rest

### Privacy Best Practices
1. **Obtain explicit consent** before processing employee photos
2. **Regular database audits** - remove departed employees
3. **Access control** - limit who can run the system
4. **Data retention policies** - define how long to keep face data
5. **Compliance** - ensure GDPR/local privacy law compliance

### GDPR Compliance Features
```python
# Delete specific person from database
pipeline.remove_person("john_smith")

# Export person's data
person_data = pipeline.export_person_data("john_smith")

# Clear all data
pipeline.clear_database()
```

## 📈 Monitoring & Analytics

### Built-in Statistics
The system tracks:
- Total frames processed
- Detection accuracy rates
- Recognition confidence scores
- Processing latency
- Person appearance duration
- Unique person count

### Accessing Statistics
```python
# Get real-time statistics
stats = pipeline.get_statistics()
print(json.dumps(stats, indent=2))

# Save statistics to file
with open('pipeline_stats.json', 'w') as f:
    json.dump(stats, f, indent=2, default=str)
```

## 🔄 Integration Guide

### API Usage
```python
from video_pipeline import VideoPipeline

# Initialize pipeline
pipeline = VideoPipeline(config_path="config.json")

# Set up callback for results
def process_results(frame_result):
    print(f"Frame {frame_result.frame_number}: {len(frame_result.detected_persons)} persons")
    for person in frame_result.detected_persons:
        print(f"  - {person['display_name']} (confidence: {person['confidence']:.2f})")

pipeline.set_results_callback(process_results)

# Start pipeline (non-blocking)
pipeline.start(display=False)
```

### REST API Integration
```python
from flask import Flask, jsonify
from video_pipeline import VideoPipeline

app = Flask(__name__)
pipeline = VideoPipeline()

@app.route('/api/status')
def get_status():
    return jsonify(pipeline.get_statistics())

@app.route('/api/persons')
def get_current_persons():
    # Get latest frame results
    return jsonify(latest_results)

if __name__ == '__main__':
    pipeline.start(display=False)
    app.run(host='0.0.0.0', port=5000)
```

### Database Integration
```python
import sqlite3
from datetime import datetime

def store_detection_results(frame_result):
    conn = sqlite3.connect('detections.db')
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            frame_number INTEGER,
            person_id TEXT,
            person_name TEXT,
            is_known BOOLEAN,
            confidence REAL,
            face_location TEXT
        )
    ''')
    
    # Insert detection data
    for person in frame_result.detected_persons:
        cursor.execute('''
            INSERT INTO detections 
            (timestamp, frame_number, person_id, person_name, is_known, confidence, face_location)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            frame_result.timestamp,
            frame_result.frame_number,
            person['person_id'],
            person['display_name'],
            person['is_known'],
            person['confidence'],
            str(person['face_location'])
        ))
    
    conn.commit()
    conn.close()

# Use with pipeline
pipeline.set_results_callback(store_detection_results)
```

## 📱 Advanced Features

### Multi-Camera Support
```python
# Initialize multiple pipelines for different cameras
pipeline_1 = VideoPipeline()
pipeline_1.config['video']['source'] = 0  # First camera

pipeline_2 = VideoPipeline()
pipeline_2.config['video']['source'] = 1  # Second camera

# Start both pipelines
pipeline_1.start(display=False)
pipeline_2.start(display=False)
```

### Video File Processing
```python
# Process pre-recorded video
pipeline = VideoPipeline()
pipeline.config['video']['source'] = 'meeting_recording.mp4'
pipeline.start()
```

### Batch Processing
```python
import os
from pathlib import Path

def process_video_batch(video_folder, output_folder):
    pipeline = VideoPipeline()
    
    for video_file in Path(video_folder).glob('*.mp4'):
        print(f"Processing {video_file}")
        
        # Configure for this video
        pipeline.config['video']['source'] = str(video_file)
        
        # Set up result storage
        results = []
        def collect_results(frame_result):
            results.append(frame_result)
        
        pipeline.set_results_callback(collect_results)
        
        # Process video
        pipeline.start(display=False)
        
        # Save results
        output_file = Path(output_folder) / f"{video_file.stem}_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {output_file}")

# Usage
process_video_batch('./input_videos/', './output_results/')
```

## 🧪 Testing & Validation

### Unit Testing
```python
import unittest
from video_pipeline import FaceDetector, FaceRecognizer

class TestVideoPipeline(unittest.TestCase):
    
    def setUp(self):
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
    
    def test_face_detection(self):
        # Load test image
        test_image = cv2.imread('test_images/test_face.jpg')
        
        # Mock detection (replace with actual test)
        detections = self.detector._detect_faces_sync(test_image)
        
        self.assertGreater(len(detections), 0)
        self.assertIn('face_locations', detections)
    
    def test_employee_database_loading(self):
        # Test database loading
        count = self.recognizer.load_employee_database('./test_employee_photos')
        self.assertGreater(count, 0)

if __name__ == '__main__':
    unittest.main()
```

### Performance Benchmarking
```python
import time
import statistics

def benchmark_pipeline():
    pipeline = VideoPipeline()
    pipeline.start(display=False)
    
    processing_times = []
    frame_count = 0
    start_time = time.time()
    
    def measure_performance(frame_result):
        nonlocal frame_count, processing_times
        frame_count += 1
        processing_times.append(frame_result.processing_time_ms)
        
        if frame_count >= 100:  # Stop after 100 frames
            pipeline.stop()
    
    pipeline.set_results_callback(measure_performance)
    
    # Wait for completion
    while pipeline.running:
        time.sleep(0.1)
    
    total_time = time.time() - start_time
    
    print(f"Processed {frame_count} frames in {total_time:.2f} seconds")
    print(f"Average FPS: {frame_count / total_time:.2f}")
    print(f"Average processing time: {statistics.mean(processing_times):.2f}ms")
    print(f"Processing time std dev: {statistics.stdev(processing_times):.2f}ms")

benchmark_pipeline()
```

## 🚀 Deployment Guide

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "video_pipeline.py", "--no-display", "--config", "config.json"]
```

```bash
# Build and run Docker container
docker build -t video-pipeline .
docker run -p 5000:5000 -v /dev/video0:/dev/video0 --device=/dev/video0 video-pipeline
```

### Production Configuration
```json
{
  "video": {
    "source": 0,
    "width": 1920,
    "height": 1080,
    "fps": 30
  },
  "performance": {
    "enable_gpu": true,
    "batch_processing": true,
    "frame_skip": 1
  },
  "logging": {
    "level": "WARNING",
    "file": "/var/log/video_pipeline.log"
  },
  "gdrive": {
    "auto_download": false,
    "local_dir": "/opt/employee_photos"
  }
}
```

### Systemd Service (Linux)
```ini
# /etc/systemd/system/video-pipeline.service
[Unit]
Description=Video Pipeline Face Recognition Service
After=network.target

[Service]
Type=simple
User=video-pipeline
WorkingDirectory=/opt/video-pipeline
ExecStart=/opt/video-pipeline/venv/bin/python video_pipeline.py --config production_config.json --no-display
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable video-pipeline.service
sudo systemctl start video-pipeline.service
sudo systemctl status video-pipeline.service
```

## 🔍 Troubleshooting Guide

### Performance Issues

**Symptom: Low FPS (< 10 FPS)**
```bash
# Solutions:
1. Reduce video resolution in config.json
2. Increase min_face_size to reduce false detections
3. Enable frame skipping
4. Use GPU acceleration if available
```

**Symptom: High CPU usage (> 90%)**
```bash
# Solutions:
1. Reduce video FPS
2. Enable batch processing
3. Increase detection thresholds
4. Process every Nth frame only
```

**Symptom: High memory usage**
```bash
# Solutions:
1. Reduce buffer sizes in config
2. Lower video resolution
3. Clear old face encodings periodically
4. Enable garbage collection
```

### Detection Issues

**Symptom: Faces not detected**
```bash
# Solutions:
1. Lower min_face_size in config
2. Adjust MTCNN confidence thresholds
3. Improve lighting conditions
4. Check camera focus and resolution
```

**Symptom: False face detections**
```bash
# Solutions:
1. Increase confidence_threshold
2. Increase min_face_size
3. Adjust steps_threshold values
4. Add post-processing filters
```

### Recognition Issues

**Symptom: Known persons not recognized**
```bash
# Solutions:
1. Add more training photos per person
2. Improve photo quality (lighting, angle, resolution)
3. Lower recognition tolerance
4. Retrain face encodings
```

**Symptom: Too many unknown persons**
```bash
# Solutions:
1. Increase unknown_threshold
2. Improve identity tracking settings
3. Add more diverse training photos
4. Check for lighting consistency
```

## 📞 Support & Community

### Getting Help
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check the wiki for detailed guides
- **Community Forum**: Ask questions and share experiences
- **Email Support**: support@yourcompany.com

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Reporting Issues
When reporting issues, please include:
- System specifications (OS, Python version, hardware)
- Complete error messages and stack traces
- Configuration file (remove sensitive information)
- Steps to reproduce the issue
- Expected vs actual behavior

---

**🎉 Congratulations!** You've successfully set up the Video Pipeline MVP. The system is now ready to detect and recognize faces in real-time, distinguishing between known employees and unknown subjects.

For the next phase, you can integrate this video pipeline with the audio processing pipeline to create the complete meeting recorder system.
