# AI-Based Yoga Trainer with Traditional Indian Asana Names

## Project Overview

An intelligent AI-powered Yoga Assistant that serves as a personal trainer capable of real-time yoga pose recognition, ambiguity detection, corrective feedback, and personalized voice guidance using traditional Sanskrit asana names.

## Key Features

- 🧘‍♀️ **Real-time Pose Detection**: Using MediaPipe for accurate pose estimation
- 🎯 **Ambiguity Detection**: Handles variations in posture and body types
- 🗣️ **Voice Feedback**: Traditional Sanskrit names with corrective guidance
- 👁️ **Webcam Integration**: Browser-based real-time video processing
- 📊 **Trainer Dashboard**: Complete control panel for instructors
- 🌐 **Web Interface**: Modern React-based frontend

## Traditional Asanas Supported

- ताड़ासन (Tadasana) - Mountain Pose
- वृक्षासन (Vrikshasana) - Tree Pose
- त्रिकोणासन (Trikonasana) - Triangle Pose
- भुजंगासन (Bhujangasana) - Cobra Pose
- पश्चिमोत्तानासन (Paschimottanasana) - Seated Forward Bend
- शलभासन (Shalabhasana) - Locust Pose
- And many more...

## Project Structure

```
yoga_ai_trainer/
├── backend/
│   ├── api/           # FastAPI endpoints
│   ├── models/        # ML models and pose detection
│   ├── utils/         # Utility functions
│   └── data/          # Data processing and storage
├── frontend/
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── pages/       # Main pages
│   │   └── utils/       # Frontend utilities
│   └── public/          # Static assets
├── database/            # Database schemas and migrations
├── docs/               # Documentation
└── tests/              # Test files
```

## Installation

1. Create and activate virtual environment:
```bash
python -m venv yoga_env
source yoga_env/bin/activate  # On Windows: yoga_env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

1. Start the backend server:
```bash
python backend/main.py
```

2. Start the frontend (in a new terminal):
```bash
cd frontend
npm install
npm start
```

3. Open browser and navigate to `http://localhost:3000`

## Development

- **Backend**: FastAPI with Python 3.8+
- **Frontend**: React.js with WebRTC
- **ML Framework**: TensorFlow + MediaPipe
- **Database**: SQLAlchemy with SQLite/PostgreSQL
- **Voice**: gTTS + pyttsx3 for multilingual support

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Traditional yoga practices and Sanskrit terminology
- MediaPipe team for pose estimation
- Open source yoga community
