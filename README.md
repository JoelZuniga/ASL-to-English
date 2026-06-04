# ASL-to-English
A computer vision and machine learning project that aims to translate American Sign Language (ASL) into English using real-time hand tracking, landmark extraction, and gesture classification. 

## Project overview 
The goal of this project is to build an ASL recognition pipeline capable of: 

- Detecting hands from a live webcam feed 
- Extracting hand landmarks using MediaPipe tasks
- Generating landmark datasets from labeled images
- Training machine learning models for static ASL letters 
- Performing real-time ASL letter prediction 
- Displaying predictions and confidence scores during live use
- Currently only supports the entire alphabet except J and Z (Coming soon)

## Installation 
Clone the repository: \
`git clone <repository-url>` \
`cd ASL-to-English` 

### Python version 
This project currently targets * Python 3.9 *. 

Create the virtual environment: \
`python3.9 -m venv .venv` 

Activate the virtual environment: \ 

### macOS/Linux 
`source .venv/bin/activate` 

### Windows
`.venv\Scripts\activate`

### Install dependencies 
`pip install -r requirements.txt` 

## Model Setup 
Download the MediaPupe Hand Landmarker model: \
[Hand Landmarker Model](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)

Place the model file inside: \
`/ASL-to-English/models/`

Download a pre-trained model: \ 
[Pre-trained ASL Aphabet model](https://drive.google.com/drive/folders/1MXLRGwxwDtJfxkDqF2k3gZkaW9rwUgvA?usp=drive_link)

## Running the Application 
From the project root: \
`python -m src.capture` 
Press `ESC` to close the application. 

## Development 
Run Ruff checks: \
`ruff check .` 

Automatically fix Ruff issues: \
`ruff check . --fix` 

Format code: \
`ruff format .`

Run type checking: \
`mypy src` 

## Roadmap 
### Completed 
- project setup
- MediaPipe Tasks integration 
- Real-time hand tracking 
- Landmark visualization
- Dataset extraction pipeline 
- Feature engineering 
- Random Forst training pipeline 
- Real-time ASL letter classification 
- Prediction confidence display 

### Planned
- Dynamic letters (J and Z)
- Word recognition 
- Sentence construction
- English translation output 
- Sequence models 

