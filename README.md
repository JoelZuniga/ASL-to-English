# ASL-to-English
A computer vision and machine learning project that aims to translate American Sign Language (ASL) into English using real-time hand tracking, landmark extraction, and gesture classification. 

## Project overview 
The goal of this project is to build a ASL recognition pipeline capable of: 

- Detecting hands from a live webcam feed 
- Extracting hand landmark coordinates using MediaPipe
- Building a labeled dataset of ASL signs 
- Training machine learning models foor sign classification 
- Translating recognized signs into English text 

## Installation 

Clone the repository: \
`git clone <repository-url>` \
`cd ASL-to-English` 

Create a virtual environment: \
`python3 -m venv venv` or `python -m venv venv` 

Activate the virtual environment: \
`venv\Scripts\activate` 

Install dependencies: \
`pip install -r requirements.txt` 

(Optional) Install pre-commit hooks: \
`pre-commit install`

## MediaPipe model Setup 
Download the MediaPupe Hand Landmarker model: \
[Hand Landmarker Model](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)

Place the model file inside: \
`/ASL-to-English/models/`

## Running the Application 
From the project root: \
`python3 -m src.capture` or `python -m src.capture` 

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
- MediaPipe integration 
- Hand landmark detection 
- Real-time visualization 
- Project tooling and quality checks 

### In progress
- Landmark dataset collection 
- Data labeling pipeline 

### Planned
- ASL letter classification 
- ASL word recognition 
- Sentence construction 
- English translation output 

## License 
Work in progress 

