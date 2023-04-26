# FacePy Bird 2.0

FacePy Bird is an OpenCV-based program that allows the user to play Flappy Bird using head nods instead of traditional keyboard inputs. The program uses [Mediapipe](https://developers.google.com/mediapipe) to estimate the user's head pose, and maps specific head gestures to the game controls.

To start the program, navigate to the root directory and run the python main.py command. Make sure that your webcam is properly connected and positioned before running the program. The game will begin automatically, and you can use head nods to control the bird's movement through the pipes.

The program uses a simple threshold-based gesture detection algorithm, which may require some tuning depending on the lighting conditions and user posture.

![Example](out.gif)

## Usage
To run this project,
1. Install Python 3.11 or above from [here](https://www.python.org/download/releases/)
2. Install [Mediapipe](https://developers.google.com/mediapipe) and its dependencies.
3. Clone the repository:
    ```
    git clone https://github.com/victor-hugo-dc/FacePyBird2.0.git
    ```
    or download as a zip file and extract.
4. To install all dependencies, run:
    ```
    pip install -r requirements.txt
    ```
5. In the root directory, run:
    ```
    python3 main.py
    ```
6. Use <kbd>Q</kbd> to close the game.