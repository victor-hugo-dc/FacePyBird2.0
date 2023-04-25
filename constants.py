SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512

SPEED = 1
GRAVITY = 1

VELOCITY_MAX = 5
ACCELERATION = 1
FLAP_ACCELERATION = -10
ROTATION_VELOCITY = 3
ROTATION_THRESHOLD = 20

PIPE_WIDTH = 80
PIPE_HEIGHT = 500
PIPE_GAP = 150
GAME_SPEED = 5

GROUND_WIDTH = 2 * SCREEN_WIDTH
GROUND_HEIGHT= 100

PITCH_THRESHOLD = 30
MIN_PITCH_THRESHOLD = 15

DIE = "./assets/audio/die.wav"
HIT = "./assets/audio/hit.wav"
POINT = "./assets/audio/point.wav"
SWOOSH = "./assets/audio/swoosh.wav"
WING = "./assets/audio/wing.wav"

import random
from pipe import Pipe

def is_off_screen(image):
    return image.x < -image.width

def get_random_pipes(xpos: int):
    size = random.randint(GROUND_HEIGHT + 20, 300)
    pipe = Pipe(size, xpos=xpos)
    pipe_inverted = Pipe(SCREEN_HEIGHT - size - PIPE_GAP, True, xpos=xpos)
    return [ pipe, pipe_inverted ]