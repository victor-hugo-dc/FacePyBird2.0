import cv2
import numpy as np
from constants import *
import playsound

class Bird:
    def __init__(self) -> None:
        self.red_bird: list = [
            cv2.imread('assets/sprites/redbird-upflap.png', cv2.IMREAD_UNCHANGED),
            cv2.imread('assets/sprites/redbird-midflap.png', cv2.IMREAD_UNCHANGED),
            cv2.imread('assets/sprites/redbird-downflap.png', cv2.IMREAD_UNCHANGED)
        ]

        self.blue_bird: list = [
            cv2.imread('assets/sprites/bluebird-upflap.png', cv2.IMREAD_UNCHANGED),
            cv2.imread('assets/sprites/bluebird-midflap.png', cv2.IMREAD_UNCHANGED),
            cv2.imread('assets/sprites/bluebird-downflap.png', cv2.IMREAD_UNCHANGED)
        ]

        self.yellow_bird: list = [
            cv2.imread('assets/sprites/yellowbird-upflap.png', cv2.IMREAD_UNCHANGED),
            cv2.imread('assets/sprites/yellowbird-midflap.png', cv2.IMREAD_UNCHANGED),
            cv2.imread('assets/sprites/yellowbird-downflap.png', cv2.IMREAD_UNCHANGED)
        ]  

        self.players = random.choice([self.red_bird, self.blue_bird, self.yellow_bird])

        self.current_player: int = 0
        self.player: np.ndarray = self.players[self.current_player]
        self.speed: int = SPEED

        self.x: int = SCREEN_WIDTH // 6
        self.y: int = SCREEN_HEIGHT // 3

        self.height, self.width, _ = self.player.shape
    
    def update(self) -> None:
        self.current_player = (self.current_player + 1) % 3
        self.player = self.players[self.current_player]

        self.y += self.speed
    
    def update_speed(self) -> None:
        if self.speed < VELOCITY_MAX:
            self.speed += GRAVITY
    
    def bump(self, acceleration) -> None:
        if acceleration:
            playsound.playsound(WING, False)
            self.speed = acceleration
    
    def begin(self):
        self.current_player = (self.current_player + 1) % 3
        self.player = self.players[self.current_player]