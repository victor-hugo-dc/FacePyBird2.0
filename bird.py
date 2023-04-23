import cv2
import numpy as np
from constants import *

class Bird:
    def __init__(self) -> None:
        self.players: list = [
            cv2.imread('assets/sprites/bluebird-upflap.png', cv2.IMREAD_UNCHANGED),
            cv2.imread('assets/sprites/bluebird-midflap.png', cv2.IMREAD_UNCHANGED),
            cv2.imread('assets/sprites/bluebird-downflap.png', cv2.IMREAD_UNCHANGED)
        ]

        self.current_player: int = 0
        self.player: np.ndarray = self.players[self.current_player]
        self.speed: int = SPEED

        self.x: int = SCREEN_WIDTH // 6
        self.y: int = SCREEN_HEIGHT // 3

        self.width = self.player.shape[1]
    
    def update(self) -> None:
        self.current_player = (self.current_player + 1) % 3
        self.player = self.players[self.current_player]
        self.speed += GRAVITY

        self.y += self.speed
    
    def bump(self):
        self.speed = -SPEED
    
    def begin(self):
        self.current_player = (self.current_player + 1) % 3
        self.player = self.players[self.current_player]