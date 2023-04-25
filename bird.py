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
        self.center: tuple = tuple(np.array((self.width, self.height)) / 2)

        self.rotation: int = 45
        self.visible_rotation: int = 20
    
    def update(self) -> None:
        self.current_player = (self.current_player + 1) % 3
        self.player = self.players[self.current_player]

        self.y += self.speed

        self.update_rotation()

        rotation_matrix: np.ndarray = cv2.getRotationMatrix2D(self.center, self.visible_rotation, 1.0)
        self.player = cv2.warpAffine(self.player, rotation_matrix, (self.width, self.height), flags = cv2.INTER_LINEAR)

    def update_speed(self) -> None:
        if self.speed < VELOCITY_MAX:
            self.speed += GRAVITY
    
    def update_rotation(self) -> None:
        if self.rotation > -90:
            self.rotation -= ROTATION_VELOCITY
        
        self.visible_rotation = self.rotation if self.rotation <= ROTATION_THRESHOLD else ROTATION_THRESHOLD
    
    def bump(self, acceleration) -> None:
        if acceleration:
            playsound.playsound(WING, False)
            self.speed = acceleration
            self.rotation = 45