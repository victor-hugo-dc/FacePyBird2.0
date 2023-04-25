import cv2
import numpy as np
from score import Score

class ScoreBoard:
    def __init__(self) -> None:
        self.scoreboard: np.ndarray = cv2.imread('assets/sprites/scoreboard.png', -1)

        self.bronze: np.ndarray = cv2.imread('assets/sprites/Bronze.png', -1)
        self.bronze: np.ndarray = cv2.resize(self.bronze, (50, 50), interpolation = cv2.INTER_AREA)

        self.silver: np.ndarray = cv2.imread('assets/sprites/Silver.png', -1)
        self.silver: np.ndarray = cv2.resize(self.silver, (50, 50), interpolation = cv2.INTER_AREA)

        self.gold: np.ndarray = cv2.imread('assets/sprites/Gold.png', -1)
        self.gold: np.ndarray = cv2.resize(self.gold, (50, 50), interpolation = cv2.INTER_AREA)
        
        self.platinum: np.ndarray = cv2.imread('assets/sprites/Platinum.png', -1)
        self.platinum: np.ndarray = cv2.resize(self.platinum, (50, 50), interpolation = cv2.INTER_AREA)
        

        # self.new: np.ndarray = cv2.imread('assets/sprites/new.png', -1)
        # self.new: np.ndarray = cv2.cvtColor(self.new, cv2.COLOR_RGB2RGBA)
        # self.new: np.ndarray = cv2.resize(self.new, (0, 0), fx = 0.6, fy = 0.6)

        self.height, self.width, _ = self.scoreboard.shape

        self.score = Score()
        self.score.numbers = [cv2.resize(self.score.numbers[i], (0, 0), fx = 0.6, fy = 0.6) for i in range(10)]

    def overlay(self, image: np.ndarray, x: int, y: int) -> None:
        y1, y2 = max(0, y), min(self.scoreboard.shape[0], y + image.shape[0])
        x1, x2 = max(0, x), min(self.scoreboard.shape[1], x + image.shape[1])
        
        y1o, y2o = max(0, -y), min(image.shape[0], self.scoreboard.shape[0] - y)
        x1o, x2o = max(0, -x), min(image.shape[1], self.scoreboard.shape[1] - x)

        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return

        channels: int = self.scoreboard.shape[2]
        alpha: float = image[y1o:y2o, x1o:x2o, 3] / 255.0
        alpha_inv: float = 1.0 - alpha

        for c in range(channels):
            self.scoreboard[y1:y2, x1:x2, c] = (alpha * image[y1o:y2o, x1o:x2o, c] + alpha_inv * self.scoreboard[y1:y2, x1:x2, c])
    
    def create_scoreboard(self, score: int):

        if score >= 40:
            self.overlay(self.platinum, 30, 110)
            
        elif score >= 30:
            self.overlay(self.gold, 30, 110)

        elif score >= 20:
            self.overlay(self.silver, 30, 110)

        elif score >= 10:
            self.overlay(self.bronze, 30, 110)

        # draw the score on the board
        points, width, _, _ = self.score.score(score, 0, 0)
        x, y = 210 - width, 103
        for i in points:
            self.overlay(self.score.numbers[i], x, y)
            x += self.score.numbers[i].shape[1]
        
        # TODO: similar logic for high score
        return self.scoreboard
