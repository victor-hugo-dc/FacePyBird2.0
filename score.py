import cv2
import numpy as np

class Score:
    def __init__(self) -> None:
        self.numbers: list = [cv2.imread(f'./assets/sprites/{i}.png', -1) for i in range(10)]
    
    def score(self, score: int, x: int, y: int):

        points: list = [int(i) for i in str(score)]
        width: int = np.sum([self.numbers[i].shape[1] for i in points])
        xoff, yoff = x - (width // 2), y - (self.numbers[0].shape[0] // 2)

        return points, width, xoff, yoff
