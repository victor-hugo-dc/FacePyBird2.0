import cv2

class Message:
    def __init__(self) -> None:
        self.image = cv2.imread('assets/sprites/message.png', -1)
        self.image = cv2.resize(self.image, (135, 180), interpolation = cv2.INTER_AREA)
        self.height, self.width, _ = self.image.shape
    
    def message(self, x: int, y: int):
        xoff, yoff = x - (self.width // 2), y - (self.height // 2)
        return xoff, yoff
