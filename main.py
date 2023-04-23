import cv2
import numpy as np
import random
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import mediapipe as mp

from constants import *
from pipe import Pipe
from ground import Ground
from bird import Bird

def is_off_screen(image):
    return image.x < -image.width

def get_random_pipes(xpos: int):
    size = random.randint(GROUND_HEIGHT + 20, 300)
    pipe = Pipe(size, xpos=xpos)
    pipe_inverted = Pipe(SCREEN_HEIGHT - size - PIPE_GAP, True, xpos=xpos)
    return [ pipe, pipe_inverted ]

class FlappyBird:

    def __init__(self) -> None:
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.segmentor = SelfiSegmentation()
        self.backgrounds = [
            cv2.imread(f'assets/sprites/background-day.png'),
            cv2.imread(f'assets/sprites/background-night.png')
        ]
        self.background = random.choice(self.backgrounds)
        self.capture = cv2.VideoCapture(1)

        self.ground_group = []
        for i in range (2):
            ground = Ground(GROUND_WIDTH * i)
            self.ground_group.append(ground)
        
        self.pipe_group = []
        for i in range (2):
            pipes = get_random_pipes(SCREEN_WIDTH * i + 800)
            self.pipe_group.append(pipes[0])
            self.pipe_group.append(pipes[1])
        
        self.bird = Bird()
        self.bump = False # whether the bird should be bumped or not

    def resize(self):
        scale_percent = 75
        width = int(self.frame.shape[1] * scale_percent / 100)
        height = int(self.frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        self.frame = cv2.resize(self.frame, dim, interpolation = cv2.INTER_AREA)
    
    def get_pitch(self):
        image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.face_mesh.process(image)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks[:1]:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])       
                
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                rmat, jac = cv2.Rodrigues(rot_vec)

                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                return angles[0] * 360
            
        return None
    

    def remove_background(self):
        height, width, _ = self.frame.shape
        center_x, center_y = width // 2, height // 2
        self.frame = self.frame[center_y - 256 : center_y + 256, center_x - 144 : center_x + 144]
        self.frame = self.segmentor.removeBG(self.frame, self.background, threshold=0.8)
    
    def overlay(self, image: np.ndarray, x: int, y: int) -> None:
        y1, y2 = max(0, y), min(self.frame.shape[0], y + image.shape[0])
        x1, x2 = max(0, x), min(self.frame.shape[1], x + image.shape[1])
        
        y1o, y2o = max(0, -y), min(image.shape[0], self.frame.shape[0] - y)
        x1o, x2o = max(0, -x), min(image.shape[1], self.frame.shape[1] - x)

        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return

        channels: int = self.frame.shape[2]
        alpha: float = image[y1o:y2o, x1o:x2o, 3] / 255.0
        alpha_inv: float = 1.0 - alpha

        for c in range(channels):
            self.frame[y1:y2, x1:x2, c] = (alpha * image[y1o:y2o, x1o:x2o, c] + alpha_inv * self.frame[y1:y2, x1:x2, c])
    
    def main(self):
        while True:
            _, self.frame = self.capture.read()
            self.frame = cv2.flip(self.frame, 1)
            
            self.resize()
            self.remove_background()

            pitch = self.get_pitch()
            if not self.bump and (pitch or 0) >= PITCH_THRESHOLD:
                self.bump = True
                # self.bird.bump()
                print("bump")
            
            elif self.bump and (pitch or 0) < MIN_PITCH_THRESHOLD:
                self.bump = False


            for pipe in self.pipe_group:
                self.overlay(pipe.pipe, pipe.x, pipe.y)
                pipe.update()
            
            if is_off_screen(self.pipe_group[0]):
                self.pipe_group = self.pipe_group[2:]
                pipes = get_random_pipes(SCREEN_WIDTH * 2)
                self.pipe_group.extend(pipes)
            
            self.overlay(self.bird.player, self.bird.x, self.bird.y)
            self.bird.update()

            for ground in self.ground_group:
                self.overlay(ground.base, ground.x, ground.y)
                ground.update()
            
            if is_off_screen(self.ground_group[0]):
                self.ground_group.pop(0)
                ground = Ground(self.ground_group[-1].x + self.ground_group[-1].width)
                self.ground_group.append(ground)


            cv2.imshow("Face-Py Bird", self.frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

if __name__ == '__main__':
    FlappyBird().main()