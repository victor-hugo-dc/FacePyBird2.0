import cv2
import numpy as np
import random
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import mediapipe as mp

from constants import *
from ground import Ground
from bird import Bird
from score import Score
from message import Message
from scoreboard import ScoreBoard

import playsound

class FlappyBird:

    def __init__(self) -> None:
        self.window = 'FacePy Bird by @victor-hugo-dc'

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.segmentor = SelfiSegmentation()
        self.backgrounds = [
            cv2.imread(f'assets/sprites/background-day.png'),
            cv2.imread(f'assets/sprites/background-night.png')
        ]
        self.capture = cv2.VideoCapture(1)

        self.message: Message = Message()

        self.score_drawer: Score = Score()
        self.reset_variables()

        self.get_highscore()

        self.start: bool = True

        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # self.out = cv2.VideoWriter('./out.mp4', fourcc, 20.0, (SCREEN_WIDTH, SCREEN_HEIGHT))
    
    def reset_variables(self):
        self.background = random.choice(self.backgrounds)
        self.ground_group: list[Ground] = [Ground(GROUND_WIDTH * i) for i in range(2)]
        
        self.pipe_group: list[Pipe] = []
        for i in range (2):
            pipes = get_random_pipes(SCREEN_WIDTH * i + 800)
            self.pipe_group.extend(pipes)
        
        self.bird: Bird = Bird()
        self.acceleration = FLAP_ACCELERATION

        self.score = 0

        self.forehead_landmark = None
        self.nose_landmark = None

        self.scoreboard = ScoreBoard()

    def resize(self):
        scale = 0.75
        height, width, _ = self.frame.shape
        height, width = int(height * scale), int(width * scale)
        self.frame = cv2.resize(self.frame, (width, height), interpolation = cv2.INTER_AREA)
    
    
    def get_pitch(self):
        image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.face_mesh.process(image)

        face_3d, face_2d = [], []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks[:1]:

                self.forehead_landmark = face_landmarks.landmark[151]
                self.nose_landmark = face_landmarks.landmark[2]

                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        x, y = int(lm.x * SCREEN_WIDTH), int(lm.y * SCREEN_HEIGHT)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])       
                
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * SCREEN_WIDTH

                cam_matrix = np.array([ [focal_length, 0, SCREEN_HEIGHT / 2],
                                        [0, focal_length, SCREEN_WIDTH / 2],
                                        [0, 0, 1]])

                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                return angles[0] * 360
            
        return None
    
    def get_highscore(self):
        with open('assets/highscore.txt', 'r') as f:
            self.highscore = int(f.read().strip())
    
    def set_highscore(self):
        with open('assets/highscore.txt', 'w') as f:
            f.write(str(self.highscore))
    
    def remove_background(self):
        height, width, _ = self.frame.shape
        center_x, center_y = width // 2, height // 2
        xoff, yoff = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        self.frame = self.frame[center_y - yoff : center_y + yoff, center_x - xoff : center_x + xoff]
        self.frame = self.segmentor.removeBG(self.frame, self.background, threshold = 0.8)
    
    def update_frame(self):
        _, self.frame = self.capture.read()
        self.frame = cv2.flip(self.frame, 1)
        self.resize()
        self.remove_background()
    
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
    
    def show_pipes(self):
        if is_off_screen(self.pipe_group[0]):
            self.pipe_group = self.pipe_group[2:]
            pipes = get_random_pipes(SCREEN_WIDTH * 2)
            self.pipe_group.extend(pipes)

        for pipe in self.pipe_group:
            self.overlay(pipe.pipe, pipe.x, pipe.y)
            pipe.update()
    
    def show_ground(self):
        if is_off_screen(self.ground_group[0]):
            self.ground_group.pop(0)
            ground = Ground(self.ground_group[-1].x + self.ground_group[-1].width)
            self.ground_group.append(ground)

        for ground in self.ground_group:
            self.overlay(ground.base, ground.x, ground.y)
            ground.update()
            
    def show_bird(self, gameover: bool = False):
        self.overlay(self.bird.player, self.bird.x, self.bird.y)
        self.bird.update(gameover)
    
    def show_score(self):
        if not self.forehead_landmark:
            return
        
        x, y = int(self.forehead_landmark.x * SCREEN_WIDTH), int(self.forehead_landmark.y * SCREEN_HEIGHT)
        points, _, xoff, yoff = self.score_drawer.score(self.score, x, y)
        for no in points:
            number: np.ndarray = self.score_drawer.numbers[no]

            self.overlay(number, xoff, yoff)
            xoff += number.shape[1]
    
    def show_message(self):
        if not self.nose_landmark:
            return
        
        x, y = int(self.nose_landmark.x * SCREEN_WIDTH), int(self.nose_landmark.y * SCREEN_HEIGHT)
        x, y = self.message.message(x, y)
        self.overlay(self.message.image, x, y)
    
    def show_scoreboard(self, scoreboard):
        _, width, _ = scoreboard.shape
        x, y = int(self.forehead_landmark.x * SCREEN_WIDTH) - (width // 2), int(self.forehead_landmark.y * SCREEN_HEIGHT)
        self.overlay(scoreboard, x, y)

    def check_nod(self):
        pitch = self.get_pitch() or 0
        if pitch <= MIN_PITCH_THRESHOLD:
            self.acceleration = FLAP_ACCELERATION
        
        elif pitch >= PITCH_THRESHOLD:
            self.bird.bump(self.acceleration)
            self.acceleration = None
    
    def check_collision(self):

        if self.bird.y <= 0 or self.bird.y + self.bird.height >= self.ground_group[0].y:
            # bird is out of bounds
            return True

        for pipe in self.pipe_group:
            if pipe.x <= self.bird.x + self.bird.width <= pipe.x + pipe.width:
                # check collision with top pipe
                if pipe.y <= self.bird.y <= pipe.y + pipe.height:
                    return True

                # check collision with bottom pipe
                if pipe.y <= self.bird.y + self.bird.height <= pipe.y + pipe.height:
                    return True
        
        return False

    def check_score(self):
        for pipe in self.pipe_group[::2]:
            past_midpoint: bool = self.bird.x + (self.bird.width // 2) >= pipe.x + (pipe.width // 2)
            if not pipe.scored and past_midpoint:
                self.score += 1
                pipe.scored = True
                playsound.playsound(POINT, False)
    
    def intro(self) -> bool:
        while self.capture.isOpened():

            self.update_frame()

            pitch = self.get_pitch() or 0
            if self.start and pitch >= PITCH_THRESHOLD:
                self.start = False
                return True
            
            elif pitch <= MIN_PITCH_THRESHOLD:
                self.start = True

            self.show_message()
            self.show_ground()
            cv2.imshow(self.window, self.frame)
            # self.out.write(self.frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                return False

    def main(self) -> bool:

        while self.capture.isOpened():

            if self.check_collision():
                playsound.playsound(HIT, False)
                playsound.playsound(DIE, False)
                return True

            self.update_frame()
            self.check_nod()
            self.bird.update_speed()
            self.show_pipes()
            self.check_score()
            self.show_score()
            self.show_ground()
            self.show_bird()
            cv2.imshow(self.window, self.frame)
            # self.out.write(self.frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                return False
    
    def gameover(self) -> bool:
        self.bird.speed = VELOCITY_MAX
        self.highscore = max(self.score, self.highscore)
        scoreboard = self.scoreboard.create_scoreboard(self.score, self.highscore)

        while self.capture.isOpened():

            self.update_frame()

            pitch = self.get_pitch() or 0
            if self.start and pitch >= PITCH_THRESHOLD:
                self.start = False
                self.reset_variables()
                return True
            
            elif pitch <= MIN_PITCH_THRESHOLD:
                self.start = True
            
            for ground in self.ground_group:
                ground.game_speed = 0
            
            for pipe in self.pipe_group:
                pipe.game_speed = 0

            self.show_scoreboard(scoreboard)
            self.show_pipes()
            self.show_ground()

            self.show_bird(True)
            cv2.imshow(self.window, self.frame)
            # self.out.write(self.frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        
        self.set_highscore()
        cv2.destroyAllWindows()
        return False