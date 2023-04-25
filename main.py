from flappybird import FlappyBird

if __name__ == '__main__':
    fb = FlappyBird()
    while fb.intro() \
        and fb.main() \
        and fb.gameover():
        continue