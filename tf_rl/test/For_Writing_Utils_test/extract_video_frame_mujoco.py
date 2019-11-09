import cv2

ENVS = [
    "ant",
    "halfcheetah",
    "hopper",
    "humanoid",
    "walker2d"
]

for env_name in ENVS:
    print(env_name)
    vidcap = cv2.VideoCapture("../../logs/video/Thesis/{}.mp4".format(env_name))
    success, image = vidcap.read()

    if success:
        for i in range(100):
            if i == 40:
                cv2.imwrite("./images/{}.jpg".format(env_name), image)
            success, image = vidcap.read()
            # print('Read a new frame: ', success)
    else:
        assert False, "failed to open the file"
