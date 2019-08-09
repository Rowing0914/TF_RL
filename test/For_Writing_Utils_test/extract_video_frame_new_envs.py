import cv2

ENVS = [
    "ant_with_goal",
    "centipedeFour",
    "centipedeSix"
]

for env_name in ENVS:
    print(env_name)
    vidcap = cv2.VideoCapture("./videos/{}.mp4".format(env_name))
    success, image = vidcap.read()

    if success:
        for i in range(100):
            if i == 10:
                cv2.imwrite("{}.jpg".format(env_name), image)
                break
            success, image = vidcap.read()
            # print('Read a new frame: ', success)
    else:
        assert False, "failed to open the file"
