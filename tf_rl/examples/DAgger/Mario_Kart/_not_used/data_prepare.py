def prepare(samples):
    print("Preparing data")

    X = []
    y = []

    for sample in samples:
        print(sample)

        # load sample
        image_files, joystick_values = load_sample(sample)

        # add joystick values to y
        y.append(joystick_values)

        # load, prepare and add images to X
        for image_file in image_files:
            image = imread(image_file)
            vec = resize_image(image)
            X.append(vec)

    print("Saving to file...")
    X = np.asarray(X)
    y = np.concatenate(y)

    np.save("data/X", X)
    np.save("data/y", y)

    print("Done!")
    return