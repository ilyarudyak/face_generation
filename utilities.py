

def scale(x, feature_range=(-1, 1)):
    """ Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1.
       This function assumes that the input x is already scaled from 0-1."""
    # assume x is scaled to (0, 1)
    return 2. * x - 1.
