import numpy as np


def source(uri, consts=None):
    with open(uri, 'r') as fp:
        content = fp.read()

    if consts:
        for key, value in consts.items():
            content = content.replace("__{}__".format(key), str(value))

    return content


def np_image_from_buffer(buffer, width, height, *, scale=None):
    output = np.frombuffer(buffer.read(), dtype=np.float32)
    output = output.reshape((height, width, 4))

    scale_value = 255 / scale / output_max if scale else 255
    output = np.multiply(output, scale_value).astype(np.uint8)
    return output
