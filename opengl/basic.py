import os

import moderngl
import numpy as np
import imageio

from common.utils import source

W = 512
H = 512
X = W
Y = 1
Z = 1

CONSTS = {
    'W': W,
    'H': H,
    'X': X + 1,
    'Y': Y,
    'Z': Z,
}

FRAMES = 50
SOURCE_PATH = './median_5x5.comp'
OUTPUT_PATH = './output'

if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    context = moderngl.create_standalone_context(require=430)
    compute_shader = context.compute_shader(source(SOURCE_PATH, CONSTS))

    buffer_a_data = np.random.uniform(0.0, 1.0, (H, W, 4)).astype('f4')
    buffer_a = context.buffer(buffer_a_data)
    buffer_b_data = np.zeros((H, W, 4)).astype('f4')
    buffer_b = context.buffer(buffer_b_data)

    imgs = []
    last_buffer = buffer_b
    print('starting computation...')
    for i in range(FRAMES):
        toggle = i % 2 != 0
        buffer_a.bind_to_storage_buffer(1 if toggle else 0)
        buffer_b.bind_to_storage_buffer(0 if toggle else 1)

        last_buffer = buffer_a if toggle else buffer_b

        compute_shader.run(group_x=H, group_y=1)

        output = np.frombuffer(last_buffer.read(), dtype=np.float32)
        output = output.reshape((H, W, 4))
        output = np.multiply(output, 255).astype(np.uint8)
        imgs.append(output)
    print('...done')

    print('writing output to a file')
    imageio.mimwrite(OUTPUT_PATH + '/basic.gif', imgs, 'GIF', duration=0.15)
    print('...done')
