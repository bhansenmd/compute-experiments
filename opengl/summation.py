import os

import moderngl
import numpy as np
import imageio
import time

from common.utils import source, np_image_from_buffer

WIDTH = 2048
HEIGHT = 2048

WORK_GROUP_SIZE = 32
NUM_GROUPS = WIDTH * HEIGHT // WORK_GROUP_SIZE

CONSTS = {
    'WIDTH': WIDTH,
    'HEIGHT': HEIGHT,
    'WORK_GROUP_SIZE': WORK_GROUP_SIZE,
}

SOURCE_PATH = './summation.comp'
OUTPUT_PATH = './output'

if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    context = moderngl.create_standalone_context(require=430)
    compute_shader = context.compute_shader(source(SOURCE_PATH, CONSTS))

    # buffer_raw_data = np.random.uniform(0.0, 1.0, (HEIGHT * WIDTH, 4)).astype('f4')
    buffer_raw_data = np.ones((HEIGHT, WIDTH, 4)).astype('f4')
    buffer_raw = context.buffer(buffer_raw_data)
    buffer_raw.bind_to_storage_buffer(0)

    buffer_sum_data = np.zeros((HEIGHT, WIDTH, 4)).astype('f4')
    buffer_sum = context.buffer(buffer_sum_data)
    buffer_sum.bind_to_storage_buffer(1)

    print(buffer_raw_data[0])

    print('running compute shader')
    start_time = time.time()
    compute_shader.run(group_x=NUM_GROUPS)
    results = np.frombuffer(buffer_sum.read(), dtype=np.float32)\
        .reshape((HEIGHT, WIDTH, 4))
    the_sum = sum(results[::NUM_GROUPS])
    print('--- %d seconds ---' % (time.time() - start_time))
    print('--- the_sum = %d ---' % the_sum)

    print(buffer_raw_data)

    print('running normal python')
    start_time = time.time()
    the_sum = sum(buffer_raw_data[0][0])
    print('--- %d seconds ---' % (time.time() - start_time))
    print('--- the_sume = %d ---' % the_sum)

    print(results[-1:NUM_GROUPS:-1])
    print(results[::NUM_GROUPS])

    # print('normalizing buffers')
    # raw_output = np_image_from_buffer(buffer_raw, WIDTH, HEIGHT)
    # sum_output = np_image_from_buffer(buffer_sum, WIDTH, HEIGHT)
    #
    # print('saving to PNG')
    # imageio.imwrite(OUTPUT_PATH + '/summation.raw_output.png', raw_output, 'PNG')
    # imageio.imwrite(OUTPUT_PATH + '/summation.sum_output.png', sum_output, 'PNG')
