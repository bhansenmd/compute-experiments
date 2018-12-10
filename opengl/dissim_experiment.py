from enum import Enum
from time import time

import moderngl
import numpy as np
import pandas as pd


def dissim_experiment(number_of_things):
    data_in = np.random.uniform(-1.0, 1.0, (number_of_things, 4)) \
        .astype('f4')

    def create_shader_func():
        context = moderngl.create_standalone_context(require=430)

        buffer_in = context.buffer(data_in)
        buffer_in.bind_to_storage_buffer(0)

        data_out = np.zeros((number_of_things, number_of_things), dtype=np.float32)
        buffer_out = context.buffer(data_out)
        buffer_out.bind_to_storage_buffer(1)

        with open('./dissim_experiment.comp', 'r') as fp:
            shader_source = fp.read()
            shader_source = shader_source.replace('__NUMBER_OF_THINGS__', str(number_of_things))

        shader = context.compute_shader(shader_source)

        def shader_func():
            shader.run(group_x=number_of_things, group_y=number_of_things)
            results = np.frombuffer(buffer_out.read(), dtype=np.float32)\
                .reshape((number_of_things, number_of_things))

            return results

        return shader_func

    df_results = pd.DataFrame(columns=['method', 'results', 'time'])

    print('starting shader w/ reqs')
    start_time = time()
    shader_func = create_shader_func()
    results = shader_func()
    df_results.loc[0] = ['shader w/ reqs', str(results), time() - start_time]

    print('starting shader')
    start_time = time()
    results = shader_func()
    df_results.loc[1] = ['shader', str(results), time() - start_time]

    print('starting numpy for loop')
    results = np.zeros((number_of_things, number_of_things), dtype=np.float32)
    start_time = time()
    for x in range(0, number_of_things):
        for y in range(0, number_of_things):
            if x == y:
                results[x][y] = 0
                continue

            a = data_in[x][:-1]
            b = data_in[y][:-1]

            na = a / np.linalg.norm(a)
            nb = b / np.linalg.norm(b)

            results[x][y] = (1 - np.dot(na, nb)) / 2
    df_results.loc[2] = ['numpy for loop', str(results), time() - start_time]

    return df_results


if __name__ == '__main__':
    print('number_of_things = 32')
    print(dissim_experiment(32))
    print('number_of_things = 512')
    print(dissim_experiment(512))
