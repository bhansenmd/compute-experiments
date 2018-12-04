from time import time

import moderngl
import numpy as np
import pandas as pd


def sum_experiment(number_of_things, work_group_size=32):
    data_raw = np.ones(number_of_things, dtype=np.float32)
    data_out = np.zeros(number_of_things, dtype=np.float32)

    def create_shader_func():
        context = moderngl.create_standalone_context(require=430)

        buffer_raw = context.buffer(data_raw)
        buffer_raw.bind_to_storage_buffer(0)

        buffer_out = context.buffer(data_out)
        buffer_out.bind_to_storage_buffer(1)

        with open('./sum_experiment.comp', 'r') as fp:
            shader_source = fp.read()
            shader_source = shader_source.replace('__WORK_GROUP_SIZE__', str(work_group_size))

        shader = context.compute_shader(shader_source)

        def shader_func():
            shader.run(group_x=number_of_things // work_group_size)
            results = np.frombuffer(buffer_out.read(), dtype=np.float32)
            return sum(results[work_group_size-1::work_group_size])

        return shader_func

    df_results = pd.DataFrame(columns=['method', 'sum', 'time'])

    start_time = time()
    shader_func = create_shader_func()
    result = shader_func()
    df_results.loc[0] = ['shader w/ reqs', result, time() - start_time]

    start_time = time()
    result = shader_func()
    df_results.loc[1] = ['shader', result, time() - start_time]

    start_time = time()
    result = np.sum(data_raw)
    df_results.loc[2] = ['numpy', result, time() - start_time]

    start_time = time()
    result = sum(data_raw)
    df_results.loc[3] = ['python', result, time() - start_time]

    return df_results


if __name__ == '__main__':
    work_group_size = 2 ** 5
    number_of_things = 2 ** 20
    print(sum_experiment(number_of_things, work_group_size))
    print('----------------')
    print('number_of_things = {}'.format(number_of_things))
    print('work_group_size = {}'.format(work_group_size))
    print('number_of_work_groups = {}'.format(number_of_things // work_group_size))
