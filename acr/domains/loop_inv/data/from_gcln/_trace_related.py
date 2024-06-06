# Codes copied from G-CLN, reformatted here for easier access

import os
from .run_nla import PROBLEMS as CONFIGS

def get_traces(problem_name, loop_index, trace_path,):
    problem_num = problem_name
    config = CONFIGS[problem_num][loop_index]
    pre_sample = config.pre_sample
    fractional_sampling = False
    if type(problem_number) == int:
        df = pd.read_csv(trace_path + str(problem_number) + ".csv", skipinitialspace=True)
        df_data = df.drop(columns=['trace_idx', 'init', 'final'], errors='ignore')
    else:
        if pre_sample:
            df = pd.read_csv(trace_path + str(problem_number) + ".csv", skipinitialspace=True)
            if fractional_sampling is True and problem_number in ['ps5', 'ps6']:
                _, __, simple_invariants_ = setup_polynomial_data(df.drop(columns=['trace_idx']), gen_poly=False, problem_number=problem_number)
                df = pd.read_csv(trace_path + str(problem_number) + "_fractional.csv", skipinitialspace=True)
        else:
            df = pd.read_csv(trace_path + str(problem_number) + '_' + str(loop_index) + ".csv", skipinitialspace=True)
        df_data = df[df['trace_idx'] == loop_index].drop(columns=['trace_idx'])
    # print("data: \n", df_data)
    return df_data

