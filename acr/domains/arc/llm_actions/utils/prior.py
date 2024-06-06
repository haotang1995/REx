import numpy as np
from collections import Counter

def similarity_color(a,b):
    # a origin b test
    # filter out dominant color
    color_set_a, color_set_b = {},{}
    for i in range(10):
        color_set_a[i] = 0
        color_set_b[i] = 0

    a = np.array(a)
    b = np.array(b)
    a = a.flatten()
    b = b.flatten()
    counter_a = Counter(a)
    counter_b = Counter(b)  
    max_key_a = max(color_set_a, key=color_set_a.get)
    max_key_b = max(color_set_b, key=color_set_b.get)
    flag_a = True
    flag_b = True
    for (key, value) in counter_a.items():
        if key != max_key_a and value * 2 >= counter_a[max_key_a]:
            flag_a = False

    for (key, value) in counter_b.items():
        if key != max_key_b and value * 2 >= counter_b[max_key_b]:
            flag_b = False
    
    color_set_a_vec, color_set_b_vec = [], []
    for (key, value) in counter_a.items():
        if key not in color_set_a.keys():
            continue
        color_set_a[key] = value    
    for (key, value) in counter_b.items():
        if key not in color_set_a.keys():
            continue
        color_set_b[key] = value

    if flag_a and flag_b and max_key_a == max_key_b and len(color_set_a.keys()) > 1:
        color_set_a.pop(max_key_a)
        color_set_b.pop(max_key_b)

    color_set_tem = color_set_a.copy()
    for (key, value) in color_set_tem.items():
        if color_set_a[key] == 0 and color_set_b[key] == 0:
            color_set_a.pop(key)
            color_set_b.pop(key)
    for (key, value) in color_set_a.items():
        color_set_a_vec.append(value)
    for (key, value) in color_set_b.items():
        color_set_b_vec.append(value)

    color_set_a_vec = np.array(color_set_a_vec)
    color_set_b_vec = np.array(color_set_b_vec)

    color_set_a_vec = np.where(color_set_a_vec == 0, -1, color_set_a_vec)
    color_set_b_vec = np.where(color_set_b_vec == 0, -1, color_set_b_vec)
    if len(color_set_a_vec) != len(color_set_b_vec):
        print(color_set_a_vec, color_set_b_vec)
        print(counter_a)
        print(counter_b)
    cosine_similarity = np.dot(color_set_a_vec, color_set_b_vec) / (np.linalg.norm(color_set_a_vec) * np.linalg.norm(color_set_b_vec))
    euclidean_distance = np.linalg.norm(color_set_a_vec - color_set_b_vec)
    manhattan_distance = np.sum(np.abs(color_set_a_vec - color_set_b_vec))

    euclidean_similarity = 1 if euclidean_distance == 0 else 1.0 / euclidean_distance
    manhattan_similarity = 1 if manhattan_distance == 0 else 1.0 / manhattan_distance
    cosine_similarity = 0 if cosine_similarity < 0 else cosine_similarity

    return cosine_similarity, euclidean_similarity, manhattan_similarity
    

def get_pass_rate(Correct_task, Wrong_task, RE_task, ratio_type):
    if ratio_type == "pass_task":
        pass_ratio = 1.0 * len(Correct_task) / (len(Correct_task) + len(Wrong_task) + len(RE_task))
    else:
        pass_ratio = 1.0 * len(Correct_task)
        for wrong in Wrong_task:
            cos, euler, manhattan = similarity_color(wrong["correct_output"], wrong["code_output"])
            pass_ratio += euler * 0.4
        pass_ratio = 1.0 * pass_ratio / (len(Correct_task) + len(Wrong_task) + len(RE_task))
    return pass_ratio