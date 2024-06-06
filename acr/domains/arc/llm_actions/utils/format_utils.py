import re
def transform_list_to_str(task_list):
    task_list = task_list.replace("], [", "]\n[")
    task_list = task_list.replace(", ", " ")
    return task_list

def transform_task_examples(task_train):
    str_format = ""
    for i, pair in enumerate(task_train):
        str_format += f"Example {i}:\n"
        input_str = transform_list_to_str(str(pair['input']))
        output_str = transform_list_to_str(str(pair['output']))
        str_format += f"Input:\n{input_str}\nOutput:\n{output_str}\n"
    return str_format

def get_python_code(content):
    if content is None:
        return None
    start_tag = "```python\n"
    end_tag = "```"
    start = content.find(start_tag) + len(start_tag)
    end = content.find(end_tag, start)
    if start == -1 or end == -1:
        return None
    python_part = content[start:end]
    clean_print = re.sub(r'print\(.*\)', '', python_part)
    return clean_print

def get_prompt_by_code_result(Wrong_task):
    refine_str = ""
    if len(Wrong_task) > 0:
        for i, task in enumerate(Wrong_task):
            refine_str += f"Task {i+1}:\n"
            refine_str += f"Input:\n {task['input']}\n"
            refine_str += f"Correct Output:\n {task['correct_output']}\n"
            refine_str += f"Code Output:\n {task['code_output']}\n"
        refine_str += "\n"
    return refine_str