from .utils.format_utils import transform_task_examples, get_python_code, get_prompt_by_code_result
from .utils.prompt import feedback_re_code,feedback_wrong_code
from .utils.run_code import get_code_answer,test_code_final
from .utils.prior import get_pass_rate

def refine_code_arc(data, hypo, llm, refine_type, code, previous_result, prior_type='pass_task', logger = None):
    task_train_str = transform_task_examples(data['train'])
    Correct_task, Wrong_task, RE_task = previous_result
    if refine_type == 'RE':
        user = feedback_re_code(code, RE_task[0]['code_output'], task_train_str, hypo)
    else:
        wrong_information = get_prompt_by_code_result(Wrong_task)
        user = feedback_wrong_code(code, wrong_information, task_train_str, hypo)
    system = ""
    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    max_try = 3
    response = None
    code = None
    with llm.track() as costs:
        for _ in range(max_try):
            response = llm(prompt)
            if response is not None:
                code = get_python_code(response.choices[0].message.content)
                if code is not None:
                    break

    if code is None:
        return {
            'get_answer': False,
        }
    else:
        Correct_task, Wrong_task, RE_task, pass_train = get_code_answer(data, code, logger)
        heuristic_reward = get_pass_rate(Correct_task, Wrong_task, RE_task, prior_type)
        refine_type = 'RE' if len(RE_task) > 0 else 'WA'
        pass_test = False
        if pass_train:
            pass_test, _ = test_code_final(data, code, logger)
        return{
            'get_answer': True,
            'code': code,
            'run_result': (Correct_task, Wrong_task, RE_task),
            'pass_train': pass_train,
            'refine_type':refine_type,
            'heuristic_reward': heuristic_reward,
            'pass_test': pass_test,
            'cost': costs.usage,
        }