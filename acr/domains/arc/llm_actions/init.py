from .utils.format_utils import transform_task_examples, get_python_code
from .utils.prompt import get_code_prompt
from .utils.run_code import get_code_answer,test_code_final
from .utils.prior import get_pass_rate

def init_code_arc(data, hypo, llm, prior_type='pass_task', logger = None):
    system = ""
    task_train_str = transform_task_examples(data['train'])
    user = get_code_prompt(task_train_str, hypo)
    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    max_try = 3
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
            'pass_test': pass_test,
            'heuristic_reward': heuristic_reward,
            'cost': costs.usage,
        }
