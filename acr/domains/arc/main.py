from ..base import _Domain
from ...utils.llm import get_llm
from .data.arc_data import get_arc_data
from .llm_actions.init import init_code_arc
from .llm_actions.refine import refine_code_arc

class InitCode:
    def __init__(self, data, problem_id, hypo, llm):
        self.data = data
        self.hypo = hypo
        self.llm = llm
        self.problem_id = problem_id
        self.name = f"InitCode(task id: {problem_id})"
    def run(self):
        return init_code_arc(self.data, self.hypo,self.llm)
    
class RefineCode:
    def __init__(self, data, hypo, run_result, code, llm, refine_type, index, problem_id):
        self.data = data
        self.hypo = hypo
        self.code = code
        self.run_result = run_result
        self.llm = llm
        self.refine_type = refine_type
        self.index = index
        self.name = f"RefineCode(task id: {problem_id}, code index: {index})"
    def run(self):
        return refine_code_arc(self.data, self.hypo, self.llm, self.refine_type, self.code, self.run_result)
    
class ARCDomain(_Domain):
    def __init__(self, args,):
        self.llm = get_llm(args)
        self.dataset = get_arc_data()
        self.task_run_result = {}
        self.seed = args.llm_seed

    def reset(self, problem_id,):
        # -> list of actions [(action_index, action_name, heuristic_reward)]
        self.problem_id = problem_id
        self.data = self.dataset[problem_id]['data']
        self.human_hypo = self.dataset[problem_id]['human_description']
        self.actions = [InitCode(self.data, problem_id, self.human_hypo, self.llm),]        
        self.cur_step = 0
        self.task_run_result[problem_id] = {
            'success' : False,
            'pass_train': False,
            'pass_test': False,
            'use_step': 0,
            'hypo': self.human_hypo,
            'llm-seed': self.seed,
            'heuristic_reward': 0
        }
        return [(0, self.actions[0].name, None)]
    
    def step(self, action_index):
        # -> reward, done
        # -> list of new actions [(action_index, action_name, heuristic_reward)]
        assert 0 <= action_index < len(self.actions)
        self.cur_step += 1
        action = self.actions[action_index]
        print()
        print('='*10, f'Step {self.cur_step}', f'Action {action.name}', '='*10)
        while True:
            result = action.run()
            if result['get_answer']:
                break
        if result['get_answer'] == False:
            reward = 0
            done = False
            new_actions = []
        else:
            # refine RE code or WA code
            new_action_index = len(self.actions)
            new_action = RefineCode(self.data, self.human_hypo, result['run_result'], result['code'], 
                                    self.llm, result['refine_type'], new_action_index, self.problem_id)
            new_action_name = new_action.name
            self.actions.append(new_action)

            new_actions = [(new_action_index, new_action_name, result['heuristic_reward'])]
            done = result['pass_train']
            reward = 1 if done else 0

            # update result
            if done:
                self.task_run_result[self.problem_id]['pass_train'] = True
                self.task_run_result[self.problem_id]['success'] = True
                self.task_run_result[self.problem_id]['pass_test'] = result['pass_test']
                print('='*10, f'Pass task in step {self.cur_step}', '='*10)
        self.task_run_result[self.problem_id]['use_step'] = self.cur_step
        self.task_run_result[self.problem_id]['heuristic_reward'] = result['heuristic_reward']
            
        return reward, done, new_actions
    
    def get_metrics(self):
        return self.task_run_result[self.problem_id].copy()
    
    
    def __len__(self):
        return len(self.dataset)
    
    def __str__(self):
        return f'ARCDomain({len(self.dataset)})'
    
    def __repr__(self):
        return str(self)
    
    def summarize_results(self, results):
        assert len(results) == len(self)
        pass_step = []
        for problem_id in self.task_run_result.keys():
            result = self.task_run_result[problem_id]
            if result['pass_train']:
                pass_step.append(result['use_step'])
        pass_step = sorted(pass_step)
        pass_task_require = {}
        for i in range(10, 26):
            if i < len(pass_step):
                print(f"Pass {i} steps: {pass_step[i]}")
                pass_task_require[i] = pass_step[i]
            else:
                print(f'Fail to pass {i}')    
        return {
            'pass_task': f'{len(pass_step)} / {len(self.task_run_result)}',
            'pass_task_require': pass_task_require,
            'pass_task_rate': len(pass_step) / len(self.task_run_result),
        }
    
    def set_seed(self, seed):
        self.llm.set_seed(seed)