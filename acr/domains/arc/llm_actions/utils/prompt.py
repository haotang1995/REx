role_system_hypo_sum_prompt = """
You are a genius solving language puzzles.
"""

def get_code_prompt(task_examples, hypothesis):
    return f"""
{task_examples}
Now, please write a python program transform_grid(input_grid: np.ndarray[int]) -> np.ndarray[int] that transforms the input grid to the corresponding output grid.
Hint: You may want to use the following guidance to implement the function:
{hypothesis}
The number in the input grid can be mapped to the following colors:0:black; 1:blue; 2:red; 3:green; 4:yellow; 5:grey; 6:fuschia; 7:orange; 8:teal; 9:brown
Just reply with the implementation of transform_grid(input_grid: np.ndarray[int]) in Python and nothing else, each cell in the output should only be numbers from 0 to 9. Please contains the necessary import statements.
"""

def feedback_re_code(code, re_information, task_examples, hypothesis):
        return f"""
{task_examples}
Now, please write a python program transform_grid(input_grid: np.ndarray[int]) -> np.ndarray[int] that transforms the input grid to the corresponding output grid.
Hint: You may want to use the following guidance to implement the function:
{hypothesis}
The number in the input grid can be mapped to the following colors:0:black; 1:blue; 2:red; 3:green; 4:yellow; 5:grey; 6:fuschia; 7:orange; 8:teal; 9:brown
Just reply with the implementation of transform_grid(input_grid: np.ndarray[int]) in Python and nothing else, each cell in the output should only be numbers from 0 to 9.
This is the code you wrote last time:
```
{code}
```
It generates an error: 
{re_information}
Please correct the error and generate the code. Just reply with the implementation of transform_grid(input_grid: np.ndarray[int]) in Python and the and nothing else, each cell in the output should only be numbers from 0 to 9. Please contains the necessary import statements.
"""

def feedback_wrong_code(code, wrong_information, task_examples, hypothesis):
         return f"""
{task_examples}
Now, please write a python program transform_grid(input_grid: np.ndarray[int]) -> np.ndarray[int] that transforms the input grid to the corresponding output grid.
Hint: You may want to use the following guidance to implement the function:
{hypothesis}
The number in the input grid can be mapped to the following colors:0:black; 1:blue; 2:red; 3:green; 4:yellow; 5:grey; 6:fuschia; 7:orange; 8:teal; 9:brown
Just reply with the implementation of transform_grid(input_grid: np.ndarray[int]) in Python and nothing else, each cell in the output should only be numbers from 0 to 9.
This is the code you wrote last time:
```
{code}
```
These are the failed examples of the code:
{wrong_information}
Please correct the error and generate the code. Just reply with the implementation of transform_grid(input_grid: np.ndarray[int]) in Python and nothing else, each cell in the output should only be numbers from 0 to 9. Please contains the necessary import statements.
"""


