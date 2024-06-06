import aiohttp  # for making API calls concurrently
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os
import re  # for matching endpoint from request URL
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from dataclasses import dataclass  # for storing API inputs, outputs, and metadata
import copy
import math
import platform
import sys
from .utility import read_jsonl, write_jsonl, read_problems_data, convert_jsonl_to_d
from typing import List
import random
random.seed(47)


if platform.system() == 'Windows':
    import _locale
    _locale._getdefaultlocale = (lambda *args: ['en_US', 'utf8'])


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    oai_keys: List[str],
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 3
    seconds_to_sleep_each_loop = 0.001  # 1 ms limits max throughput to 1,000 requests per second

    # initialize logging
    logging.basicConfig(level=logging_level, stream=sys.stdout)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    # request_header = {"Authorization": f"Bearer {api_key}"}
    current_index = 0
    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()  # generates integer IDs of 1, 2, 3, ...
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    # initialize file reading
    with open(requests_filepath) as file:
        # `requests` will provide requests one at a time
        requests = file.__iter__()
        logging.debug(f"File opened. Entering main loop")

        while True:
            # get next request (if one is not already waiting for capacity)
            if next_request is None:
                if not queue_of_requests_to_retry.empty():
                    next_request = queue_of_requests_to_retry.get_nowait()
                    logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
                elif file_not_finished:
                    try:
                        # get new request
                        request_json = json.loads(next(requests))
                        placeholder_id = request_json.pop('placeholder_id')
                        if os.path.exists(save_filepath.format(placeholder_id)):
                            assert next_request is None  # reset next_request to empty
                            continue
                        next_request = APIRequest(
                            task_id=next(task_id_generator),
                            request_json=request_json,
                            token_consumption=num_tokens_consumed_from_request(request_json, api_endpoint, token_encoding_name),
                            attempts_left=max_attempts,
                            save_filepath=save_filepath.format(placeholder_id)
                        )
                        status_tracker.num_tasks_started += 1
                        status_tracker.num_tasks_in_progress += 1
                        logging.debug(f"Reading request {next_request.task_id}: {next_request}")
                    except StopIteration:
                        # if file runs out, set flag to stop reading it
                        logging.debug("Read file exhausted")
                        file_not_finished = False

            # update available capacity
            current_time = time.time()
            seconds_since_update = current_time - last_update_time
            available_request_capacity = min(
                available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
                max_requests_per_minute,
            )
            available_token_capacity = min(
                available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
                max_tokens_per_minute,
            )
            last_update_time = current_time

            # if enough capacity available, call API
            if next_request:
                next_request_tokens = next_request.token_consumption
                if (
                    available_request_capacity >= 1
                    and available_token_capacity >= next_request_tokens
                ):
                    # update counters
                    available_request_capacity -= 1
                    available_token_capacity -= next_request_tokens
                    next_request.attempts_left -= 1

                    # call API
                    current_key = oai_keys[current_index]
                    asyncio.create_task(
                        next_request.call_api(
                            request_url=request_url,
                            request_header={"Authorization": f"Bearer {current_key}"},
                            retry_queue=queue_of_requests_to_retry,
                            status_tracker=status_tracker,
                        )
                    )
                    current_index = (current_index + 1) % len(oai_keys)
                    next_request = None  # reset next_request to empty

            # if all tasks are finished, break
            if status_tracker.num_tasks_in_progress == 0:
                break

            # main loop sleeps briefly so concurrent tasks can run
            await asyncio.sleep(seconds_to_sleep_each_loop)

            # if a rate limit error was hit recently, pause to cool down
            seconds_since_rate_limit_error = (time.time() - status_tracker.time_of_last_rate_limit_error)
            if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
                remaining_seconds_to_pause = (seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)
                await asyncio.sleep(remaining_seconds_to_pause)
                # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                logging.warning(f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}")

        # after finishing, log final status
        logging.info(f"""Parallel processing complete. Results saved to {save_filepath}""")
        if status_tracker.num_tasks_failed > 0:
            logging.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}.")
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate.")


# dataclasses


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    save_filepath: str
    result = []

    async def call_api(
        self,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        status_tracker: StatusTracker,
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=request_url, headers=request_header, json=self.request_json
                ) as response:
                    response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # rate limit errors are counted separately

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}")
                write_to_jsonl_multiple([self.request_json, [str(e) for e in self.result]], self.save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            write_to_jsonl_multiple([self.request_json, response], self.save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {self.save_filepath}")


# functions


def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    match = re.search('^https://[^/]+/v\\d+/(.+)$', request_url)
    return match[1]


def write_to_jsonl_multiple(data, filename: str) -> None:
    request_json, full_response = data
    text_responses = []
    for i in range(request_json['n']):
        text_response = full_response['choices'][i]['message']['content']
        text_responses.append(text_response)
        print('######')
        print(re.sub(r'[^\x00-\x7F]+', '', text_response))  # compatibility to print
        print('######')
    result = [{'full_response': full_response, **request_json, 'generation': text_responses}]
    write_jsonl(result, filename)


def construct_initial_prompts(json_data):
    prompt_template = 'You are given a hard python programming contest problem:\n\nQuestion:\n\n{0}\n\n{2}\n\nCan you write a correct solution and make sure it passes the example test cases? (Please start the solution segment with ```python as usual{1})'
    for i in range(len(json_data)):
        question = json_data[i]['question']
        evaluate_mode = json_data[i]['evaluate_mode_string']
        starter_code = json_data[i]['starter_code']
        starter_code = starter_code.rstrip()
        if evaluate_mode == 'function_call':
            instruction_string = ''
            starter_code_instruction = 'The starter code is:\n\n{0}'.format(starter_code)
        else:
            instruction_string = ' and use Standard Input format'
            starter_code_instruction = ''
        prompt = prompt_template.format(question, instruction_string, starter_code_instruction)
        json_data[i]['prompt'] = prompt
        json_data[i]['placeholder_id'] = json_data[i]['question_name'] + '/'


def construct_iterative_prompts(evaluation_results):
    compilation_error_prompt_template = 'You are given a hard python programming contest problem:\n\nQuestion:\n\n{0}\n\n{2}\n\nBelow is a failed attempt to solve the problem:\n\n{3}\n\nThe failure is due to {4}\n\nCan you\n1. Analyze why the compilation error occurs?\n2. Correct the code and make sure it passes the example test cases? (Please start the solution segment with ```python as usual{1})'
    runtime_error_generic_prompt_template = 'You are given a hard python programming contest problem:\n\nQuestion:\n\n{0}\n\n{2}\n\nBelow is a failed attempt to solve the problem:\n\n{3}\n\nThe failure is due to {4}\n\nCan you\n1. Analyze why the runtime error occurs?\n2. Correct the code and make sure it passes the example test cases? (Please start the solution segment with ```python as usual{1})'
    runtime_error_timeout_prompt_template = 'You are given a hard python programming contest problem:\n\nQuestion:\n\n{0}\n\n{2}\n\nBelow is a failed attempt to solve the problem:\n\n{3}\n\nThe failure is due to {4}\n\nCan you\n1. Analyze why the timeout error occurs?\n2. Make the code more efficient and make sure it passes the example test cases? (Please start the solution segment with ```python as usual{1})'
    test_case_failure_normal_prompt_template = 'You are given a hard python programming contest problem:\n\nQuestion:\n\n{0}\n\n{2}\n\nBelow is a failed attempt to solve the problem:\n\n{3}\n\nThe failure is due to {4}\n\nCan you\n1. Analyze why the expected output should be {5}?\n2. Analyze why the actual output is {6}?\n3. Correct the code and make sure it passes the failed test case as well as example test cases? (Please start the solution segment with ```python as usual{1})'
    test_case_failure_no_output_prompt_template = 'You are given a hard python programming contest problem:\n\nQuestion:\n\n{0}\n\n{2}\n\nBelow is a failed attempt to solve the problem:\n\n{3}\n\nThe failure is due to {4}\n\nCan you\n1. Analyze why the expected output should be {5}?\n2. Analyze why the code did not output anything?\n3. If the above is due to miss-handling input and output, correct the code by making sure it reads from input, prints output properly and does not contain if __name__ == "__main__". Also make sure the code passes the failed test case as well as example test cases. (Please start the solution segment with ```python as usual{1})'
    unknown_error_prompt_template = 'You are given a hard python programming contest problem:\n\nQuestion:\n\n{0}\n\n{2}\n\nBelow is a failed attempt to solve the problem:\n\n{3}\n\nThe failure is due to an unknown error\n\nCan you\n1. Correct the code and make sure it passes the example test cases? (Please start the solution segment with ```python as usual{1})'
    starter_code_signature_mismatch_template = 'You are given a hard python programming contest problem:\n\nQuestion:\n\n{0}\n\n{2}\n\nBelow is a failed attempt to solve the problem:\n\n{3}\n\nThe failure is due to {4}\n\nCan you\n1. Correct the code and make sure it uses the same starter code signature and passes the example test cases? (Please start the solution segment with ```python as usual{1})'
    all_results = []
    for i in range(len(evaluation_results)):
        current_evaluation_result = evaluation_results[i]
        evaluate_mode = current_evaluation_result['evaluate_mode_string']
        starter_code = current_evaluation_result['starter_code']
        starter_code = starter_code.rstrip()
        if evaluate_mode == 'function_call':
            instruction_string = ''
            starter_code_instruction = 'The starter code is:\n\n{0}'.format(starter_code)
        else:
            instruction_string = ' and use Standard Input format'
            starter_code_instruction = ''
        current_message = current_evaluation_result.pop('evaluation_message')
        current_solution = current_evaluation_result.pop('evaluation_solution')
        if current_solution == 'evaluation failed':
            assert current_message == 'evaluation failed'
            prompt = unknown_error_prompt_template.format(current_evaluation_result['question'], instruction_string, starter_code_instruction, current_solution)
        elif current_message == 'no solution extracted':
            continue
        elif current_message == 'all test cases passed':
            continue
        elif current_message.startswith('starter code signature mismatch error.'):
            prompt = starter_code_signature_mismatch_template.format(current_evaluation_result['question'], instruction_string, starter_code_instruction, current_solution, current_message)
        elif current_message.startswith('compilation error.'):
            prompt = compilation_error_prompt_template.format(current_evaluation_result['question'], instruction_string, starter_code_instruction, current_solution, current_message)
        elif current_message.startswith('runtime error at'):
            if 'Error type: TimeoutException' in current_message:
                prompt = runtime_error_timeout_prompt_template.format(current_evaluation_result['question'], instruction_string, starter_code_instruction, current_solution, current_message)
            else:
                prompt = runtime_error_generic_prompt_template.format(current_evaluation_result['question'], instruction_string, starter_code_instruction, current_solution, current_message)
        elif current_message.startswith('test case failed'):
            if 'The code did not output anything' in current_message:
                expected_output_start_index = current_message.index('and expected output ')
                expected_output_end_index = current_message.index('. The code did not output anything')
                expected_output = current_message[expected_output_start_index + len('and expected output '):expected_output_end_index]
                prompt = test_case_failure_no_output_prompt_template.format(current_evaluation_result['question'], instruction_string, starter_code_instruction, current_solution, current_message, expected_output)
            else:
                expected_output_start_index = current_message.index('and expected output ')
                expected_output_end_index = current_message.index(', actual output is ')
                expected_output = current_message[expected_output_start_index + len('and expected output '):expected_output_end_index]
                actual_output = current_message[expected_output_end_index + len(', actual output is '):]
                prompt = test_case_failure_normal_prompt_template.format(current_evaluation_result['question'], instruction_string, starter_code_instruction, current_solution, current_message, expected_output, actual_output)
        else:
            raise NotImplementedError
        current_evaluation_result['prompt'] = prompt
        all_results.append(current_evaluation_result)
    return all_results


def count_prompt_num_tokens(messages_d):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = 0
    for message in messages_d:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens -= 1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # TODO hardcoded, but should be fine for gpt3.5 and gpt4
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 800)  # 800 is probably the average token consumed
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError('Expecting either string or list of strings for "prompt" field in completion request')
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')


def change_model_if_necessary(current_model, prompt_num_tokens, leave_num_token_response):
    assert current_model in ['gpt-3.5-turbo', 'gpt-4']
    context_length_d = {'gpt-3.5-turbo': 4096, 'gpt-4': 8192, 'gpt-3.5-turbo-16k': 16384, 'gpt-4-32k': 32768}
    if prompt_num_tokens + leave_num_token_response > context_length_d[current_model]:
        if current_model == 'gpt-3.5-turbo':
            if prompt_num_tokens + leave_num_token_response > context_length_d['gpt-3.5-turbo-16k']:
                return None
            return 'gpt-3.5-turbo-16k'
        else:
            return None
    else:
        return current_model


def generate_branching_factor(generation_count_d, placeholder_id, branching_factor_low, branching_factor_high, limit):
    problem_name = placeholder_id.split('/')[0]
    aggregated_generation_count_d = {}  # k is a problem name, v is aggregated generation on that problem
    for k, v in generation_count_d.items():
        current_problem_name = k.split('/')[0]
        if current_problem_name not in aggregated_generation_count_d:
            aggregated_generation_count_d[current_problem_name] = 0
        aggregated_generation_count_d[current_problem_name] += v
    current_generation = aggregated_generation_count_d[problem_name]
    if current_generation >= limit:
        generation_count_d[placeholder_id] = 0
        return 0
    elif limit - current_generation < branching_factor_low:
        generation_count_d[placeholder_id] = limit - current_generation
        return limit - current_generation
    else:
        actual_high = min(limit - current_generation, branching_factor_high)
        actual_branching_factor = random.randint(branching_factor_low, actual_high)
        generation_count_d[placeholder_id] = actual_branching_factor
        return actual_branching_factor


def construct_inference_json_chunk(message_d, json_data, generation_count_d, branching_factor, limit, maximum_n, generation_params, use_longer_context_model_if_needed, inference_json_attribute='prompt', leave_num_token_response=900):
    # break prompts into math.ceil(branching_factor / maximum_n) chunks to do inference
    results = []
    total_actual_generations = 0
    for i in range(len(json_data)):
        prompt = json_data[i][inference_json_attribute]
        placeholder_id = json_data[i]['placeholder_id']
        if type(branching_factor) == list:
            assert len(branching_factor) == 2
            actual_branching_factor = generate_branching_factor(generation_count_d, placeholder_id, branching_factor[0], branching_factor[1], limit)
        else:
            actual_branching_factor = branching_factor
        # hardcoded
        message_d_copy = copy.deepcopy(message_d)
        message_d_copy[-1]['content'] = prompt
        repeats = math.ceil(actual_branching_factor / maximum_n)
        for j in range(repeats):
            generation_params_copy = copy.deepcopy(generation_params)
            if j != repeats - 1:
                generation_params_copy['n'] = maximum_n
            else:
                generation_params_copy['n'] = actual_branching_factor - maximum_n*j
            if placeholder_id == json_data[i]['question_name'] + '/':
                current_placeholder_id = placeholder_id + 't{0}'.format(j)
            else:
                current_placeholder_id = placeholder_id + '-t{0}'.format(j)
            current_d = {'messages': message_d_copy, **generation_params_copy, 'placeholder_id': current_placeholder_id}
            if use_longer_context_model_if_needed:
                prompt_num_tokens = count_prompt_num_tokens(message_d_copy)
                new_model_proposed = change_model_if_necessary(generation_params_copy['model'], prompt_num_tokens, leave_num_token_response)
                if new_model_proposed is None:
                    print('skipping {0} due to context length'.format(current_placeholder_id))
                    generation_count_d[placeholder_id] -= generation_params_copy['n']
                    continue  # no more children nodes
                else:
                    if new_model_proposed != generation_params_copy['model']:
                        print('changing model from {0} to {1}'.format(generation_params_copy['model'], new_model_proposed))
                        current_d['model'] = new_model_proposed
            results.append(current_d)
            total_actual_generations += generation_params_copy['n']
    return results, total_actual_generations


def check_finished(requests_filepath, save_filepath_template):
    data = read_jsonl(requests_filepath)
    all_ids = [e['placeholder_id'] for e in data]
    for placeholder_id in all_ids:
        if not os.path.exists(save_filepath_template.format(placeholder_id)):
            return False
    return True


def complete_current_inference(oai_keys, requests_filepath, save_filepath_template, max_requests_per_minute, max_tokens_per_minute, request_url='https://api.openai.com/v1/chat/completions', max_attempts=1000000, logging_level=logging.INFO):
    completed = check_finished(requests_filepath, save_filepath_template)
    while not completed:
        asyncio.run(
            process_api_requests_from_file(
                requests_filepath=requests_filepath,
                save_filepath=save_filepath_template,
                request_url=request_url,
                oai_keys=oai_keys,
                max_requests_per_minute=float(max_requests_per_minute),
                max_tokens_per_minute=float(max_tokens_per_minute),
                token_encoding_name='',
                max_attempts=int(max_attempts),
                logging_level=int(logging_level),
            )
        )
        completed = check_finished(requests_filepath, save_filepath_template)
        print('completion status {0}'.format(completed))


def combine_and_expand_inference_chunks(requests_filepath, save_filepath_template, delete_temporary=True):
    data = read_jsonl(requests_filepath)
    all_ids = [e['placeholder_id'] for e in data]
    combine_d = {}
    for e in all_ids:
        assert 't' in e
        if '-t' in e:
            prefix = e[:e.index('-t')]
        else:
            prefix = e[:e.index('t')]
        if prefix not in combine_d:
            combine_d[prefix] = []
        combine_d[prefix].append(save_filepath_template.format(e))
    results = {}
    for k, v in combine_d.items():
        if k not in results:
            results[k] = []
        for i in range(len(v)):
            current_data = read_jsonl(v[i])[0]  # each generation should only be a large json
            for j in range(len(current_data['generation'])):
                current_data_copy = copy.deepcopy(current_data)
                current_data_copy['generation'] = current_data_copy['generation'][j]
                results[k].append(current_data_copy)
    for k, v in results.items():
        for i in range(len(v)):
            if k.endswith('/'):
                new_save_id = k + str(i)
            else:
                new_save_id = k + '-' + str(i)
            write_jsonl([v[i]], save_filepath_template.format(new_save_id))
        # if delete_temporary:
        #     for e in combine_d[k]:
        #         # os.system('rm -rf {0}'.format(e))
        #         print('dry run: rm -rf {0}'.format(e))


def aggregate_evaluation_results(request_data, requests_filepath, problem_data, current_depth_branching_factor, generation_count_d):
    evaluation_path_template = requests_filepath.rsplit('/', 1)[0] + '/evaluation/{0}.jsonl'
    results = []
    succeed_problems = []
    for e in request_data:
        problem_name = e['placeholder_id'].split('/')[0]
        current_problem_data = problem_data[problem_name]
        if type(current_depth_branching_factor) == list:
            actual_branching_factor = generation_count_d[e['placeholder_id']]
            assert actual_branching_factor <= current_depth_branching_factor[1]
        else:
            actual_branching_factor = current_depth_branching_factor
        for i in range(actual_branching_factor):
            current_problem_data_copy = copy.deepcopy(current_problem_data)
            if e['placeholder_id'].endswith('/'):
                new_placeholder_id = e['placeholder_id'] + str(i)
            else:
                new_placeholder_id = e['placeholder_id'] + '-' + str(i)
            current_evaluation_path = evaluation_path_template.format(new_placeholder_id)
            if os.path.exists(current_evaluation_path):
                evaluation_result = read_jsonl(current_evaluation_path)[0]
                current_problem_data_copy['evaluation_message'] = evaluation_result['evaluation']
                current_problem_data_copy['evaluation_solution'] = evaluation_result['solution']
                if evaluation_result['evaluation'] == 'all test cases passed':
                    if problem_name not in succeed_problems:
                        succeed_problems.append(problem_name)
            else:
                current_problem_data_copy['evaluation_message'] = 'evaluation failed'
                current_problem_data_copy['evaluation_solution'] = 'evaluation failed'
            current_problem_data_copy['placeholder_id'] = new_placeholder_id
            results.append(current_problem_data_copy)
    return results


def iterative_inference(oai_keys, data_path, requests_filepath, save_filepath_template, problem_names, message_d, branching_factors, branching_factors_type, maximum_n, generation_params, max_requests_per_minute, max_tokens_per_minute, use_longer_context_model_if_needed, start_eval_test_cases_depth=0, run_all_tests=0, shuffle_inference=1, generation_limit=100):
    if save_filepath_template is None:
        save_filepath_template = requests_filepath.rsplit('/', 1)[0] + '/model_predictions/{0}.jsonl'
    problem_data = read_problems_data(data_path, problem_names)
    generation_count_d = {k + '/': 0 for k in problem_names}  # each k is an individual request, v is the corresponding actual branching factor
    problem_data_d = convert_jsonl_to_d(problem_data)
    total_evaluation_done = 0
    if branching_factors_type == 'random_interval':
        max_depth = branching_factors[-1]
    else:
        max_depth = len(branching_factors)
    for i in range(max_depth):
        if i == 0:
            request_data = copy.deepcopy(problem_data)  # do not want to touch problem data information
            construct_initial_prompts(request_data)
        else:
            request_data = construct_iterative_prompts(evaluation_results)
        if branching_factors_type == 'random_interval':
            current_depth_branching_factor = branching_factors[:2]
        else:
            current_depth_branching_factor = branching_factors[i]
        requests_chunked, total_actual_generations = construct_inference_json_chunk(message_d, request_data, generation_count_d, current_depth_branching_factor, generation_limit, maximum_n, generation_params, use_longer_context_model_if_needed)
        if shuffle_inference:
            random.shuffle(requests_chunked)
        write_jsonl(requests_chunked, requests_filepath)
        complete_current_inference(oai_keys, requests_filepath, save_filepath_template, max_requests_per_minute, max_tokens_per_minute)
        combine_and_expand_inference_chunks(requests_filepath, save_filepath_template)
        if i >= start_eval_test_cases_depth:
            os.system('bash scripts/evaluate_solutions.sh {0} {1} {2} {3} {4}'.format(requests_filepath.rsplit('/', 1)[0] + '/', total_actual_generations, i, run_all_tests, data_path))
        else:
            print('skipping running test cases again for depth {0} until depth {1}'.format(i, start_eval_test_cases_depth))
        evaluation_results = aggregate_evaluation_results(request_data, requests_filepath, problem_data_d, current_depth_branching_factor, generation_count_d)
        total_evaluation_done += len(evaluation_results)
        print('total solutions generated so far: {0}'.format(total_evaluation_done))
        print('finished round {0}'.format(i))
