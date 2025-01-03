import re
import json
from bfcl.model_handler.oss_model.base_oss_handler import OSSHandler

class FalconHandler(OSSHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)

    def _fix_functions_format(self, functions):
        if isinstance(functions, dict):
            return {
                "name": functions["name"],
                "description": functions["description"],
                "parameters": {
                    k: v for k, v in functions["parameters"].get("properties", {}).items()
                }
            }
        elif isinstance(functions, list):
            return [self._fix_functions_format(f) for f in functions]
        else:
            return functions

    def _format_prompt(self, messages, function):
        # We first format the function signature and then add the messages
        function = self._fix_functions_format(function)

        formatted_prompt = f"""<|system|>
You are helpful AI assistant with tool calling capabilities.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{function}
</tools>

For each function call, return a json object with function name and arguments.
The output MUST strictly adhere to the following JSON format, and NO other text MUST be included.
The example format is as follows. Please make sure the parameter type is correct.
[
    {{"name": "func_name1", "arguments": {{"argument1": "value1", "argument2": "value2"}}}},
    ... (more tool calls as required)
]
"""
        
        for message in messages:
            formatted_prompt += f"<|user|>{message['content']}\n"

        formatted_prompt += "<|assistant|>\n"
        return formatted_prompt
    
    def decode_ast(self, result, language="Python"):
        # The output is a list of dictionaries, where each dictionary contains the function name and its arguments
        result = result.strip()
        result = result.replace("'", '"') # replace single quotes with double quotes
        result = json.loads(result)

        func_calls = []
        for item in result:
            function_name = item["name"]
            arguments = item["arguments"]
            func_calls.append({function_name: arguments})

        return func_calls
    
    def decode_execute(self, result):
        # The output is a list of dictionaries, where each dictionary contains the function name and its arguments
        result = result.strip()
        result = result.replace("'", '"') # replace single quotes with double quotes
        result = json.loads(result)

        # put the functions in format function_name(arguments)
        function_call_list = []
        for item in result:
            function_name = item["name"]
            arguments = item["arguments"]
            function_call_list.append(f"{function_name}({arguments})")

        execution_list = []
        for function_call in function_call_list:
            for key, value in function_call.items():
                execution_list.append(
                    f"{key}({','.join([f'{k}={repr(v)}' for k,v in value.items()])})"
                )
        return execution_list