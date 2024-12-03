import re
import json
from bfcl.model_handler.oss_model.base_oss_handler import OSSHandler

class QwenCotHandler(OSSHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)

    def _convert_functions_to_qwen_format(self, functions):
        if isinstance(functions, dict):
            return {
                "name": functions["name"],
                "description": functions["description"],
                "parameters": {
                    k: v for k, v in functions["parameters"].get("properties", {}).items()
                }
            }
        elif isinstance(functions, list):
            return [self._convert_functions_to_qwen_format(f) for f in functions]
        else:
            return functions

    def _format_prompt(self, messages, function):
        # We first format the function signature and then add the messages
        function = self._convert_functions_to_qwen_format(function)

        formatted_prompt = f"""<|im_start|>system
You are helpful AI assistant with tool calling capabilities.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{function}
</tools>

# Your Process:
1. Analyze the query step by step to understand the user's intent.
2. Identify the required functions and arguments needed to address the query.
3. Validate the arguments to ensure their types and values are correct. If arguments are missing, use defaults if available or infer values based on context.
4. Construct the function calls in strict compliance with the JSON format.

# Example Output Format:
Your output MUST strictly adhere to the following JSON format:
{{
    "thought": "Step-by-step reasoning for the query..."
    "function_calls": [
        {{"name": "func_name1", "arguments": {{"argument1": "value1", "argument2": "value2"}}}},
        ... (more tool calls as required)
    ]
}}
<|im_end|>
"""
        
        for message in messages:
            formatted_prompt += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"

        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt
    
    def get_json(self, result):
        # The output is a list of dictionaries, where each dictionary contains the function name and its arguments
        result = result.strip()

        result = json.loads(result)
        result = result["function_calls"]
        result = json.loads(result)

        return result
    
    def decode_ast(self, result, language="Python"):
        # The output is a list of dictionaries, where each dictionary contains the function name and its arguments
        result = self.get_json(result)

        func_calls = []
        for item in result:
            function_name = item["name"]
            arguments = item["arguments"]
            func_calls.append({function_name: arguments})

        return func_calls
    
    def decode_execute(self, result):
        # The output is a list of dictionaries, where each dictionary contains the function name and its arguments
        result = self.get_json(result)

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