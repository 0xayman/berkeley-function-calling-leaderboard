import json
from bfcl.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override

class QwenDistilHandler(OSSHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)

    def _convert_functions_format(self, functions):
        if isinstance(functions, dict):
            return {
                "name": functions["name"],
                "description": functions["description"],
                "parameters": {
                    k: v for k, v in functions["parameters"].get("properties", {}).items()
                }
            }
        elif isinstance(functions, list):
            return [self._convert_functions_format(f) for f in functions]
        else:
            return functions

    def _extract_json_from_text(self, text):
    # Split the text at the closing </think> tag and take the part after it
        split_text = text.split('</think>', 1)
        
        # If the split was successful (there was a </think> tag), process the second part
        if len(split_text) > 1:
            # Strip whitespace from the remaining text to isolate the JSON
            json_str = split_text[1].strip()
        else:
            # Return empty string if no </think> tag found (or handle differently if needed)
            json_str = "[]"
        
        return json_str

    @override
    def _format_prompt(self, messages, function):
        # We first format the function signature and then add the messages
        tools = self._convert_functions_format(function)

        formatted_prompt = f"""<|im_start|>system
You are helpful AI assistant with tool calling capabilities.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

Respond in the format {{"name": function name, "parameters": dictionary of argument name and its value}}.Do not use variables.
<|im_end|>
"""
        
        for message in messages:
            formatted_prompt += "<|im_start|>user"
            formatted_prompt += f"{message['content']}<|im_end|>\n"

        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt
    
    @override
    def decode_ast(self, result, language="Python"):
        # The output is a list of dictionaries, where each dictionary contains the function name and its arguments
        result = result.strip()
        result = result.replace("'", '"') # replace single quotes with double quotes
        result = self._extract_json_from_text(result)
        result = json.loads(result)

        func_calls = []
        for item in result:
            function_name = item["name"]
            arguments = item["arguments"]
            func_calls.append({function_name: arguments})

        return func_calls
    
    @override
    def decode_execute(self, result):
        # The output is a list of dictionaries, where each dictionary contains the function name and its arguments
        result = result.strip()
        result = result.replace("'", '"') # replace single quotes with double quotes
        result = self._extract_json_from_text(result)
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