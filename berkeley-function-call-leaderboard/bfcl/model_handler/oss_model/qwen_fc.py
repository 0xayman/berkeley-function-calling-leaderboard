import re
import json
from bfcl.model_handler.oss_model.base_oss_handler import OSSHandler

class QwenFcHandler(OSSHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)

    def _format_prompt(self, messages, function):
        formatted_prompt = f"""<|im_start|>system
        You are helpful AI assistant with tool calling capabilities.
        Current Date: 2024-11-15

        # Tools
        
        You may call one or more functions to assist with the user query.

        You are provided with functions signatures within <tools></tools> XML tags.
        <tools>
        {function}
        </tools>

        For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> XML tags as follows:
        <tool_call>
        {{"name": <function-name>, "arguments": <args-json-object>}}
        </tool_call>
        <|im_end|>
"""
        
        for message in messages:
            formatted_prompt += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"

        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt
    
    def decode_ast(self, result, language="Python"):
        lines = result.split("\n")
        flag = False
        func_call = []
        for line in lines:
            if "<tool_call>" == line:
                flag = True
            elif "</tool_call>" == line:
                flag = False
            else:
                if flag:
                    line = line.replace("'", '"')
                    tool_result = json.loads(line)
                    func_call.append({tool_result["name"]: tool_result["arguments"]})
                flag = False
        return func_call

    def decode_execute(self, result):
        lines = result.split("\n")
        flag = False
        function_call_list = []
        for line in lines:
            if "<tool_call>" == line:
                flag = True
            elif "</tool_call>" == line:
                flag = False
            else:
                if flag:
                    line = line.replace("'", '"')
                    tool_result = json.loads(line)
                    function_call_list.append(
                        {tool_result["name"]: tool_result["arguments"]}
                    )
                flag = False
        execution_list = []
        for function_call in function_call_list:
            for key, value in function_call.items():
                execution_list.append(
                    f"{key}({','.join([f'{k}={repr(v)}' for k,v in value.items()])})"
                )
        return execution_list