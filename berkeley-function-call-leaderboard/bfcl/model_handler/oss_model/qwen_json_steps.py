import re
import json
from bfcl.model_handler.oss_model.base_oss_handler import OSSHandler

class QwenJsonStepsHandler(OSSHandler):
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
You are a helpful AI assistant with the ability to call tools. Your responses should be accurate, structured, and reasoned step-by-step.

### Instructions:  
You will be given a **user query** and a set of **tools**. Your task is to follow these steps:

1. **Understand the query**: Analyze the user query to determine its intent.

2. **Identify relevant tools**: Review the available tools provided as a JSON list between <tools></tools> XML tags. List the names of the tools that could help answer the query.

3. **Evaluate required information**: For each relevant tool:  
   - Check if all the necessary arguments for that tool are present in the query.  

4. **Generate tool call(s)**: For queries requiring tool calls:  
   - Construct the tool call(s) using the appropriate tool name(s) and arguments.  
   - Format the tool call(s) as follows:  
     <tool_calls>
     [
         {{"name": "tool_name", "arguments": {{"arg1": "value1", "arg2": "value2"}}}},
         {{"name": "another_tool", "arguments": {{"argX": "valueX"}}}}
     ]
     </tool_calls>
<|im_end|>

### Examples:
- **User Query**: "What is the weather in New York?"  
- **Available Tools**:  
<tools>
[
    {{"name": "WeatherLookup", "description": "Get the current weather for a specific location.", "parameters": {{"location": {{"type": "string", "description": "The location to get the weather for."}}}}}},
    {{"name": "TimeLookup", "description": "Get the current time for a specific location.", "parameters": {{"location": {{"type": "string", "description": "The location to get the time for."}}}}}}
]
</tools>
- **Response**:  
Step 1: The query asks for the weather in New York, which requires using the `WeatherLookup` tool.  
Step 2: The relevant tool is `WeatherLookup`.  
Step 3: The query provides the required information (`location` = "New York").  
Step 4: Construct the tool call:  
<tool_calls>
[
    {{"name": "WeatherLookup", "arguments": {{"location": "New York"}}}}
]
</tool_calls>
<|im_end|>
"""
        
        for message in messages:
            formatted_prompt += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"

        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt
    
    def decode_ast(self, result, language="Python"):
        # The output is a list of dictionaries, where each dictionary contains the function name and its arguments
        result = result.strip()

        # extract the tool calls from the result, between the <tool_calls></tool_calls> tags
        tool_calls = re.search(r'<tool_calls>(.*?)</tool_calls>', result, re.DOTALL)
        if tool_calls:
            tool_calls = tool_calls.group(1)
            tool_calls = tool_calls.replace("'", '"')
            tool_calls = json.loads(tool_calls)
        else:
            tool_calls = []

        result = tool_calls
        

        func_calls = []
        for item in result:
            function_name = item["name"]
            arguments = item["arguments"]
            func_calls.append({function_name: arguments})

        return func_calls
    
    def decode_execute(self, result):
        # The output is a list of dictionaries, where each dictionary contains the function name and its arguments
        result = result.strip()

        # extract the tool calls from the result, between the <tool_calls></tool_calls> tags
        tool_calls = re.search(r'<tool_calls>(.*?)</tool_calls>', result, re.DOTALL)
        if tool_calls:
            tool_calls = tool_calls.group(1)
            tool_calls = tool_calls.replace("'", '"')
            tool_calls = json.loads(tool_calls)
        else:
            tool_calls = []

        result = tool_calls

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