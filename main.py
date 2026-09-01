import json

from groq import Groq
import os
from geminiAi import solve_arc as solve_arc_gemini
from nvidiaApi import solve_arc as solve_arc_nvidia
from huggingfaceApi import solve_arc as solve_arc_huggingface
from openrouterApi import solve_arc as solve_arc_openrouter
from groqApi import solve_arc as solve_arc_groq


client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def load_task(path):
    with open(path) as f:
        return json.load(f)
    

def grid_to_text(grid):
    return "\n".join(" ".join(map(str,row)) for row in grid)


def build_prompt(task):

    prompt = "You are solving an ARC reasoning task.\n\n"

    for i,example in enumerate(task["train"]):

        prompt += f"Example {i+1}\n"
        prompt += "Input:\n"
        prompt += grid_to_text(example["input"])
        prompt += "\n\nOutput:\n"
        prompt += grid_to_text(example["output"])
        prompt += "\n\n"

    test = task["test"][2]["input"]

    prompt += "Test Input:\n"
    prompt += grid_to_text(test)

    prompt += """
    
Explain the transformation rule that could also applied to both example and test input and produce the output grid only in the following format. No further explanations or comments, only the JSON object with the rule and output grid.
Do not explain your reasoning. Do not output <think> tags.


Return JSON:
{
 "rule": "...",
 "output": [[...]]
}
"""

    return prompt

def solve_arc(prompt):
        
        #solve_arc_gemini(prompt)
        #solve_arc_nvidia(prompt)
        #solve_arc_huggingface(prompt)
        solve_arc_openrouter(prompt)
        #solve_arc_groq(prompt)


    

jsonTrainData = load_task("trainingData.json")
"""
for example in jsonTrainData["train"]:

    input_grid = example["input"]
    output_grid = example["output"]

    print(grid_to_text(input_grid))

"""

colorBasedPrompt = """
You will be tasked with solving tasks involving input grids with objects that transform into output grids. Your goal is to dicern the transformations
applied to the input to achieve the corresponding output.

Example: input: "black, black, black, black\n magenta, black, red, black\n black, black, red, black\n black, black, black, black"
        output: "black, black, black, black\n black, magenta, red, black\n black, black, red, black\n black, black, black, black"
        Answer: Transformation applied: 1. Move color 6 object to color 2 object until they touch.
Please answer by providing the output and answer like in the example. And please output also the linebreak "\n" if there is any for better readability.

Task:
Training Example 1
Training Example 1
Input:
"black, black, blue, black, black, yellow\n black, blue, black, black, black, yellow\n blue, black, black, black, black, yellow\n black, black, black, black, black, black\n magenta, black, black, red, red, red\n magenta, black, black, black, black, black"

Output:
"black, black, black, black, black, yellow\n black, black, black, black, black, yellow\n black, black, black, black, black, yellow\n black, black, black, black, black, black\n magenta, black, black, black, black, black\n magenta, black, black, black, black, black"

Training Example 2
Input:
"black, black, green, black, black, black\n black, green, black, red, red, red\n green, black, black, black, black, black\n black, black, black, black, brown, black\n orange, orange, orange, black, brown, black\n black, black, black, black, brown, black"

Output:
"black, black, black, black, black, black\n black, black, black, red, red, red\n black, black, black, black, black, black\n black, black, black, black, black, black\n orange, orange, orange, black, black, black\n black, black, black, black, black, black"

Test:
Input:
"black, black, blue, black, black, black\n black, blue, black, black, black, black\n blue, black, black, black, black, gray\n black, black, black, black, gray, black\n black, orange, black, black, black, black\n black, orange, black, green, green, green"

What would be the output and the transformation rule?
"""

prompt = build_prompt(jsonTrainData)
print(prompt + "\n\n")
response = solve_arc(prompt)
#print("Response:" + response)