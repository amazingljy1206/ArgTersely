import os
import sys
import json
import fire
import torch
from tqdm import tqdm
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from utils.prompter import Prompter
import numpy as np

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = True,
    base_model: str = "./llama-7b-hf",
    lora_weights: str = "./models/argllama",
    prompt_template: str = "arg_llama"  # The prompt template to use, will default to alpaca.
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
#-------*******************************-------
    prompter = Prompter(prompt_template)

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.
        model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)
    def clean(text):
        text = text.strip()
        if '\n' not in text:
            return text
        else:
            return ' '.join(text.split('\n'))    
    filtering_model = torch.load("./filtering/models/bert-base-finetuned.pt")
    with open('./arg-llama/cot_instructions/factual_prompt.json') as f:
        dicts = json.load(f)
        f_dict1, f_dict2 = dicts[0], dicts[1]
    with open('./arg-llama/cot_instructions/logical_prompt.json') as f:
        dicts = json.load(f)
        l_dict1, l_dict2 = dicts[0], dicts[1]
    with open('./arg-llama/cot_instructions/bias_prompt.json') as f:
        dicts = json.load(f)
        c_dict1, c_dict2 = dicts[0], dicts[1]
    with open('ArgTersely/test/argument.txt', 'r', encoding='UTF-8') as f:
        lines = f.readlines()
    instruction_prompt = 'Example:\nInput: {}\nCounter-argument: {}'
    for line in tqdm(lines):
        arg = line.strip()
        output1 = evaluate(instruction=instruction_prompt.format(f_dict1['input'], 
                                                                 f_dict1['output']),
                        	input = arg)
        output2 = evaluate(instruction=instruction_prompt.format(f_dict2['input'], 
                                                                 f_dict2['output']),
                        	input = arg)        
        output3 = evaluate(instruction=instruction_prompt.format(l_dict1['input'], 
                                                                 l_dict1['output']),
                        	input = arg)
        output4 = evaluate(instruction=instruction_prompt.format(l_dict2['input'], 
                                                                 l_dict2['output']),
                        	input = arg)
        output5 = evaluate(instruction=instruction_prompt.format(c_dict1['input'], 
                                                                 c_dict1['output']),
                        	input = arg)
        output6 = evaluate(instruction=instruction_prompt.format(c_dict2['input'], 
                                                                 c_dict2['output']),
                        	input = arg)
        output_list = list(map(lambda text: clean(text), [output1, output2, output3, output4, output5, output6]))
        output = filtering_model.inference(arg, output_list)
        print(output)
        with open('Experiments/argllama_res.txt', 'a+', encoding='UTF-8') as f:
            f.write(output + '\n')

    
if __name__ == "__main__":
    fire.Fire(main)