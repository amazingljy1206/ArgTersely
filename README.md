Official repository for the paper ***Argue with Me Tersely: Towards Sentence-Level Counter-Argument
Generation***.



### ArgTersely

We introduce benchmark ArgTersely for sentence-level counter-argument generation task. The ArgTersely dataset is saved in folder `./argtersely`.

### Instruct-tuning the LLaMA with LoRA
The Alpaca instructions, Argumentation instructions, and seed instructions are saved in folder `./data`. Run `./finetune.py` to get the language model.

### Training the Filtering
The RD dataset is saved in folder `./filtering/rd`. Run `./filtering/train.py` to get the filtering.

### Generation Pipeline
The CoT instructions are saved in folder `./cot_instructions`. Run `./arg-llama/main.py` to generate the pipeline with CoT instructions and filtering.

### Validation of Arg-Judge
The QSD dataset is saved in `./qsd_valid/qsd.txt`. Run `./qsd_valid/qsd_valid_argjudge.py` to get the consistency between Arg-Judge and human evaluation.
