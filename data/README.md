### Instruction Dataset Description

Based on Self-Instruct method, we use ChatGPT to generate an ***Argumentation Instruction Set***. It contains 2772 instances, which is highly related to logical instructions. 

During instruct-tuning, we combine the Alpaca instructions and Argumentation instructions as training data.

The Alpaca instructions are saved in `alpaca_data.json`; The combination of Alpaca and Argumentation instructions are saved in `alpaca_argument.json`; 10 seed instructions are saved in `seed_instructions.json`.