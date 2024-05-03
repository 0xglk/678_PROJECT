# Beat LLMs at Their Own Game: Zero-Shot LLM-Generated Text Detection via Querying ChatGPT
## Data
* For the Finance dataset, we use the ChatGPT-generated texts collected by Guo et al. [1]
* The human-written texts and ChatGPT-generated texts of the Finance dataset are in the finance.jsonl file.
* The ChatGPT-revised version for human-written texts of the Finance dataset is in the revised_human_finance.txt file.
* The ChatGPT-revised version for ChatGPT-generated texts of the Finance dataset is in the revised_chatgpt_finance.txt file.
* When we revise texts with ChatGPT, we use the
gpt-3.5-turbo API provided by OpenAI. All experiments that call gpt-3.5-turbo API in
the paper are done before June 2023, with the gpt-3.5-turbo API being gpt-3.5-turbo-0301.

## Installation
Set up your environment to run the detection code by following these steps:

 Install the required Python packages:
   ```bash
   pip install -r requirements.txt
```
Then run the code 
   ```bash
   python finance_bart_auroc.py
```

## Code
* To evaluate the detection performance of our method on the Finance dataset, please run the code "python finance_bart_auroc.py".
* Before running the program to enable **LLM functionality**, you need to add the ChatGPT API Key to the code. It's simply done by including it in the code.

**OPEN AND EDIT finance_bart_auroc.py**
   ```bash
   # OpenAI API key setup
client = openai.api_key='YOUR KEY HERE'
```
  
  We use the BARTScore-CNN [2] as
the similarity metric to calculate similarity scores.

### Key Addition:
- **Switching Compute Devices Section**: Added a section on how to switch the compute device from CPU to GPU for users who might want to utilize GPU acceleration. 
change on **bart_score.py** and **finance_bart_auroc.py** code 
Find the line in each script that specifies the device:
change it from "**device='cpu'** to **device='cuda:0'**"


 ## Reference
[1] Biyang Guo, Xin Zhang, Ziyuan Wang, Minqi Jiang,
Jinran Nie, Yuxuan Ding, Jianwei Yue, and Yupeng
Wu. 2023. How close is chatgpt to human experts?
comparison corpus, evaluation, and detection. arXiv
preprint arXiv:2301.07597.

[2] Weizhe Yuan, Graham Neubig, and Pengfei Liu. 2021.
Bartscore: Evaluating generated text as text generation. In Advances in Neural Information Processing
Systems, volume 34, pages 27263–27277. Curran Associates, Inc.
