任务:完整的任务，step1和step2。<br>
step1:通过贪心算法来选择coverage更大的5个5-shot prompt<br>
step2:构建prompt selector来动态选择prompt

选出来的5-shot prompt：
|  prompt   | acc on dev  | acc on test |
|  ----  | ----  | ----| 
| prompt_0  | 68.67% |66.26%|
| prompt_1  | 70.53% |69.45%|
| prompt_2  | 68.29% |68.76%|
| prompt_3  | 70.33% |68.39%|
| prompt_4  | 69.88% |68.31%|
| 5 * prompts | 86.81% |86.88%|
| bert-base-uncased |  |71.42%|
| xlm-roberta-large  |  ||
| deberta-v3-large  |  ||

