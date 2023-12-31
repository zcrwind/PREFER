# PREFER
This is the official implementation of the AAAI-2024 paper [*PREFER: Prompt Ensemble Learning via Feedback-Reflect-Refine*](https://arxiv.org/abs/2308.12033).

## Overview
![](./high_level_overview.png)


## Requirements
The dependency packages can be found in `requirements.txt` file. One can use `pip install -r requirements.txt` to configure the environment. We use python 3.8 to run the experiments. We highly recommend using the conda environment for deployment.


## Prepare the Data
Similar to our baselines (PromptBoosting, APO), we use the same datasets (SNLI, MNLI, QNLI, RTE, Ethos, Liar, ArSarcasm, etc). One can directly download the data from the official links provided by the paper [PromptBoosting: Black-box text classification with ten forward passes](https://arxiv.org/abs/2212.09257) and [Automatic Prompt Optimization with "Gradient Descent" and Beam Search](https://arxiv.org/abs/2305.03495).


## Template
If you want to evaluate PREFER on the different datasets, we need to replace the template in `template.py` (see the comments in the code files for more details). There are several modes one can select, e.g., adding in-context learning, adding Chain-of-Thought (CoT), etc.


## Running the experiments
The implementation of PREFER is in the `prefer.py` and `prefer_pair.py`. The difference between them is that the latter works for the pair-formed datasets, e.g., SNLI, MNLI, QNLI, RTE. 


## Main experiments
To run the codes, use the following command:
```{sh}
python prefer.py --adaboost_lr 1.0 --dataset ethos --max_error 0.8 --eval_trigger_threshold 0.9 --adaboost_weak_cls 6 --batch_size 8 --num_feedbacks 8 --generate_cnt 4 --num_monte_carlo 2 --patience 2 --timeout 10 --num_test 20 --average_mode macro
```
Explanations on the parameters:

`adaboost_lr`: the learning of the adaboost. In the paper, we use 1.0 for all experiments unless otherwise specified.

`dataset`: which dataset to evaluate, choices include `snli, mnli, qnli, rte, ethos, liar` etc.

`max_error`: 

`eval_trigger_threshold`: whether randomly change the prompt for each weak classifier training. If we use the same prompt for all weak classifiers, then the performance will be very weak. See Table 3 in the paper.

`adaboost_weak_cls`: the number of the weak learners for boosting.

`batch_size`: the number of instances in one batch.

`num_feedbacks`: how many feedbacks (reflection) will be generated by the LLM.

`generate_cnt`: how many new prompts will be generated by the LLM.

`num_monte_carlo`: the number of Monte Carlo sampling during the prompts selection.

`patience`: the maximum number of attempts to access the LLM interface.

`timeout`: the timeout limit for each request.

`num_test`: the number of the test instances.

`average_mode`: the average mode for evaluation, choices include macro and micro.


**Note**: Remember to set your app ID and url of ChatGPT/GPT-4 via modifing the global variables `APPID` and `URL`:

```
APPID = "YOUR APPID HERE"
URL = "YOUR API URL HERE"
```


## Citation
```tex
@article{zhang2023prefer,
  title={Prefer: Prompt ensemble learning via feedback-reflect-refine},
  author={Zhang, Chenrui and Liu, Lin and Wang, Jinpeng and Wang, Chuyuan and Sun, Xiao and Wang, Hongyu and Cai, Mingchen},
  journal={arXiv preprint arXiv:2308.12033},
  year={2023}
}
```
