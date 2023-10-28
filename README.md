## [LLM Lies: Hallucinations are not Bugs, but Features as Adversarial Examples](http://arxiv.org/abs/2310.01469)

<div align="center">
    <a href="http://arxiv.org/abs/2310.01469">
        <img alt="arXiv" src="https://img.shields.io/badge/Arxiv-2310.01469-b31b1b.svg?logo=arXiv" />
    </a>
    <a href="https://github.com/PKU-YuanGroup/Hallucination-Attack/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/Code%20License-MIT-yellow" />
    </a>
    <a href="https://zhuanlan.zhihu.com/p/661444210">
        <img alt="zhihu" src="https://img.shields.io/badge/知乎-0084FF" />
    </a>
</div>

### Brief Intro
LLMs (e.g., GPT-3.5, LLaMA, and PaLM) suffer from **hallucination**&mdash;fabricating non-existent facts to cheat users without perception.
And the reasons for their existence and pervasiveness remain unclear.
We demonstrate that non-sense Out-of-Distribution(OoD) prompts composed of random tokens can also elicit the LLMs to respond with hallucinations.
This phenomenon forces us to revisit that **hallucination may be another view of adversarial examples**, and it shares similar features with conventional adversarial examples as the basic feature of LLMs.
Therefore, we formalize an automatic hallucination triggering method called **hallucination attack** in an adversarial way.
Following is a fake news example generating by hallucination attack.

#### Hallucination Attack generates fake news
<div align="center">
  <img src="assets/example-fake.png" width="100%">
</div>

### Weak Semantic Prompt & OoD Prompt
Weak semantic prompt and OoD prompt can elicit the Vicuna-7B to reply the same fake fact.
<div align="center">
  <img src="assets/fig1.png" width="100%">
</div>

### What Leads to Hallucination
We record some important milestones during the optimization process. We find that some “trigger” tokens are semantically induced, such as replacing “cabe” with “Barry”, as we hope the LLMs can ultimately output “The founder of Apple is Barry Diller”. However, many token swaps often have no semanticity, like “junl→empress” and “decidOsais→decidareais”. As a result, we finally optimize a seemingly meaningless prompt for humans, which however elicits the LLMs to respond with pre-defined hallucinations.
<div align="center">
  <img src="assets/fig2.png" width="100%">
</div>

### The Pipeline of Hallucination Attack 
We substitute tokens via gradient-based token replacing strategy, replacing token reaching smaller negative log-likelihood loss, and induce LLM within hallucinations.
<div align="center">
  <img src="assets/fig3.png" width="100%">
</div>

### Results of hallucination attack
#### - Weak Semantic Attack
<div align="center">
  <img src="assets/weak-semantic-attack.jpg" width="100%">
</div>

#### - OoD Attack
<div align="center">
  <img src="assets/ood-attack.jpg" width="100%">
</div>

### Usage
#### Models
You may config your own base models and their hyper-parameters within `config.py`. Then, you could attack the models or run our demo cases.

#### Demo
Clone this repo and run the code.
```bash
$ cd Hallucination-Attack
```
Install the requirements.
```bash
$ pip install -r requirements.txt
```
Run local demo of hallucination attacked prompt.
```bash
$ python demo.py
```

#### Attack
Start a new attack training to find a prompt trigger hallucination
```bash
$ python main.py
```

### Citation
```BibTeX
@misc{yao2023llm,
      title={LLM Lies: Hallucinations are not Bugs, but Features as Adversarial Examples}, 
      author={Jia-Yu Yao and Kun-Peng Ning and Zhen-Hui Liu and Mu-Nan Ning and Li Yuan},
      year={2023},
      eprint={2310.01469},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
