# State-of-the-art Contrastive Decoding Strategies Implementation

Contrastive Decoding is a kind of training-free strategy in the LLMs generation process for mitigating hallucination. Here, I implement some SOTA contrastive decoding strategies based on LLaVA, lmms-eval, and transformers.

## Setup

First, create a new environment with conda and clone the repository

```
conda create -n cd python=3.10 -y
git clone git@github.com:Fanziyang-v/contrastive-decoding.git
```

Second, install the package using pip with editable option

```
pip install -e lmms-eval
pip install -e LLaVA
pip install -e transformers
```

## Implementation

Currently, Visual Contrastive Decoding(VCD) is implemented. The other contrastive decoding strategies such OPEAR and DOLA will be implemented soon.

## Evaluation

Here is an example on using lmms-eval to evaluate LLaVA v1.5-7B with VCD on POPE dataset.

```
python3 -m accelerate.commands.launch 
    --num_processes=4 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-7b" \
    --gen_kwargs strategy="VCD",alpha=1.0,beta=0.1,noise_step=999 \
    --tasks pope_adv,pope_pop,pope_random \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_pope_vcd \
    --output_path ./logs/
```
