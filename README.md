# SmallCall: Post-Training a Small LLM for Tool-Calling

LoRA SFT of **Qwen2.5-1.5B-Instruct** so it knows **when not to call a tool**, benchmarked against Qwen2.5-7B on held-out tools (xLAM dataset, BFCL-style scoring).

| 550 held-out test questions | Before | After SFT | 7B reference |
|---|---|---|---|
| Calling accuracy (fair) | 84.5% | 85.2% | 87.5% |
| "No suitable tool" accuracy | 76.0% | **96.7%** | 87.3% |

**Finding:** the 1.5B model matched the 7B on calling (~3 pts apart) but trailed by 11.3 pts on refusals. Targeted SFT fixed 31 refusal errors (0 new) and surpassed the 7B, with no calling regression.

**Method:** 15% of tools held out of training; "no tool" cases made by deleting the correct tool; LoRA rank 16 on all attention + MLP layers (1.2% of params), 2 epochs, best epoch picked on validation; resumable Colab training.

**Remaining errors (64):** 46 wrong arguments, 6 text instead of call, 5 wrong call count, 5 called when none fit, 2 other. Next step: argument-focused SFT or constrained decoding.

## Keywords
- **SFT** – train on query → correct output pairs.
- **LoRA / PEFT** – train small adapter matrices, not all weights.
- **Tool calling** – model outputs a function name + JSON arguments.
- **Held-out tools** – unseen at training, so the test measures generalization.
- **Refusal accuracy** – correctly making no call when no tool fits.
- **Fair vs strict scoring** – fair ignores omitted default/empty arguments.
- **Failure taxonomy** – errors grouped by type to pick the next fix.

## Limits
One SFT round, no RLHF/GRPO. A gpt-oss-120b (Groq) comparison stopped at 78/200 (rate limit) and is excluded.

## Files
`SmallCall_Stage1_Baseline_v2 (1).ipynb` (Colab, needs `HF_TOKEN`) · `SmallCall_Project_Details.md` (full write-up)
