# Model Card: GPT2-HistoryDST

## Model Description

**Model name:** GPT2-HistoryDST (Full History --> Cumulative Targets)  
**Model type:** Language model fine-tuned for Dialogue State Tracking  
**Base model:** GPT-2 ([openai/gpt-2](https://huggingface.co/gpt2), [OpenAI model card](https://github.com/openai/gpt-2/blob/master/model_card.md))  
**Language:** English  
**Task:** Task-oriented Dialogue State Tracking (DST)  
**Dataset:** MultiWOZ 2.1  
**Author:** omo26, University of Cambridge  
**WandB Training run:** `omo26-university-of-cambridge/Dialogue State Tracking/9sib5a10`

---

## Intended Use

This model is intended for **Dialogue State Tracking** in task-oriented dialogue systems. Given the full dialogue history up to the current turn, the model generates a linearized cumulative belief state — a structured representation of all slot-value pairs mentioned by the user across the entire dialogue so far.

**Primary use case:** Academic research and evaluation on the MultiWOZ 2.1 benchmark (and hopefully getting me a good MLMI8 grade).

**Out-of-scope uses:**
- Domains outside those covered by MultiWOZ (restaurant, hotel, attraction, taxi, train)
- Languages other than English

---

## Training

### Data
- **Training set:** MultiWOZ 2.1 training split (7,888 dialogues, 54,971 turns)
- **Development set:** MultiWOZ 2.1 development split (1,000 dialogues, 7,374 turns)

Each training example consists of:
- **Input (`dst_input`):** The full dialogue history up to and including the current turn, formatted as a sequence of `<SYS>` and `<USR>` tagged utterances
- **Target (`belief_state`):** The cumulative dialogue-level belief state at that turn, linearized as `domain slot value <SEP> domain slot value ...` sorted alphabetically

Cumulative belief states for the training split were constructed by accumulating turn-level annotations from the MultiWOZ 2.1 data, with corrections handled via the `nlu_correction` field.

### Procedure
- **Fine-tuning approach:** Supervized fine-tuning from the pre-trained HuggingFace GPT-2 checkpoint
- **Optimizer:** AdamW
- **Batch size:** 8
- **Hardware:** NVIDIA A100-SXM4-80GB
- **Training time:** 48 minutes 6 seconds
- **Framework:** HuggingFace Transformers, PyTorch
- **Python version:** CPython 3.10.12

### Key design decisions
- Full dialogue history is provided as input at every turn, rather than just the current turn pair. This allows the model to resolve coreferences and context-dependent expressions (e.g. "same day", "same price range") that cannot be resolved from a single turn in isolation.
- The model is trained to predict the **cumulative** belief state directly, rather than predicting turn-level annotations that are subsequently concatenated. This avoids the exposure bias problem inherent in pipeline approaches such as Cheap-and-Cheerful DST (CC-DST), where errors in early turns propagate through the history at inference time.
- Slot-value pairs in the target belief state are sorted **alphabetically** to ensure a consistent, deterministic output ordering that matches the MultiWOZ reference annotation format.

---

## Evaluation

Evaluated on the MultiWOZ 2.1 test and development sets using **DST Average Joint Accuracy** — a turn is counted as correct only if the predicted belief state exactly matches the reference cumulative belief state at that turn.

### Results

| Step | Dev Joint Accuracy (%) | Test Joint Accuracy (%) |
|------|----------------------|------------------------|
| 60,000 | 51.10 | 49.29 |
| 100,000 | 52.45 | 51.26 |
| 120,000 | 51.06 | 50.15 |
| 160,000 | 51.91 | 50.52 |
| 180,000 | 51.67 | 50.84 |
| **200,000** | **52.41** | **51.19** |

**Best checkpoint:** model.100000 (highest dev accuracy: 52.45%, test: 51.26%)

---

## Limitations

**Trained only on MultiWOZ:** The model has been fine-tuned exclusively on MultiWOZ 2.1, which covers five domains (restaurant, hotel, attraction, taxi, train) in a Cambridge, UK setting. Performance on other domains, languages, or dialogue styles has not been evaluated and is expected to be poor without further fine-tuning.

**Bad format outputs:** The model occasionally generates malformed belief state items — for example, outputting a domain token (e.g. `hotel`) without an accompanying slot or value. These are treated as errors by the scoring script and contribute to accuracy degradation. This suggests the model has not fully learned the linearization format in all cases, particularly for complex multi-domain dialogues.

**Sequence length truncation:** The maximum input sequence length is 1,024 tokens. For long multi-domain dialogues, early turns may be truncated from the left, meaning the model loses access to information mentioned at the start of the conversation. This is a known limitation of using a fixed-context decoder model for dialogue history encoding.

---

## Related Models and References

- **Base model:** [HuggingFace GPT-2](https://huggingface.co/gpt2)
- **OpenAI GPT-2 model card:** https://github.com/openai/gpt-2/blob/master/model_card.md
- **MultiWOZ dataset:** Budzianowski et al., "MultiWOZ - A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling", EMNLP 2018
- **HuggingFace Transformers:** Wolf et al., https://arxiv.org/abs/1910.03771