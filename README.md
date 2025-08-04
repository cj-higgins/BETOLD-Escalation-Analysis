# BETOLD Escalation Analysis

This repository contains code and outputs for analyzing the [BETOLD dataset](https://github.com/stanfordnlp/betold), a corpus of chatbot–user interactions annotated for escalation behavior and user distress. The analysis focuses on sentiment trajectory, escalation triggers, and LUHF-tagged dialogs.

> 📖 Related essay: [Consultation over Escalation](https://higginscj.substack.com/p/consultation-over-escalation)

---

## 📂 Files

- `BETOLD_escalation_analysis.py` — Main analysis script
- `BETOLD_clean_escalations_detailed.csv` — Full analysis of all conversations
- `BETOLD_escalation_only.csv` — Subset of conversations with escalation
- `BETOLD_escalation_dialogs.json` — Full dialog structure of escalated conversations

> **Note:** This repo does **not** contain the dataset. Download `BETOLD_train.json` from the [official repo](https://github.com/stanfordnlp/betold) and place it in the root folder.

---

## 🚀 How to Run

1. Clone the repo
2. Place `BETOLD_train.json` in the root directory
3. Run:

```bash
python BETOLD_escalation_analysis.py
```

---

## 📊 Output Column Descriptions

### 🔍 Conversation & Escalation Metadata

| Column | Description |
|--------|-------------|
| `conversation_id` | Unique ID per conversation |
| `luhf_tag` | LUHF classification (`luhf` or `non_luhf`) |
| `escalation` | "escalation" if `transfer_agent` occurred |
| `total_turns` | Total number of utterances |
| `user_turns_before_escalation` | NLU turns before first escalation |
| `total_user_turns` | All NLU (user) turns |

### 🧠 Intent Sentiment Counts

| Column | Description |
|--------|-------------|
| `nlu_positive_count` | # of positive NLU intents |
| `nlu_neutral_count` | # of neutral NLU intents |
| `nlu_negative_count` | # of negative NLU intents |
| `nlg_positive_count` | # of positive NLG intents |
| `nlg_neutral_count` | # of neutral NLG intents |
| `nlg_negative_count` | # of negative NLG intents |
| `nlg_neg_intents` | Comma-separated list of neg. NLG intents |
| `last_bot_intent` | Final bot intent in the conversation |

### 📈 Trajectory Scores

| Column | Description |
|--------|-------------|
| `nlu_trajectory_index` | Pos – neg NLU count |
| `nlg_trajectory_index` | Pos – neg NLG count |
| `composite_trajectory_index` | Sum of NLU + NLG trajectories |
| `nlu_density_index` | NLU trajectory ÷ user turns |
| `composite_density_index` | Composite ÷ total turns |
| `adjusted_composite_index` | Composite × exp(–0.05 × length) |

### 🧪 Final Turn Sentiment

| Column | Description |
|--------|-------------|
| `final4_turns_composite_score` | Final 4 turn sentiment score |
| `final4_score_with_length_penalty` | Final 4 × exp(–0.05 × length) |
| `pre_escalation_final4_score` | Final 4 score *before* escalation |

### ☎️ Escalation-Specific Heuristics

| Column | Description |
|--------|-------------|
| `last_turn_speaker` | Final speaker (`nlu` or `nlg`) |
| `transfer_assumed_success` | True if convo ends w/ user post-transfer |
| `escalation_final_turn_proximity` | True if transfer near convo end |
| `early_escalation_but_luhf` | LUHF-tagged & escalated early (≤3rd turn) |

---

## ⚖️ License

This code is released under the [Apache 2.0 License](LICENSE). The BETOLD dataset is also Apache 2.0 licensed; please see the [original repo](https://github.com/stanfordnlp/betold) for details.

---

## ✍️ Author

**CJ Higgins**  
[Substack – Curdled Incompleteness Theorem](https://higginscj.substack.com)  
[Website](https://cj-higgins.com)
