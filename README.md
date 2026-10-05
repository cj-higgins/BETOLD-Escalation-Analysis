# BETOLD Escalation Analysis

This repository contains code and saved outputs from an exploratory analysis of the [BETOLD dataset](https://github.com/telepathylabsai/BETOLD_dataset), a privacy-preserving corpus of chatbot–user customer-service interactions labeled for late user-initiated forwards or hang-ups (LUHF). The script in this repository analyzes the 10,819-conversation training split and focuses on late-conversation intent valence, user requests for a human agent, and LUHF.

> 📖 Related Substack essay: [Consultation over Escalation](https://higginscj.substack.com/p/trusting-the-chatbot-more-than-the)

---

## 📂 Files

- `BETOLD_escalation_analysis.py` — Main analysis script
- `BETOLD_clean_escalations_detailed.csv` — Analysis of all 10,819 conversations in `BETOLD_train.json`
- `BETOLD_escalation_only.csv` — Subset containing a `transfer_agent` user intent
- `BETOLD_escalation_dialogs.json` — Full dialog structure for that subset

> **Note:** This repo does **not** contain the source dataset. Download `BETOLD_train.json` from the [official repo](https://github.com/telepathylabsai/BETOLD_dataset) and place it in the root folder.

## Scope and interpretation

- The full BETOLD dataset contains 13,524 conversations; this script uses only the 10,819-conversation training split.
- The `escalation` field is triggered by the user intent `transfer_agent`. It therefore marks a **request for a human agent**, not a verified completed handoff.
- The sentiment proxy explicitly codes selected intents as positive, neutral, or negative; intents not listed in those mappings default to neutral.
- `final4_turns_composite_score` describes the final four turns of the **whole conversation**. `pre_escalation_final4_score` separately describes the final four turns before the first `transfer_agent` request.
- These are exploratory intent-based proxies, not validated measures of rapport or customer satisfaction.

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
| `escalation` | `"escalation"` if the user intent `transfer_agent` occurred |
| `total_turns` | Total number of utterances |
| `user_turns_before_escalation` | NLU turns before first transfer request |
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
| `pre_escalation_final4_score` | Final 4 score before first transfer request |

### ☎️ Escalation-Specific Heuristics

| Column | Description |
|--------|-------------|
| `last_turn_speaker` | Final speaker (`nlu` or `nlg`) |
| `transfer_assumed_success` | Heuristic: transfer request present and conversation ends on a user turn |
| `escalation_final_turn_proximity` | True if the last transfer request occurs near conversation end |
| `early_escalation_but_luhf` | LUHF-tagged & transfer requested early (≤3rd turn) |

---

## ⚖️ License

This code is released under the [Apache 2.0 License](LICENSE). The BETOLD dataset is also Apache 2.0 licensed; please see the [official repo](https://github.com/telepathylabsai/BETOLD_dataset) for details.

---

## ✍️ Author

**CJ Higgins**  
[Substack – Curdled Incompleteness Theorem](https://higginscj.substack.com)  
[Website](https://cj-higgins.com)
