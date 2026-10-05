# BETOLD Transfer-Request Analysis

This repository contains a small exploratory analysis of the [BETOLD dataset](https://github.com/telepathylabsai/BETOLD_dataset), a privacy-preserving corpus of chatbot–user customer-service conversations. The analysis asks whether conversations containing a user request for a human agent can nevertheless end with strongly positive interaction patterns.

The script uses the 10,819-conversation training split (`BETOLD_train.json`). The full BETOLD dataset contains 13,524 conversations.

> 📖 Related Substack essay: [Trusting the Chatbot More Than the Front Desk](https://higginscj.substack.com/p/trusting-the-chatbot-more-than-the)

## What the analysis does

BETOLD does not include raw conversation text. Instead, each turn is represented by an NLU or NLG intent. I therefore use a deliberately simple intent-valence proxy:

- selected positive intents = `+1`
- selected negative intents = `-1`
- all other intents = `0`

For each conversation, `final4_score` is the sum of those values over the final four turns. A score of 3 or higher therefore means that at least three of the final four turns were coded positive, with no offsetting negative turn large enough to bring the total below 3.

The script then:

1. identifies conversations containing the user intent `transfer_agent`;
2. compares the share of high-positive endings (`final4_score >= 3`) between conversations with and without a transfer request;
3. fits a simple logistic regression, `LUHF ~ final4_score`, as a sanity check on whether the proxy moves in the expected direction against BETOLD's late user-initiated forward/hang-up label; and
4. writes a compact conversation-level analysis file plus a transfer-request subset to `outputs/`.

## Headline descriptive result

In the training split:

- 87 conversations contain a `transfer_agent` request, and 15 of them (17.2%) have `final4_score >= 3`.
- 10,732 conversations do not contain a transfer request, and 2,704 of them (25.2%) have `final4_score >= 3`.

The point is not that a chatbot should ignore requests for a human. It is that a request for human assistance and a breakdown in the chatbot interaction are not necessarily the same thing.

## Scope and interpretation

- `transfer_agent` means the user **asked to transfer to a human agent**. It does not establish that a handoff actually occurred.
- `final4_score` describes the end of the whole conversation, not necessarily the interaction state at the moment the transfer was requested.
- `pre_transfer_final4_score` is included as a separate descriptive field for the four turns immediately before the first transfer request.
- The intent-valence score is an exploratory proxy, not a transcript-level sentiment model or a direct measure of customer satisfaction.
- The logistic regression is used as a check that the proxy contains behaviorally relevant signal, not as a causal model.

## How to run

1. Clone this repository.
2. Download `BETOLD_train.json` from the [official BETOLD repository](https://github.com/telepathylabsai/BETOLD_dataset) and place it in this repository's root directory.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run:

```bash
python BETOLD_escalation_analysis.py
```

The script prints the descriptive comparison and logistic-regression summary statistics, then creates:

- `outputs/BETOLD_analysis.csv`
- `outputs/BETOLD_transfer_requests.csv`

Generated outputs and the source dataset are intentionally excluded from version control.

## Output columns

| Column | Description |
|---|---|
| `conversation_id` | Row index within the training split |
| `luhf_tag` | BETOLD label: `luhf` or `not_luhf` |
| `transfer_requested` | Whether the conversation contains a user `transfer_agent` intent |
| `total_turns` | Number of turns in the conversation |
| `final4_score` | Intent-valence score over the final four turns |
| `pre_transfer_final4_score` | Intent-valence score over the final four turns before the first transfer request; for conversations without one, the final four turns of the conversation |

## Files

- `BETOLD_escalation_analysis.py` — complete analysis
- `requirements.txt` — Python dependencies
- `.gitignore` — excludes the source dataset and generated outputs
- `LICENSE` — Apache 2.0 license

## License

This code is released under the [Apache 2.0 License](LICENSE). The BETOLD dataset is also Apache 2.0 licensed; see the [official repository](https://github.com/telepathylabsai/BETOLD_dataset) for details.

## Author

**CJ Higgins**  
[Substack – Curdled Incompleteness Theorem](https://higginscj.substack.com)  
[Website](https://cj-higgins.com)
