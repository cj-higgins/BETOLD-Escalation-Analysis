import json
import math
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

DATA_PATH = Path("BETOLD_train.json")
OUTPUT_DIR = Path("outputs")
HIGH_POSITIVE_THRESHOLD = 3

# Simple, hand-coded intent valence used as an exploratory proxy.
POSITIVE_INTENTS = {
    "salutation",
    "confirm",
    "user_proposed_date",
    "schedule",
    "client_name",
    "confirm_date_scheduled",
    "reconfirm_date_scheduled",
    "confirm_canceled_appointment",
    "confirm_change_schedule",
    "offer_to_schedule",
}

NEGATIVE_INTENTS = {
    "reschedule",
    "negate",
    "urgency",
    "cancel",
    "transfer_agent",
    "did_not_understand",
    "fail_retrieve_user_info",
    "failed_schedule_warning",
    "no_dates_available",
    "no_more_schedule_appointments",
    "no_pre_existing_schedule",
    "disambiguate_user_profile",
    "silence",
    "date_schedule_no_longer_exists",
    "transportation_type_unavailable",
    "working_on_previous_request",
    "asked_date_too_far",
}


def score_intent(intent):
    """Map an intent to +1, 0, or -1. Unlisted intents are neutral."""
    if intent in POSITIVE_INTENTS:
        return 1
    if intent in NEGATIVE_INTENTS:
        return -1
    return 0


def score_last_four(utterances):
    """Sum intent-valence scores over the last four available turns."""
    return sum(score_intent(turn["intent"]) for turn in utterances[-4:])


def process_conversation(conversation_id, dialog):
    utterances = dialog["utterances_annotations"]
    transfer_indices = [
        i
        for i, turn in enumerate(utterances)
        if turn["caller_name"] == "nlu" and turn["intent"] == "transfer_agent"
    ]

    first_transfer_index = transfer_indices[0] if transfer_indices else None
    pre_transfer_turns = (
        utterances[:first_transfer_index]
        if first_transfer_index is not None
        else utterances
    )

    return {
        "conversation_id": conversation_id,
        "luhf_tag": dialog["LUHF"],
        "transfer_requested": bool(transfer_indices),
        "total_turns": len(utterances),
        "final4_score": score_last_four(utterances),
        "pre_transfer_final4_score": score_last_four(pre_transfer_turns),
    }


def fit_luhf_logistic_regression(df):
    """Fit LUHF ~ final4_score as a simple sanity check on the proxy."""
    y = (df["luhf_tag"] == "luhf").astype(int)
    x = sm.add_constant(df[["final4_score"]], has_constant="add")
    return sm.Logit(y, x).fit(disp=False)


def summarize_results(df):
    transfer = df[df["transfer_requested"]]
    no_transfer = df[~df["transfer_requested"]]

    transfer_high = (transfer["final4_score"] >= HIGH_POSITIVE_THRESHOLD).sum()
    no_transfer_high = (
        no_transfer["final4_score"] >= HIGH_POSITIVE_THRESHOLD
    ).sum()

    model = fit_luhf_logistic_regression(df)
    beta = model.params["final4_score"]
    odds_ratio = math.exp(beta)
    odds_reduction = 1 - odds_ratio

    print(f"Conversations analyzed: {len(df):,}")
    print(
        "Transfer requested: "
        f"{len(transfer):,} conversations; "
        f"{transfer_high / len(transfer):.1%} had final4_score >= "
        f"{HIGH_POSITIVE_THRESHOLD} ({transfer_high}/{len(transfer)})"
    )
    print(
        "No transfer requested: "
        f"{len(no_transfer):,} conversations; "
        f"{no_transfer_high / len(no_transfer):.1%} had final4_score >= "
        f"{HIGH_POSITIVE_THRESHOLD} ({no_transfer_high}/{len(no_transfer)})"
    )
    print("\nLUHF logistic regression: LUHF ~ final4_score")
    print(f"beta(final4_score) = {beta:.3f}")
    print(f"odds ratio = {odds_ratio:.3f}")
    print(f"odds reduction per one-point increase = {odds_reduction:.1%}")


def main():
    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = [
        process_conversation(conversation_id, dialog)
        for conversation_id, dialog in enumerate(data)
    ]
    df = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_DIR / "BETOLD_analysis.csv", index=False)
    df[df["transfer_requested"]].to_csv(
        OUTPUT_DIR / "BETOLD_transfer_requests.csv", index=False
    )

    summarize_results(df)


if __name__ == "__main__":
    main()
