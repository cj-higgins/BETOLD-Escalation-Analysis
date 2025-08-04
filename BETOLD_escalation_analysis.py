import json
import csv
import math

# === Define intent sentiment classifications ===
nlu_positive_intents = {
    'salutation', 'confirm', 'user_proposed_date', 'schedule', 'client_name'
}
nlu_neutral_intents = {
    'inform', 'rephrase', 'inconclusive', 'unregistered_user', 'user_initial_request',
    'other', 'noise', 'ask_for_transportation_types', 'language', 'model_device',
    'phone_number', 'sector', 'time_indication', 'time_range_indication',
    'transportation_type', 'type_of_repair', 'year', 'brand_device'
}
nlu_negative_intents = {
    'reschedule', 'negate', 'urgency', 'cancel', 'transfer_agent'
}
escalation_intents = {'transfer_agent'}

nlg_positive = {
    'confirm_date_scheduled', 'reconfirm_date_scheduled', 'confirm_canceled_appointment',
    'confirm_change_schedule', 'offer_to_schedule'
}
nlg_neutral = {
    'ask_first_name', 'ask_last_name', 'ask_phone_number', 'ask_time_preference',
    'intro_assistant_1', 'intro_assistant_2', 'inform_schedule_inspection',
    'faq_open_time', 'faq_close_time', 'faq_operating_hours', 'ask_device_brand',
    'ask_device_model', 'ask_device_year', 'ask_if_current_client'
}
nlg_negative = {
    'did_not_understand', 'fail_retrieve_user_info', 'failed_schedule_warning',
    'no_dates_available', 'no_more_schedule_appointments', 'no_pre_existing_schedule',
    'disambiguate_user_profile', 'silence', 'date_schedule_no_longer_exists',
    'transportation_type_unavailable', 'working_on_previous_request', 'asked_date_too_far'
}

# === Load dataset ===
with open('BETOLD_train.json', 'r') as f:
    data = json.load(f)

output_rows = []
escalation_rows = []
escalation_dialogs = []

# === Process each conversation ===
for convo_id, dialog in enumerate(data):
    utterances = dialog['utterances_annotations']
    luhf_tag = dialog['LUHF']

    nlu_sentiments = []
    nlg_sentiments = []
    user_turns_before_escalation = 0
    total_user_turns = 0
    nlg_pos = nlg_neg = nlg_neu = 0
    nlg_neg_intents = []
    escalation_turns = []
    last_bot_intent = None

    for i, turn in enumerate(utterances):
        speaker = turn['caller_name']
        intent = turn['intent']

        if speaker == 'nlu':
            total_user_turns += 1

            if intent in escalation_intents:
                escalation_turns.append(i)

            if not escalation_turns:
                user_turns_before_escalation += 1

            if intent in nlu_positive_intents:
                nlu_sentiments.append("positive")
            elif intent in nlu_neutral_intents or intent.lower().startswith("q_"):
                nlu_sentiments.append("neutral")
            elif intent in nlu_negative_intents:
                nlu_sentiments.append("negative")
            else:
                nlu_sentiments.append("neutral")

        elif speaker == 'nlg':
            last_bot_intent = intent
            if intent in nlg_negative:
                nlg_neg += 1
                nlg_sentiments.append("negative")
                nlg_neg_intents.append(intent)
            elif intent in nlg_positive:
                nlg_pos += 1
                nlg_sentiments.append("positive")
            elif intent in nlg_neutral:
                nlg_neu += 1
                nlg_sentiments.append("neutral")
            else:
                nlg_neu += 1
                nlg_sentiments.append("neutral")

    nlu_trajectory_index = nlu_sentiments.count("positive") - nlu_sentiments.count("negative")
    nlg_trajectory_index = nlg_pos - nlg_neg
    composite_trajectory_index = nlu_trajectory_index + nlg_trajectory_index

    total_turns = len(utterances)
    nlu_density_index = nlu_trajectory_index / max(total_user_turns, 1)
    composite_density_index = composite_trajectory_index / total_turns
    adjusted_composite_index = composite_trajectory_index * math.exp(-0.05 * total_turns)

    # === Sentiment analysis for final 4 turns of conversation ===
    def get_sentiment_from_intent(intent):
        if intent in nlu_positive_intents or intent in nlg_positive:
            return "positive"
        elif intent in nlu_negative_intents or intent in nlg_negative:
            return "negative"
        elif intent in nlu_neutral_intents or intent in nlg_neutral:
            return "neutral"
        else:
            return "neutral"

    final_4_turns = utterances[-4:]
    final_sentiments = [get_sentiment_from_intent(t['intent']) for t in final_4_turns]
    final4_composite_score = sum({ "positive": 1, "neutral": 0, "negative": -1 }.get(s, 0) for s in final_sentiments)
    final4_score_with_length_penalty = final4_composite_score * math.exp(-0.05 * total_turns)

    # === Sentiment analysis for final 4 turns before escalation ===
    pre_escalation_turns = utterances[:escalation_turns[0]] if escalation_turns else utterances
    final_4_before_esc = pre_escalation_turns[-4:]
    pre_esc_final_sentiments = [get_sentiment_from_intent(t['intent']) for t in final_4_before_esc]
    pre_escalation_final4_score = sum({ "positive": 1, "neutral": 0, "negative": -1 }.get(s, 0) for s in pre_esc_final_sentiments)

    transfer_agent_count = len(escalation_turns)
    transfer_agent_first_turn_index = escalation_turns[0] if escalation_turns else None
    transfer_agent_last_turn_index = escalation_turns[-1] if escalation_turns else None
    last_turn_speaker = utterances[-1]['caller_name']

    transfer_assumed_success = (
        transfer_agent_count > 0 and last_turn_speaker == 'nlu'
    )
    escalation_final_turn_proximity = (
        total_turns - 1 - transfer_agent_last_turn_index <= 2 if transfer_agent_last_turn_index is not None else False
    )
    early_escalation_but_luhf = (
        luhf_tag == 'luhf' and transfer_agent_first_turn_index is not None and transfer_agent_first_turn_index <= 3
    )

    row = {
        "conversation_id": convo_id,
        "luhf_tag": luhf_tag,
        "escalation": "escalation" if transfer_agent_count > 0 else "not_escalation",
        "total_turns": total_turns,
        "user_turns_before_escalation": user_turns_before_escalation,
        "total_user_turns": total_user_turns,
        "nlu_positive_count": nlu_sentiments.count("positive"),
        "nlu_neutral_count": nlu_sentiments.count("neutral"),
        "nlu_negative_count": nlu_sentiments.count("negative"),
        "nlg_positive_count": nlg_pos,
        "nlg_neutral_count": nlg_neu,
        "nlg_negative_count": nlg_neg,
        "nlg_neg_intents": ",".join(nlg_neg_intents),
        "last_bot_intent": last_bot_intent or "none",
        "nlu_trajectory_index": nlu_trajectory_index,
        "nlg_trajectory_index": nlg_trajectory_index,
        "composite_trajectory_index": composite_trajectory_index,
        "nlu_density_index": nlu_density_index,
        "composite_density_index": composite_density_index,
        "adjusted_composite_index": adjusted_composite_index,
        "last_turn_speaker": last_turn_speaker,
        "transfer_assumed_success": transfer_assumed_success,
        "escalation_final_turn_proximity": escalation_final_turn_proximity,
        "early_escalation_but_luhf": early_escalation_but_luhf,
        "pre_escalation_final4_score": pre_escalation_final4_score,
        "final4_turns_composite_score": final4_composite_score,
        "final4_score_with_length_penalty": final4_score_with_length_penalty
    }

    output_rows.append(row)
    if transfer_agent_count > 0:
        escalation_rows.append(row)
        escalation_dialogs.append({"conversation_id": convo_id, "dialog": dialog})

# === Write full CSV ===
with open('BETOLD_clean_escalations_detailed.csv', 'w', newline='') as csvfile:
    fieldnames = output_rows[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

# === Write escalation-only CSV ===
with open('BETOLD_escalation_only.csv', 'w', newline='') as csvfile:
    fieldnames = escalation_rows[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(escalation_rows)

# === Save escalation dialogs JSON ===
with open('BETOLD_escalation_dialogs.json', 'w') as f:
    json.dump(escalation_dialogs, f, indent=2)
