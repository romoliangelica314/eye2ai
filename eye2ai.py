import json
import time
import joblib
import pandas as pd
from pathlib import Path

stateFile = Path("bodycam_metadata.json")
modelPath = Path("eye2ai_final_pipeline.pkl")

modelFeatures = [
    "suspect_age",
    "suspect_gender",
    "suspected_offense_type",
    "intoxication_level",
    "mental_confusion_level",
    "suspect_aggression_level",
    "suspect_compliance",
    "officer_tone",
    "environment_risk_level",
    "bystanders_present",
    "number_of_officers",
    "time_of_day"]

cueWords = [
    "stay calm",
    "lower tone",
    "speak slower",
    "one command",
    "explain",
    "stop talking",
    "risk rising"]

if not modelPath.exists():
    raise FileNotFoundError(f"Model file not found: {modelPath.resolve()}")

finalPipeline = joblib.load(modelPath)


def loadState():
    if not stateFile.exists():
        return None
    try:
        with open(stateFile, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Could not read {stateFile.name}: {e}")
        return None


def scoreCues(state):
    scores = {word: 0 for word in cueWords}

    aggression = state.get("suspect_aggression_level", 0) or 0
    compliance = state.get("suspect_compliance", 0) or 0
    confusion = state.get("mental_confusion_level", 0) or 0
    intoxication = state.get("intoxication_level", 0) or 0
    risk = state.get("environment_risk_level", 0) or 0
    tone = state.get("officer_tone", 0) or 0
    bystanders = state.get("bystanders_present", 0) or 0

    if aggression >= 3:
        scores["stay calm"] += 3
        scores["stop talking"] += 3
        scores["risk rising"] += 4
    if compliance <= 1:
        scores["explain"] += 2
        scores["one command"] += 4
        scores["stop talking"] += 1
    if confusion >= 2:
        scores["speak slower"] += 4
        scores["one command"] += 5
        scores["explain"] += 2
    if intoxication >= 2:
        scores["speak slower"] += 4
        scores["one command"] += 2
    if tone >= 2:
        scores["lower tone"] += 5
        scores["stay calm"] += 3
    if risk >= 3:
        scores["risk rising"] += 5
    if bystanders == 1:
        scores["stay calm"] += 1
        scores["explain"] += 1

    return scores


def suggestCue(state):
    scores = scoreCues(state)
    return max(scores, key=scores.get)


def scoreInteraction(state):
    row = {feature: state.get(feature) for feature in modelFeatures}
    X_live = pd.DataFrame([row])
    score = finalPipeline.predict(X_live)[0]
    return float(score)


def scoreBand(score):
    if score < 40:
        return "critical"
    elif score < 70:
        return "unstable"
    return "stable"


def processState(state):
    cue = suggestCue(state)
    score = scoreInteraction(state)
    risk_band = scoreBand(score)

    return {
        "timestamp": state.get("timestamp"),
        "cue": cue,
        "interaction_score": round(score, 2),
        "risk_band": risk_band
    }


def displayState(result):
    print(f"Suggested cue:      {result.get('cue')}")
    print(f"Predicted score:    {result.get('interaction_score'):.2f}")
    print(f"Risk band:          {result.get('risk_band', '').upper()}")


def update(poll_interval=2):
    print(f"Watching {stateFile.name} for updates...")
    last_seen = None

    while True:
        state = loadState()

        if state is not None:
            state_str = json.dumps(state, sort_keys=True)

            if state_str != last_seen:
                last_seen = state_str
                result = processState(state)
                displayState(result)

        time.sleep(poll_interval)


if __name__ == "__main__":
    update()
