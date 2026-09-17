import os
import json
import requests
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "results/intent_classifier.pkl"
PAIRS_PATH = "data/amazon_clean_pairs.csv"
GOLDEN_PATH = "evaluation/golden_set.csv"
OUTPUT_PATH = "results/llm_judge_results.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading classifier...")

model = joblib.load(MODEL_PATH)


# ============================================================
# FIND TF-IDF VECTORIZER
# ============================================================

vectorizer = None

for step in model.named_steps.values():
    if hasattr(step, "vocabulary_"):
        vectorizer = step
        break

if vectorizer is None:
    raise RuntimeError("TF-IDF vectorizer not found.")


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

print("Loading historical conversations...")

pairs = pd.read_csv(
    PAIRS_PATH,
    usecols=["customer_text", "brand_response"]
).dropna()

pairs = pairs.drop_duplicates("customer_text")

pairs["customer_text"] = pairs["customer_text"].astype(str)
pairs["brand_response"] = pairs["brand_response"].astype(str)

print(f"Historical conversations: {len(pairs)}")


# ============================================================
# PRECOMPUTE TF-IDF ONCE
# ============================================================

print("Preparing historical search index...")

historical_vectors = vectorizer.transform(
    pairs["customer_text"]
)

print("Search index ready.")


# ============================================================
# RESPONSE TEMPLATES
# ============================================================

RESPONSE_TEMPLATES = {

    "ORDER_STATUS":
        "I'm sorry you're having trouble locating your order. "
        "Please check the tracking information in your Amazon Orders "
        "page for the latest delivery update.",

    "DELIVERY_DELAY":
        "I'm sorry your delivery is delayed. Please check the tracking "
        "information in your Amazon Orders page for the latest update "
        "and revised delivery estimate.",

    "DELIVERY_PROBLEM":
        "I'm sorry there is a problem with your delivery. Please check "
        "the tracking details for the latest information. If the package "
        "is marked as delivered but has not arrived, this issue may need "
        "further investigation.",

    "REFUND_RETURN":
        "I'm sorry you're having trouble with your return or refund. "
        "Please check the return status in your Amazon Orders page for "
        "the latest refund information.",

    "CANCELLATION":
        "I can help with your cancellation request. Please check your "
        "Amazon Orders page to see whether the order can still be cancelled.",

    "PAYMENT_PROBLEM":
        "I'm sorry you're having trouble with your payment. Please check "
        "your payment method and billing information and try the payment again.",

    "ACCOUNT_PROBLEM":
        "I'm sorry you're having trouble with your Amazon account. "
        "Please check your account login details and use the account "
        "recovery options if you cannot sign in.",

    "PRIME_SERVICE":
        "I'm sorry you're having trouble with your Prime service. "
        "Please check your Prime membership status and benefits in your "
        "Amazon account.",

    "PRODUCT_PROBLEM":
        "I'm sorry there is a problem with the product you received. "
        "Please check your Amazon Orders page for available return, "
        "replacement, or refund options.",

    "DIGITAL_SERVICE":
        "I'm sorry you're having trouble with the digital service. "
        "Please check your connection, restart the application, and "
        "try accessing the content again."
}


# ============================================================
# ESCALATION
# ============================================================

ESCALATION_KEYWORDS = [
    "human",
    "real person",
    "real human",
    "agent",
    "representative",
    "supervisor",
    "manager",
    "escalate",
    "complaint",
    "lawyer",
    "legal",
    "lawsuit",
    "chargeback"
]


def should_escalate(text, confidence, similarity):

    text_lower = text.lower()

    for keyword in ESCALATION_KEYWORDS:
        if keyword in text_lower:
            return True, "Customer explicitly requested human assistance."

    if confidence < 0.55:
        return True, "Intent classification confidence is low."

    if similarity < 0.15:
        return True, "No sufficiently similar historical support case was found."

    return False, (
        "High-confidence intent classification and a similar "
        "historical Amazon support case was found."
    )


# ============================================================
# RETRIEVE SIMILAR HISTORICAL CASE
# ============================================================

def retrieve_case(customer_text):

    query_vector = vectorizer.transform(
        [customer_text]
    )

    similarities = cosine_similarity(
        query_vector,
        historical_vectors
    )[0]

    best_index = similarities.argmax()

    best_similarity = float(
        similarities[best_index]
    )

    historical_response = pairs.iloc[
        best_index
    ]["brand_response"]

    return historical_response, best_similarity


# ============================================================
# RUN AGENT
# ============================================================

def run_agent(customer_text):

    probabilities = model.predict_proba(
        [customer_text]
    )[0]

    predicted_intent = model.classes_[
        probabilities.argmax()
    ]

    confidence = float(
        probabilities.max()
    )

    historical_response, similarity = retrieve_case(
        customer_text
    )

    escalate, reason = should_escalate(
        customer_text,
        confidence,
        similarity
    )

    if escalate:

        response = (
            "I'll escalate this issue to a human support agent "
            "so they can assist you further."
        )

        status = "ESCALATED"

    else:

        response = RESPONSE_TEMPLATES.get(
            predicted_intent,
            "I'll help you with your Amazon support request."
        )

        status = "AUTO_HANDLE"

    return {
        "intent": predicted_intent,
        "confidence": confidence,
        "similarity": similarity,
        "response": response,
        "status": status,
        "reason": reason,
        "historical_response": historical_response
    }


# ============================================================
# OLLAMA JUDGE
# ============================================================

def judge_response(
    customer_text,
    intent,
    response,
    historical_response,
    status
):

    prompt = f"""
You are evaluating an AI customer support agent for Amazon.

Customer message:
{customer_text}

Predicted intent:
{intent}

AI-generated response:
{response}

Historical Amazon support response:
{historical_response}

Handling decision:
{status}

Evaluate the AI-generated response.

Give a score from 1 to 5 for:

Relevance:
Does it address the customer's issue?

Helpfulness:
Does it provide useful assistance?

Grounding:
Is it consistent with the historical support context?

Professionalism:
Is it clear and professional?

Overall:
Overall quality.

Return ONLY JSON:

{{
  "relevance": 1,
  "helpfulness": 1,
  "grounding": 1,
  "professionalism": 1,
  "overall": 1,
  "reason": "short explanation"
}}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    result = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180
    )

    result.raise_for_status()

    output = result.json()["response"].strip()

    if output.startswith("```"):
        output = output.replace("```json", "")
        output = output.replace("```", "")
        output = output.strip()

    return json.loads(output)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==============================================")
    print(" AMAZON AI SUPPORT - LLM AS JUDGE")
    print("==============================================\n")

    # Check Ollama
    try:
        requests.get(
            "http://localhost:11434",
            timeout=5
        )
    except Exception:
        print("ERROR: Ollama is not running.")
        print("Run 'ollama serve' in another terminal.")
        return

    # Load golden set
    golden = pd.read_csv(
        GOLDEN_PATH
    )

    # Evaluate 30 examples
    golden = golden.head(30)

    print(
        f"Evaluating {len(golden)} golden examples...\n"
    )

    results = []

    for i, row in enumerate(
        golden.itertuples(index=False),
        start=1
    ):

        customer_text = str(row.customer_text)
        gold_intent = str(row.gold_intent)

        print(
            f"[{i}/{len(golden)}] Processing...",
            flush=True
        )

        try:

            # Run YOUR actual agent
            agent = run_agent(
                customer_text
            )

            # Judge YOUR actual response
            scores = judge_response(
                customer_text,
                agent["intent"],
                agent["response"],
                agent["historical_response"],
                agent["status"]
            )

            results.append({

                "customer_text": customer_text,

                "gold_intent": gold_intent,

                "predicted_intent":
                    agent["intent"],

                "confidence":
                    round(agent["confidence"], 4),

                "similarity":
                    round(agent["similarity"], 4),

                "ai_response":
                    agent["response"],

                "status":
                    agent["status"],

                "decision_reason":
                    agent["reason"],

                "relevance":
                    scores["relevance"],

                "helpfulness":
                    scores["helpfulness"],

                "grounding":
                    scores["grounding"],

                "professionalism":
                    scores["professionalism"],

                "overall":
                    scores["overall"],

                "judge_reason":
                    scores["reason"]
            })

        except Exception as e:

            print(
                f"ERROR: {e}"
            )


    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        "results",
        exist_ok=True
    )

    df = pd.DataFrame(
        results
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n==============================================")
    print(" LLM-AS-A-JUDGE RESULTS")
    print("==============================================")

    if len(df) == 0:
        print("No results generated.")
        return

    score_columns = [
        "relevance",
        "helpfulness",
        "grounding",
        "professionalism",
        "overall"
    ]

    for column in score_columns:

        score = pd.to_numeric(
            df[column],
            errors="coerce"
        ).mean()

        print(
            f"{column.capitalize():20s}: "
            f"{score:.2f}/5"
        )

    print("\nSaved:")
    print(
        "results/llm_judge_results.csv"
    )


if __name__ == "__main__":
    main()