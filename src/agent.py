import re
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "results/intent_classifier.pkl"
DATA_PATH = "data/amazon_clean_pairs.csv"


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)

df = df[["customer_text", "brand_response"]].dropna()

df["customer_text"] = df["customer_text"].astype(str)
df["brand_response"] = df["brand_response"].astype(str)


# ============================================================
# FIND TF-IDF VECTORIZER FROM TRAINED PIPELINE
# ============================================================

vectorizer = next(
    step
    for step in model.named_steps.values()
    if hasattr(step, "transform") and hasattr(step, "vocabulary_")
)


# Create vectors for historical customer messages
retrieval_vectors = vectorizer.transform(
    df["customer_text"].tolist()
)


# ============================================================
# CLEAN HISTORICAL RESPONSE
# ============================================================

def clean_response(response):

    response = str(response)

    # Remove Twitter usernames
    response = re.sub(r"@\w+", "", response)

    # Remove URLs
    response = re.sub(r"https?://\S+", "", response)

    # Remove Twitter agent markers
    response = re.sub(r"\^[A-Za-z]+", "", response)

    # Remove extra spaces
    response = re.sub(r"\s+", " ", response)

    return response.strip()


# ============================================================
# ESCALATION DECISION
# ============================================================

def check_escalation(customer_text, confidence, similarity):

    text = customer_text.lower()

    escalation_keywords = [
        "human",
        "real person",
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

    # Explicit request for human support
    for keyword in escalation_keywords:

        if keyword in text:

            return True, "Customer explicitly requested human assistance."

    # Low classifier confidence
    if confidence < 0.55:

        return True, (
            "The system was not sufficiently confident "
            "about the customer's intent."
        )

    # No useful historical match
    if similarity < 0.15:

        return True, (
            "No sufficiently similar historical Amazon "
            "support case was found."
        )

    # Otherwise automatically handle
    return False, (
        "The intent was classified with high confidence "
        "and a similar historical support case was found."
    )


# ============================================================
# CUSTOMER-FACING RESPONSE
# ============================================================

def generate_customer_response(intent, customer_text):

    responses = {

        "ORDER_STATUS":
            (
                "I'm sorry you're having trouble locating your order. "
                "Please open your Amazon Orders page and select the "
                "order to view its latest tracking information and "
                "estimated delivery date."
            ),

        "DELIVERY_DELAY":
            (
                "I'm sorry your delivery is delayed. "
                "Please check the tracking information in your Amazon "
                "Orders page for the latest update and revised delivery "
                "estimate. If the package remains delayed beyond the "
                "updated delivery date, please contact Amazon support "
                "for further assistance."
            ),

        "DELIVERY_PROBLEM":
            (
                "I'm sorry you haven't received your package. "
                "Please check the tracking details and confirm the "
                "delivery location. If the order is marked as delivered "
                "but you cannot find it, please check nearby safe "
                "locations and with others at your address. If it is "
                "still missing, Amazon support can investigate the "
                "delivery."
            ),

        "REFUND_RETURN":
            (
                "I can help with your return or refund issue. "
                "Please open your Amazon Orders page and select the "
                "relevant order to check the available return or refund "
                "options. If a refund has already been issued, please "
                "allow the required processing time. If it is still "
                "missing, please contact Amazon support."
            ),

        "CANCELLATION":
            (
                "To cancel an order, open your Amazon Orders page and "
                "select the relevant order. If cancellation is available, "
                "choose the cancellation option and follow the instructions "
                "shown. If the order has already entered the shipping "
                "process, cancellation may no longer be available."
            ),

        "PAYMENT_PROBLEM":
            (
                "I'm sorry you're having trouble with your payment. "
                "Please check that your payment method is valid and that "
                "your billing information is correct. You can also try "
                "another available payment method. If the problem "
                "continues, please contact Amazon support."
            ),

        "ACCOUNT_PROBLEM":
            (
                "I'm sorry you're having trouble accessing your Amazon "
                "account. Please use the Amazon sign-in and account "
                "recovery options if you cannot sign in. Follow the "
                "verification steps provided. If you still cannot access "
                "your account, please contact Amazon support."
            ),

        "PRIME_SERVICE":
            (
                "I'm sorry you're having trouble with Amazon Prime. "
                "Please check your Prime membership status and payment "
                "details in your Amazon account. If your membership or "
                "Prime benefits are not working as expected, please "
                "contact Amazon support."
            ),

        "PRODUCT_PROBLEM":
            (
                "I'm sorry the product did not arrive as expected. "
                "Please open your Amazon Orders page and select the "
                "affected order. Check the available return or "
                "replacement options and follow the instructions "
                "provided."
            ),

        "DIGITAL_SERVICE":
            (
                "I'm sorry you're having trouble with the digital "
                "service. Please check your internet connection, "
                "restart the Amazon app or Prime Video, and make sure "
                "the app is updated. Then try playing the content again. "
                "If the problem continues, please contact Amazon support."
            )
    }

    return responses.get(
        intent,
        (
            "I'm sorry you're experiencing this issue. "
            "Please check your Amazon account for the relevant "
            "order or service details. If the problem continues, "
            "please contact Amazon support for further assistance."
        )
    )


# ============================================================
# MAIN AGENT
# ============================================================

def run_agent(customer_text):

    # --------------------------------------------------------
    # 1. CLASSIFY CUSTOMER MESSAGE
    # --------------------------------------------------------

    probabilities = model.predict_proba([customer_text])[0]

    classes = model.classes_

    best_class_index = probabilities.argmax()

    intent = classes[best_class_index]

    confidence = probabilities[best_class_index]


    # --------------------------------------------------------
    # 2. RETRIEVE SIMILAR HISTORICAL AMAZON CASE
    # --------------------------------------------------------

    query_vector = vectorizer.transform([customer_text])

    similarities = cosine_similarity(
        query_vector,
        retrieval_vectors
    )[0]

    best_match_index = similarities.argmax()

    similarity = similarities[best_match_index]

    # Historical response is retrieved internally.
    # It is NOT directly shown to the customer.
    historical_response = df.iloc[best_match_index]["brand_response"]

    historical_response = clean_response(historical_response)


    # --------------------------------------------------------
    # 3. DECIDE AUTO-HANDLE OR ESCALATE
    # --------------------------------------------------------

    escalate, reason = check_escalation(
        customer_text,
        confidence,
        similarity
    )


    # --------------------------------------------------------
    # 4. DRAFT RESPONSE
    # --------------------------------------------------------

    if escalate:

        response = (
            "I'll escalate this issue to a human support agent "
            "so they can assist you further."
        )

    else:

        response = generate_customer_response(
            intent,
            customer_text
        )


    # --------------------------------------------------------
    # 5. DISPLAY RESULT
    # --------------------------------------------------------

    print()
    #print("=" * 50)

    print("INTENT:")
    print(intent)

    print()

    print("AI RESPONSE:")
    print(response)

    print()

    print("STATUS:")

    if escalate:
        print("⚠ Escalated to human support")
    else:
        print("✓ Automatically handled")

    print()

    print("REASON:")
    print(reason)

    print("=" * 50)


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 55)
    print("           AMAZON AI SUPPORT AGENT")
    print("=" * 55)

    print("\nType 'exit' to stop.\n")

    while True:

        customer_text = input("\nCustomer: ").strip()

        if customer_text.lower() == "exit":

            print("Exiting..")
            break

        if not customer_text:

            continue

        run_agent(customer_text)