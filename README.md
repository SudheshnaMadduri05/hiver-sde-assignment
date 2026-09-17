# Amazon AI Support Agent

An AI-powered customer support agent for AmazonHelp that classifies customer queries, retrieves similar historical support conversations, drafts a response, and decides whether the issue should be automatically handled or escalated to a human.

---

## Problem Statement

Build an AI support agent that understands Amazon customer queries, identifies their intent, provides a response grounded in historical Amazon support interactions, and decides whether the query should be auto-handled or escalated.

---

## Project Structure

```text
hiver-sde-assignment/
│
├── data/
│   ├── amazon_clean_pairs.csv
│   ├── amazon_labeled.csv
│   ├── amazon_pairs.csv
│   └── amazon_support.csv
│
├── src/
│   ├── agent.py
│   ├── analyze_other.py
│   ├── analyze_pairs.py
│   ├── build_pairs.py
│   ├── discover_intents.py
│   ├── extract_brand.py
│   ├── inspect_data.py
│   ├── label_intents.py
│   └── train_classifier.py
│
├── baselines/
│   └── baseline.py
│
├── evaluation/
│   ├── create_golden_set.py
│   ├── evaluate.py
│   ├── failure_analysis.py
│   ├── golden_set.csv
│   ├── human_llm_agreement.py
│   └── llm_judge.py
│
├── results/
│   ├── baseline_results.txt
│   ├── golden_evaluation.txt
│   ├── golden_errors.csv
│   ├── intent_classifier.pkl
│   └── top_5_failures.txt
│
├── .gitignore
└── README.md

## Technologies Used
Python
Pandas
NumPy
Scikit-learn
TF-IDF
Logistic Regression
Cosine Similarity
Joblib
Ollama / Gemma
Customer Support on Twitter Dataset
Problem Framing
What does "good" mean for Amazon?

##A good support agent should:

Correctly identify the customer's main issue.
Provide a clear and useful response.
Use historical Amazon support interactions as grounding.
Automatically handle straightforward queries.
Escalate cases that require human assistance.
What I chose not to build

##The project does not attempt to:

Connect to Amazon's internal order systems.
Perform real refunds or cancellations.
Access customer accounts.
Track real orders.
Deploy a production-scale customer support system.
Dataset

##The project uses the Customer Support on Twitter dataset.

##Selected Brand

##AmazonHelp

The AmazonHelp subset was extracted and customer messages were paired with corresponding Amazon support responses.

After cleaning and deduplication:

Total Amazon conversation pairs: 168,814
Useful unique conversations: 152,752
Intent Classification

##10 support intents were defined:

Intent	Description
ORDER_STATUS	Order tracking and order status
DELIVERY_DELAY	Delayed or late deliveries
DELIVERY_PROBLEM	Delivered-but-not-received and delivery issues
REFUND_RETURN	Refund and return issues
CANCELLATION	Order cancellation
PAYMENT_PROBLEM	Payment and charging issues
ACCOUNT_PROBLEM	Login and account issues
PRIME_SERVICE	Amazon Prime membership and benefits
PRODUCT_PROBLEM	Damaged, defective, or incorrect products
DIGITAL_SERVICE	Prime Video and digital service issues
##System Approach
Customer Query
      ↓
Intent Classification
      ↓
Historical Conversation Retrieval
      ↓
Response Drafting
      ↓
Auto-Handle / Escalate Decision
      ↓
Final Response + Reason
1. Intent Classification

TF-IDF features and Logistic Regression are used to classify the customer query into one of the 10 intents.

2. Historical Retrieval

Cosine similarity is used to find similar historical AmazonHelp customer-support conversations.

3. Response Drafting

A customer-facing response is generated based on the identified support intent and historical support context.

4. Escalation

The system escalates when:

The customer explicitly requests a human.
Intent confidence is low.
No sufficiently similar historical case is found.

A reason is provided for the decision.

##Results vs Baselines

Two classification baselines were evaluated.

Baseline 1 — Majority Class

The majority-class baseline always predicts the most frequent intent.

Accuracy: 10.13%

Baseline 2 — TF-IDF + Logistic Regression

The simple machine-learning classifier uses TF-IDF features with Logistic Regression.

Accuracy: 94.73%

Model	Accuracy
Majority Class	10.13%
TF-IDF + Logistic Regression	94.73%

The classifier achieved approximately 0.95 macro F1-score across the 10 intents.

##Failure Analysis

The main failure modes identified were:

1. Similar Delivery-Related Intents

Example:

"My package is late, where is it?"

Possible issue: ORDER_STATUS vs DELIVERY_DELAY

Hypothesis: Both intents contain similar words such as order, package, delivery, and arrive.

2. Delivery Problem vs Order Status

Example:

"My order says delivered but I didn't receive it."

Possible issue: DELIVERY_PROBLEM vs ORDER_STATUS

Hypothesis: Both categories contain strong order and delivery vocabulary.

3. Payment and Order Ambiguity

Example:

"I was charged but my order isn't showing."

Possible issue: PAYMENT_PROBLEM vs ORDER_STATUS

Hypothesis: The query contains information belonging to two different support categories.

4. Prime Service and Digital Service Overlap

Example:

"Prime Video isn't working even though I have Prime."

Possible issue: DIGITAL_SERVICE vs PRIME_SERVICE

Hypothesis: Both categories frequently contain the word "Prime".

5. Short or Ambiguous Queries

Example:

"Help with my order."

Possible issue: Incorrect classification between order-related intents.

Hypothesis: Very short messages do not contain enough information for reliable classification.

Note: Replace these examples with the actual misclassified examples from results/golden_errors.csv if available.

What Is Misleading About My Headline Number?

The headline number of 94.73% accuracy only measures intent classification performance on the held-out evaluation split.

It does not mean that 94.73% of complete customer interactions will be successfully resolved.

The end-to-end system also depends on:

Quality of historical retrieval.
Quality of the generated response.
Correct escalation decisions.
Ambiguous or unseen customer queries.

Therefore, classification accuracy should be interpreted as a classification metric, not an overall agent success rate.

##Golden Evaluation Set

A separate golden evaluation set was created for evaluating the intent classifier.

The target evaluation set contains 200 examples distributed across the 10 defined intents.

Each example is intended to be reviewed according to the customer's primary support request.

The evaluation pipeline measures:

Accuracy
Precision
Recall
F1-score
Classification errors
LLM-as-a-Judge

An LLM-based evaluation pipeline is included to assess response quality.

The response is evaluated on:

Relevance
Helpfulness
Grounding
Professionalism
Overall quality

Each criterion is scored from 1 to 5.

The project uses Ollama with Gemma for local evaluation without depending on an external API.

Results are saved to:

results/llm_judge_results.csv
What I Would Do With One More Week
Improve manual intent labelling.
Use semantic embeddings for better retrieval.
Replace template responses with LLM-based response generation.
Improve escalation detection.
Expand the golden evaluation set.
Perform more extensive LLM-based response evaluation.
Test the system on more unseen customer queries.

##Decision Log
Selected AmazonHelp because it provides a large number of customer-support interactions for one brand.
Used a dataset subsample to make experimentation and evaluation practical.
Created customer-response conversation pairs by linking customer messages with corresponding brand responses.
Defined 10 intents to keep the classification task focused and manageable.
Excluded OTHER from classifier training because it contained highly diverse and noisy examples.
Used TF-IDF because it provides a simple and interpretable representation for text classification.
Used Logistic Regression as the main classifier because it is effective for sparse text features.
Applied class balancing because the intent distribution was uneven.
Used historical conversation retrieval to provide support context instead of generating responses without reference to the dataset.
Added escalation rules for human requests, low confidence, and weak historical similarity.

##Limitations
Twitter customer-support data contains noisy and incomplete responses.
Weakly labelled data can introduce classification errors.
The system does not connect to real Amazon services.
Current response generation uses predefined customer-facing templates.
LLM evaluation depends on the quality of the local judging model.
