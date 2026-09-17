# Amazon AI Customer Support Agent

## 1. Project Overview

This project implements an AI-powered customer support agent using historical Amazon customer-support conversations from the Customer Support on Twitter dataset.

The system:

1. Classifies the customer's issue into a predefined support intent.
2. Retrieves similar historical Amazon support conversations.
3. Uses historical resolutions to draft a grounded response.
4. Decides whether the request can be automatically handled or should be escalated.
5. Evaluates intent classification using a hand-labelled golden evaluation set.
6. Compares the model against two baselines.
7. Performs failure analysis on incorrect predictions.

---

## 2. Dataset

Dataset:

**Customer Support on Twitter**

Source:

https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

Selected brand:

**AmazonHelp**

Original dataset contains millions of customer-support tweets.

For this project, AmazonHelp conversations were extracted and cleaned.

### Dataset processing

| Stage | Records |
|---|---:|
| AmazonHelp tweets | 169,840 |
| Conversation pairs | 168,814 |
| Clean unique conversations | 152,752 |
| Final labelled intent data | 152,752 |

---

## 3. Intent Taxonomy

Ten customer-support intents were defined:

| Intent | Description |
|---|---|
| ORDER_STATUS | Customer asks where an order is or requests tracking |
| DELIVERY_DELAY | Delivery is late or missed the expected date |
| DELIVERY_PROBLEM | Package is marked delivered but was not received or has a carrier issue |
| REFUND_RETURN | Refund or return-related requests |
| CANCELLATION | Order cancellation requests |
| PAYMENT_PROBLEM | Payment, Amazon Pay or payment failure issues |
| ACCOUNT_PROBLEM | Login, password or account-access problems |
| PRIME_SERVICE | Prime membership and Prime benefits |
| PRODUCT_PROBLEM | Damaged, defective, incorrect or poor-quality products |
| DIGITAL_SERVICE | Prime Video, apps and digital-service technical problems |

---

## 4. System Architecture

```text
Customer Message
       |
       v
Intent Classifier
(TF-IDF + Logistic Regression)
       |
       v
Predicted Intent
       |
       v
Historical Conversation Retrieval
       |
       v
Similar Amazon Support Cases
       |
       v
Grounded Response Draft
       |
       v
Escalation Decision
       |
       +----------------+
       |                |
       v                v
 AUTO_HANDLE         ESCALATE