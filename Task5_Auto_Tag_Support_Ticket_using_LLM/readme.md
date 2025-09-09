#  🤖Automated Support Ticket Tagging with LLMs

This project showcases how **Large Language Models (LLMs)** can be applied for the **automatic classification of support tickets** into meaningful categories.  
The implementation demonstrates and compares the performance of **zero-shot** and **few-shot** prompting techniques for predicting the most relevant labels.

The experiments are powered by the [`google/flan-t5-large`](https://huggingface.co/google/flan-t5-large) model from Hugging Face.

---

## 🔍 Overview

Modern customer support systems receive thousands of tickets daily.  
Manually tagging these tickets is time-consuming and prone to inconsistency.  
By leveraging LLMs, we can **automatically assign tags to tickets**, improving both efficiency and accuracy.

The system is evaluated on a **simulated dataset of support issues** (e.g., login problems, payment failures, crashes).  
Predictions are generated in two settings:
1. **Zero-Shot Prompting**  
   - The model receives the ticket text and the list of possible categories.  
   - It predicts the tag purely based on its pre-trained knowledge.  

2. **Few-Shot Prompting**  
   - In addition to the ticket text and tag list, the model is shown a few example ticket–tag pairs.  
   - This improves contextual understanding and often leads to better accuracy.  

Finally, results from both methods are compared using **Top-3 Accuracy**.

---

## ✨ Features

- **Zero-Shot & Few-Shot Tagging**: Straightforward implementation of both prompting strategies.  
- **Top-3 Prediction Ranking**: Ensures multiple possible tags are considered for robustness.  
- **Performance Evaluation**: Computes Top-3 accuracy for both methods.  
- **LLM Integration**: Uses Hugging Face’s `transformers` library for seamless model loading and inference.  
- **Customizable Dataset**: Simulated dataset included, but can easily be swapped with real-world support data.  
- **Fine-Tuning Placeholder**: Provides guidance for extending the script with fine-tuned models.  

---

## ⚙️ How It Works

1. **Dataset Setup**  
   - Tickets are provided as text along with their correct tag.  
   - Example categories: `login_issue`, `password_reset`, `app_crash`, `payment_issue`, etc.  

2. **Prompt Construction**  
   - **Zero-Shot Prompt**: Ticket text + full list of tags.  
   - **Few-Shot Prompt**: Ticket text + list of tags + 2–3 example ticket–tag pairs.  

3. **Inference with Flan-T5**  
   - Uses beam search to generate multiple predictions.  
   - Filters and ranks tags to output the **Top-3 candidates**.  

4. **Evaluation**  
   - A prediction is considered correct if the true tag is within the Top-3 predictions.  
   - Compares accuracy between zero-shot and few-shot settings.  
---
## 📊 Example Output
Ticket: "I can't log in to my account. It says invalid credentials."
True Tag: login_issue
Zero-shot Top-3: ['password_reset', 'login_issue', 'account_update']
Few-shot Top-3: ['login_issue', 'password_reset', '2fa_issue']

Performance summary:
- Zero-Shot Accuracy: 72%  
- Few-Shot Accuracy: 85%  

*(Accuracy values depend on dataset and random examples used in few-shot mode.)*

---

## 🚀 Getting Started

### ✅ Prerequisites
- Python 3.8+  
- Install required libraries:
pip install torch transformers scikit-learn

# 🔮 Future Enhancements

Fine-Tuning: Train the model on a larger, domain-specific dataset for better accuracy.

Visualization: Add plots for performance comparison (accuracy, confusion matrix).

Scalability: Deploy as an API for real-time ticket tagging.

Integration: Connect with ticketing platforms like Zendesk or Freshdesk.

# 📚 References

Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers
Model Card: google/flan-t5-large
Scikit-learn Metrics: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
```bash


