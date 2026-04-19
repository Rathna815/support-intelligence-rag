Customer Support Intelligence System

> RAG-powered pipeline that classifies support tickets, detects sentiment, 
> and generates context-aware replies using past resolved tickets as reference.

Problem Statement

Billing support teams handle thousands of repetitive tickets daily. 
Manual triage and reply drafting wastes agent hours on queries that follow 
predictable patterns — and urgent, frustrated customers get delayed responses.

Dataset

- Source: Customer Support Ticket Dataset — [Kaggle](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)
- Size: 8,469 support tickets across 5 categories
- Features: Ticket text, category, priority, satisfaction rating

Approach

Stage 1 — ML Ticket Classification
- Preprocessed ticket text with Pandas
- Converted to numerical features via TF-IDF (5,000 features, bigrams)
- Trained Logistic Regression classifier across 5 categories
- Added TextBlob sentiment scoring to auto-flag frustrated customers

Stage 2 — RAG Pipeline
- Embedded all tickets with `sentence-transformers` (all-MiniLM-L6-v2)
- Indexed 384-dimensional embeddings in FAISS for fast similarity search
- Retrieved top 3 semantically similar resolved tickets per new query

Stage 3 — LLM Reply Generation
- Prompt: ticket text + predicted category + sentiment + 3 retrieved examples
- Called Groq API (LLaMA 3.1) for professional reply generation
- Stored ticket, prediction, and reply in SQLite
