# 🛡️ Spam Message Detector

A small machine learning project that tells whether an SMS-style message looks like spam. It uses a Naive Bayes classifier trained on real SMS messages and comes with a Streamlit web interface.

I built it step by step to learn the basics of machine learning: from simple `if` rules to a trained and evaluated model.

## Features

- Classifies a message as **spam** or **ham** (normal message)
- Shows the model's confidence (normal % vs spam %)
- Replaces long digit strings (phone numbers, short codes) with one shared token, `longnumber`
- Modern web interface with example messages

## How it works

1. **Data:** the SMS Spam Collection (`spam.csv`), 5,572 messages: 4,825 ham and 747 spam.
2. **Cleaning:** every number with 5 or more digits becomes `longnumber`.
3. **Features:** `CountVectorizer` turns each message into word counts (bag of words).
4. **Model:** `MultinomialNB(alpha=0.1)`.
5. **Evaluation:** 80/20 train/test split with `random_state=42` (see `load_data.py`).

## Results

Measured on the 20% of messages the model never saw during training (1,115 messages):

| Metric | Value |
|---|---|
| Accuracy | 98.7% |
| Spam precision | 0.97 |
| Spam recall | 0.93 |
| Good messages blocked | 4 |
| Spam missed | 10 |

Note: `alpha=0.1` and the `longnumber` rule were chosen while looking at errors on this test set, so the real accuracy on new data may be slightly lower.

## Run it

```bash
python -m pip install -r requirements.txt

# Evaluate the model in the terminal
python load_data.py

# Launch the web app (run from this folder, so spam.csv is found)
python -m streamlit run app.py
```

## Project structure

```
spam-message-detector/
├── app.py            # Streamlit web app
├── load_data.py      # Training, evaluation and error analysis
├── spam.csv          # Dataset
├── requirements.txt
├── README.md
└── learning/         # Early steps: rules-based version and first tiny dataset
    ├── main.py
    └── dataset.py
```

## Limitations

- The model learned the statistics of English SMS spam, not the meaning of sentences. It can be **confident and still wrong**.
- Example: `The winner is here` is flagged as spam (about 90% confidence), because the word "winner" appears mostly in prize-draw spam in the dataset.
- Bag of words ignores word order, and unknown words are ignored.
- Spammers misspell words on purpose, and the model misses those more often.
- It works poorly on text that does not look like SMS (emails, file paths, other languages).
- The dataset is old and small compared with modern spam.

## Ideas for improvement

- Add more normal messages containing words like "winner", "prize" or "free"
- Try `TfidfVectorizer` or `ngram_range=(1, 2)`
- Use cross-validation to choose settings like `alpha` fairly
- Add a validation split so the test set stays untouched

## Dataset

SMS Spam Collection (UCI Machine Learning Repository), also available on Kaggle as `spam.csv`.