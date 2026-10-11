from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

messages = [
    "You won a prize",
    "Hello, How are you my friend",
    "Real Madrid has won the spanish league title last year",  
    "Free to meet tomorrow?",
    "Happy birthday! Let's have dinner this weekend",
    "Congratulations! You have been selected to receive a FREE iPhone. Click here now",
    "URGENT: your bank account is blocked. Confirm your password at this link",
    "Earn $5000 per week from home, no experience needed",
    "Can you send me the homework before tonight?",
    "Claim your prize now, limited offer, reply YES",
]

vectorizer = CountVectorizer()

# Learn the vocabulary and convert your messages
X = vectorizer.fit_transform(messages)

# Print the vocabulary
print(vectorizer.get_feature_names_out())

# Print the table of numbers
print(X.toarray())

labels = ["spam", "ham", "ham", "ham", "ham", "spam", "spam", "spam", "ham", "spam"]

# Create the model
model = MultinomialNB()

# Train it: it learns from the numbers (X) and the correct answers (labels)
model.fit(X, labels)

# A new message the model has never seen (a list, even with one message)
new_messages = [
    "Claim your free prize",
    "Hello my friend, how are you",
    "Free to meet tomorrow?",
    "Are you free tomorrow?",
    "You are the lucky winner, click the link",
    "Can we meet at the library tomorrow?",
    "Win a free gift now",
    "The match was great, you should have seen it",
    ]
new_X = vectorizer.transform(new_messages)

# Predict
prediction = model.predict(new_X)
print(prediction)

# See the model’s confidence
print(model.classes_)
probabilities = model.predict_proba(new_X)
for m, p in zip(new_messages, probabilities):
    print(m, "->", p)

print(len(messages), len(labels))