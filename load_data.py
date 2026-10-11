import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the file (Kaggle version: comma-separated, latin-1 encoding)
df = pd.read_csv("spam.csv", encoding="latin-1")

# 2. Keep only the two useful columns
df = df[["v1", "v2"]]

# 3. Rename them
df.columns = ["label", "message"]

# 4. Look at the data
print(df.head())
print(df.shape)
print(df["label"].value_counts())

df["message"] = df["message"].str.replace(r"\d{5,}", " longnumber ", regex=True)

X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42
)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model score is = 98.4%
model = MultinomialNB(alpha=0.1)
model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)

print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Read errors in machine learning
#errors = X_test[(y_test != predictions)]
#for message in errors :
    #print(message)

# show the true and predicted label for each error
mask = (y_test !=predictions) 

for message, true_label, predicted_label in zip(X_test[mask], y_test[mask], predictions[mask]):
    print("true :", true_label, "| predicted :", predicted_label, "| ", message)

# After measured the model, Now I'll try own message 
# Use like a real detector

# show the confidence in the loop
def check_message(text) :
    text = pd.Series([text]).str.replace(r"\d{5,}", " longnumber ", regex=True)
    vec = vectorizer.transform(text)
    label = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0] # akes the first (and only) row, so you get [0.07 0.93]
    confidence = probabilities.max() # picks the bigger number. It is the model’s confidence 
    return label, confidence


# Return Spam
print(check_message("Congratulations! Call 09012345678 to claim your free prize"))
# Return ham
print(check_message("Are you coming to dinner tonight?"))

print(check_message("your clear spam here"))
print(check_message("your clear ham here"))
print(check_message("your tricky message here"))

vec = vectorizer.transform(pd.Series(["The winner is here"]))
print(model.classes_)
print(model.predict_proba(vec))

while True :
    text = input("Enter a message (or 'q' to quit): ")
    if text == "q" :
        break
    result, confidence = check_message(text)
    if (result == "spam") :
        print(f"Spam probable ({confidence * 100:.1f}%)")
    else : 
        print(f"Message normal ({confidence * 100:.1f}%)")




