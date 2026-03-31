import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Veri oku
df = pd.read_csv("spam.csv", encoding="latin-1")

# Sadece gerekli kolonlar
df = df[["v1", "v2"]]
df.columns = ["label", "text"]

# Label encode (spam=1, ham=0)
df["label"] = df["label"].map({"spam":1, "ham":0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2)

# Text vectorize
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Tahmin
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Test et
sample = ["Congratulations! You won a free ticket"]
sample_vec = vectorizer.transform(sample)
print("Prediction:", model.predict(sample_vec))
