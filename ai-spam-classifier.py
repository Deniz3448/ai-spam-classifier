import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Veri oku
df = pd.read_csv("spam.csv", encoding="latin-1")

# Gereksiz kolonları sil ve isimleri düzenle
df = df[["v1", "v2"]]
df.columns = ["label", "text"]

# Label encode (spam=1, ham=0)
df["label"] = df["label"].map({"spam": 1, "ham": 0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Text vectorize
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model oluştur ve eğit
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Tahmin yap
y_pred = model.predict(X_test_vec)

# Accuracy yazdır
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print("\n--- Spam Classifier Ready ---")
print("Type a message to test (type 'exit' to quit)\n")

# Kullanıcıdan input al
while True:
    msg = input("Enter message: ")

    if msg.lower() == "exit":
        print("Exiting...")
        break

    # Mesajı modele ver
    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)

    # Sonucu yazdır
    if prediction[0] == 1:
        print("Result: 🚨 Spam\n")
    else:
        print("Result: ✅ Not Spam\n")
