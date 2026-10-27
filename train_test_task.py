import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("houses.csv")

print(df.head())
print(df.shape)
df.info()
print(df.isna().sum())

X = df[["Area", "Bedrooms", "Bathrooms", "Age"]]
y = df["Price"]

print("Features:")
print(X.head())

print("Target:")
print(y.head())

print("Number of features:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Total records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Number of features:", X_train.shape[1])
print("Target column:", y.name)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


X_train_70, X_test_30, y_train_70, y_test_30 = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

comparison = pd.DataFrame({
    "Split": ["80/20", "70/30"],
    "Training Records": [len(X_train), len(X_train_70)],
    "Testing Records": [len(X_test), len(X_test_30)],
    "Features": [X_train.shape[1], X_train_70.shape[1]]
})

print(comparison)