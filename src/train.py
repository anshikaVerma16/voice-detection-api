import json
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


file = open("advanced_features.json", "r")
data = json.load(file)
file.close()

X = []
y = []
languages = []

for i in range(len(data)):
    X.append(data[i]["features"])
    y.append(data[i]["label"])
    languages.append(data[i]["language"])

X = np.array(X)
y = np.array(y)
languages = np.array(languages)


X_train, X_test, y_train, y_test, lang_train, lang_test = train_test_split(
    X, y, languages, test_size=0.2, random_state=42, stratify=y
)


rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_accuracy = rf_model.score(X_test, y_test)


encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train_encoded)
xgb_accuracy = xgb_model.score(X_test, y_test_encoded)


gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=7,
    random_state=42
)

gb_model.fit(X_train, y_train)
gb_accuracy = gb_model.score(X_test, y_test)


rf_predictions = rf_model.predict(X_test)
xgb_predictions = encoder.inverse_transform(xgb_model.predict(X_test))
gb_predictions = gb_model.predict(X_test)

final_predictions = []

for i in range(len(X_test)):
    votes = []
    votes.append(rf_predictions[i])
    votes.append(xgb_predictions[i])
    votes.append(gb_predictions[i])

    final_predictions.append(max(set(votes), key=votes.count))

final_predictions = np.array(final_predictions)

ensemble_accuracy = 0
correct = 0

for i in range(len(final_predictions)):
    if final_predictions[i] == y_test[i]:
        correct += 1

ensemble_accuracy = correct / len(final_predictions)


print("Per language accuracy:")
for lang in np.unique(lang_test):
    total = 0
    right = 0
    for i in range(len(lang_test)):
        if lang_test[i] == lang:
            total += 1
            if final_predictions[i] == y_test[i]:
                right += 1
    if total > 0:
        print(lang, right / total)


joblib.dump(rf_model, "model_rf.pkl")
joblib.dump(xgb_model, "model_xgb.pkl")
joblib.dump(gb_model, "model_gb.pkl")
joblib.dump(encoder, "label_encoder.pkl")


print("Random Forest accuracy:", rf_accuracy)
print("XGBoost accuracy:", xgb_accuracy)
print("Gradient Boosting accuracy:", gb_accuracy)
print("Ensemble accuracy:", ensemble_accuracy)


