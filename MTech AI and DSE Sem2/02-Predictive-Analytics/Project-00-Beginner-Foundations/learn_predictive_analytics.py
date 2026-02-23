# =============================================================================
#  PROJECT-00 : Predictive Analytics — Beginner Foundations
#  "All 12 concepts, one file, zero confusion"
# =============================================================================
#
#  WHO IS THIS FOR?
#  → You know Python basics (loops, functions, lists).
#  → You have never built an ML project before.
#  → You want to understand WHAT and WHY, not just copy-paste.
#
#  HOW TO RUN:
#    python learn_predictive_analytics.py
#
#  WHAT YOU'LL LEARN (matches README Parts 1-12):
#    Part 1  → What IS predictive analytics?
#    Part 2  → Three types of prediction problems
#    Part 3  → Feature engineering (turning raw data into useful inputs)
#    Part 4  → Train / Test split (why we need it)
#    Part 5  → Baseline model (the dumbest correct answer)
#    Part 6  → Linear & Logistic Regression
#    Part 7  → Tree-based models (Decision Tree → Random Forest)
#    Part 8  → Evaluation metrics (when accuracy lies)
#    Part 9  → Overfitting & Underfitting
#    Part 10 → Feature importance (why did the model decide this?)
#    Part 11 → Real-world pitfalls (what kills projects)
#    Part 12 → Save & load a model (deploy it)
#
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.linear_model    import LinearRegression, LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (
    mean_absolute_error, accuracy_score,
    confusion_matrix, roc_auc_score, f1_score
)
from sklearn.dummy           import DummyClassifier
import joblib
import os
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: a pretty section printer so output is easy to read
# ─────────────────────────────────────────────────────────────────────────────
def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def note(text):
    print(f"\n  💡 {text}")

def result(text):
    print(f"     ➜  {text}")


# ─────────────────────────────────────────────────────────────────────────────
#  PART 1 : WHAT IS PREDICTIVE ANALYTICS?
# ─────────────────────────────────────────────────────────────────────────────
section("PART 1 — What is Predictive Analytics?")

note("Predictive analytics = using PAST data to guess FUTURE outcomes.")
note("It's not magic. It's just finding patterns and reusing them.")

print("""
  Real example in 3 lines:
    Past data   →  "Students who studied 5h/day scored 90%"
    Pattern     →  More study hours → higher score
    Prediction  →  "If Ravi studies 4h/day, he'll probably score ~85%"
""")

note("The Predictive Analytics process is always:")
print("""
    Business Question
        ↓
    Collect Data
        ↓
    Clean + Engineer Features
        ↓
    Build Model
        ↓
    Evaluate (is it good enough?)
        ↓
    Deploy (use it in real life)
""")


# ─────────────────────────────────────────────────────────────────────────────
#  PART 2 : THREE TYPES OF PREDICTION PROBLEMS
# ─────────────────────────────────────────────────────────────────────────────
section("PART 2 — Three Types of Prediction Problems")

note("Before writing ANY code, ask: What does the answer LOOK like?")

print("""
  TYPE 1 → REGRESSION  (Answer is a number)
    Examples: house price, salary, temperature tomorrow
    Output:   ₹45,00,000  or  ₹82,000  or  32.5°C

  TYPE 2 → CLASSIFICATION  (Answer is a category)
    Examples: spam/not-spam, will churn / won't, fraud / legit
    Output:   Yes / No   or   Cat / Dog / Fish

  TYPE 3 → TIME SERIES  (Answer is a sequence of future numbers)
    Examples: next 30 days of sales, tomorrow's stock price
    Output:   [₹1000, ₹1050, ₹980, ₹1100, ...]
""")

note("TRICK QUESTION: Is customer satisfaction score regression or classification?")
print("""
    If the score is 1-10 continuous  →  REGRESSION
    If the score is Good/Neutral/Bad →  CLASSIFICATION
    The BUSINESS definition decides, not the data!
""")


# ─────────────────────────────────────────────────────────────────────────────
#  PART 3 : FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
section("PART 3 — Feature Engineering (The Most Important Skill)")

note("Feature = one piece of information you give the model.")
note("Feature Engineering = creating BETTER inputs from raw data.")

# Let's use a tiny, hand-typed dataset of 10 students
print("\n  Raw data (what we START with):")
raw = pd.DataFrame({
    "student_id":   ["S1","S2","S3","S4","S5","S6","S7","S8","S9","S10"],
    "study_hours":  [  2,   5,   1,   8,   3,   6,   4,   7,   2,    9],
    "sleep_hours":  [  9,   7,   8,   6,   7,   7,   8,   6,   9,    5],
    "num_absences": [  5,   1,   8,   0,   4,   2,   3,   1,   6,    0],
    "score":        [ 42,  78,  35,  92,  55,  80,  65,  88,  40,   95],
})
print(raw.to_string(index=False))

note("Now we ENGINEER better features from what we have:")
raw["study_to_sleep_ratio"] = (raw["study_hours"] / raw["sleep_hours"]).round(2)
raw["is_regular"]           = (raw["num_absences"] <= 2).astype(int)  # 1=regular, 0=irregular
raw["study_score"]          = raw["study_hours"] - raw["num_absences"]  # net productive effort

print("\n  After feature engineering:")
cols = ["student_id", "study_hours", "num_absences",
        "study_to_sleep_ratio", "is_regular", "study_score", "score"]
print(raw[cols].to_string(index=False))

note("Why do this? Raw 'study_hours' is ok. But 'study_score' (study minus absences)")
print("  captures NET effort much better. Good features = better model.")


# ─────────────────────────────────────────────────────────────────────────────
#  PART 4 : TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
section("PART 4 — Train / Test Split")

note("WHY do we split? If we test on the same data we trained on,")
print("  the model just MEMORISES — like a student who only practices")
print("  past papers. You need a fresh 'exam' it has never seen.")

# We'll use a bigger dataset from now — 200 students (synthetic)
np.random.seed(42)
n = 200
study  = np.random.randint(1, 10, n)
sleep  = np.random.randint(4, 10, n)
absent = np.random.randint(0, 8, n)
score  = (study * 9) - (absent * 4) + np.random.normal(0, 5, n)
score  = np.clip(score, 0, 100).astype(int)

students = pd.DataFrame({
    "study_hours":  study,
    "sleep_hours":  sleep,
    "num_absences": absent,
    "score":        score,
    "passed":       (score >= 50).astype(int)  # 1=pass, 0=fail
})

X = students[["study_hours", "sleep_hours", "num_absences"]]
y_score  = students["score"]    # for regression
y_passed = students["passed"]   # for classification

X_train, X_test, y_train, y_test = train_test_split(
    X, y_score, test_size=0.20, random_state=42
)

result(f"Total students : {n}")
result(f"Training set   : {len(X_train)} students (80%) — model LEARNS from this")
result(f"Test set       : {len(X_test)} students (20%) — we EVALUATE on this")

note("GOLDEN RULE: Never touch the test set until the very end.")
print("  Peeking at it is like giving the student the final exam answers early!")


# ─────────────────────────────────────────────────────────────────────────────
#  PART 5 : BASELINE MODEL
# ─────────────────────────────────────────────────────────────────────────────
section("PART 5 — Baseline Model (Always Build This First!)")

note("Baseline = the DUMBEST possible prediction that still makes sense.")
note("If your fancy model can't beat the dumbest guess, it's useless.")

# Regression baseline: always predict the AVERAGE score
average_score = y_train.mean()
baseline_preds = [average_score] * len(y_test)
baseline_mae   = mean_absolute_error(y_test, baseline_preds)

result(f"Average score in training data : {average_score:.1f}")
result(f"If we predict {average_score:.0f} for EVERY student:")
result(f"  Baseline MAE = {baseline_mae:.1f} points off on average")

# Classification baseline: always predict the MAJORITY class
_, y_pass_test  = train_test_split(students["passed"], test_size=0.2, random_state=42)
_, y_pass_train = train_test_split(students["passed"], test_size=0.8, random_state=42)
most_common = y_pass_train.mode()[0]
bl_class_preds = [most_common] * len(y_pass_test)
bl_accuracy = accuracy_score(y_pass_test, bl_class_preds)

result(f"\nMost common class in training : {'Pass' if most_common==1 else 'Fail'}")
result(f"If we predict {'Pass' if most_common==1 else 'Fail'} for EVERY student:")
result(f"  Baseline Accuracy = {bl_accuracy:.1%}")

note("Your model MUST beat these numbers to prove it's learning something!")


# ─────────────────────────────────────────────────────────────────────────────
#  PART 6 : LINEAR & LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
section("PART 6 — Linear & Logistic Regression")

# --- Linear Regression ---
note("LINEAR REGRESSION: predicts a NUMBER (score, price, salary)")
print("  Idea: draw a best-fit straight line through your data.")
print("  Formula: score = (w1 × study_hours) + (w2 × sleep) + (w3 × absences) + bias")

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_mae   = mean_absolute_error(y_test, lr_preds)

result(f"Linear Regression MAE : {lr_mae:.1f} points  (baseline was {baseline_mae:.1f})")
result(f"Improvement over baseline: {baseline_mae - lr_mae:.1f} points better")

note("What the model learned (coefficients):")
for feature, coef in zip(X.columns, lr.coef_):
    direction = "↑ score" if coef > 0 else "↓ score"
    print(f"    1 more {feature:20s} → {coef:+.2f} {direction}")

result(f"\nPredict score for: study=7h, sleep=7h, absences=1")
new_student = [[7, 7, 1]]
predicted   = lr.predict(new_student)[0]
result(f"  Predicted score: {predicted:.0f}/100")

# --- Logistic Regression ---
note("\nLOGISTIC REGRESSION: predicts a CATEGORY (pass/fail, yes/no)")
print("  Same idea, but the output is a PROBABILITY (0 to 1).")
print("  P > 0.5 → Predict 'Pass'.  P ≤ 0.5 → Predict 'Fail'.")

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, students["passed"], test_size=0.2, random_state=42
)
log_reg = LogisticRegression()
log_reg.fit(X_train_c, y_train_c)
log_preds  = log_reg.predict(X_test_c)
log_acc    = accuracy_score(y_test_c, log_preds)

result(f"Logistic Regression Accuracy : {log_acc:.1%}  (baseline was {bl_accuracy:.1%})")

prob = log_reg.predict_proba([[7, 7, 1]])[0]
result(f"\nFor study=7h, sleep=7h, absences=1:")
result(f"  Probability of FAILING : {prob[0]:.1%}")
result(f"  Probability of PASSING : {prob[1]:.1%}")
result(f"  Prediction → {'PASS ✅' if prob[1] > 0.5 else 'FAIL ❌'}")


# ─────────────────────────────────────────────────────────────────────────────
#  PART 7 : TREE-BASED MODELS
# ─────────────────────────────────────────────────────────────────────────────
section("PART 7 — Tree-Based Models (Decision Tree → Random Forest)")

note("Decision Tree: asks YES/NO questions until it reaches an answer.")
print("""
  Example tree in your head:
    Does study_hours > 5?
      YES → Does num_absences < 3?  → YES → PASS
                                    → NO  → FAIL
      NO  → Does sleep_hours > 7?   → YES → Borderline (PASS)
                                    → NO  → FAIL
""")

# Decision Tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train_c, y_train_c)
dt_preds = dt.predict(X_test_c)
dt_acc   = accuracy_score(y_test_c, dt_preds)

result(f"Decision Tree Accuracy: {dt_acc:.1%}")

note("PROBLEM with Decision Trees: they memorise too easily!")
dt_overfit = DecisionTreeClassifier(max_depth=None, random_state=42)  # no limit!
dt_overfit.fit(X_train_c, y_train_c)
train_acc = dt_overfit.score(X_train_c, y_train_c)
test_acc  = dt_overfit.score(X_test_c, y_test_c)
result(f"Deep tree — Train accuracy: {train_acc:.1%},  Test accuracy: {test_acc:.1%}")
print("  See the gap? That's OVERFITTING — memorised training data, fails on new data.")

note("RANDOM FOREST: 100 trees, each seeing DIFFERENT random slices of data.")
print("  They all vote → majority wins → much more reliable.")

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train_c, y_train_c)
rf_preds = rf.predict(X_test_c)
rf_acc   = accuracy_score(y_test_c, rf_preds)

result(f"Random Forest Accuracy: {rf_acc:.1%}  ← Usually the best!")

note("Summary so far:")
print(f"""
    Baseline (Majority Class) : {bl_accuracy:.1%}
    Logistic Regression       : {log_acc:.1%}
    Decision Tree (depth=3)   : {dt_acc:.1%}
    Random Forest             : {rf_acc:.1%}
""")


# ─────────────────────────────────────────────────────────────────────────────
#  PART 8 : EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
section("PART 8 — Evaluation Metrics (When Accuracy Lies)")

note("Imagine: 95 students pass, 5 fail. If we predict 'PASS' for everyone:")
print("  Accuracy = 95%  ← Looks amazing! But we never caught the 5 failures.")
print("  This is WHY accuracy alone is a bad metric for imbalanced data.\n")

note("CONFUSION MATRIX: the table that tells the whole truth.")
cm = confusion_matrix(y_test_c, rf_preds)
print(f"""
        Predicted FAIL  Predicted PASS
Actual FAIL  {cm[0,0]:4d}  (True Neg)    {cm[0,1]:4d}  (False Pos ← wrongly said Pass)
Actual PASS  {cm[1,0]:4d}  (False Neg)   {cm[1,1]:4d}  (True Pos  ← correctly said Pass)
""")

tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

note("What each number means in plain English:")
result(f"Precision  = {precision:.1%}  — Of all predicted PASS, how many actually passed?")
result(f"Recall     = {recall:.1%}  — Of all actual PASS, how many did we CATCH?")
result(f"F1 Score   = {f1:.1%}  — Balance between precision and recall")

try:
    rf_prob   = rf.predict_proba(X_test_c)[:, 1]
    auc_score = roc_auc_score(y_test_c, rf_prob)
    result(f"AUC-ROC    = {auc_score:.3f}  — 0.5=random, 1.0=perfect. Higher is better.")
except Exception:
    pass

note("WHEN to use which metric?")
print("""
    → Balanced data, all errors equal?          Use Accuracy
    → Imbalanced data (rare events)?            Use AUC-ROC or F1
    → Missing a positive is very costly?        Focus on Recall
      (e.g., missing a cancer patient = bad)
    → False alarms are very costly?             Focus on Precision
      (e.g., wrongly arresting someone = bad)
""")


# ─────────────────────────────────────────────────────────────────────────────
#  PART 9 : OVERFITTING & UNDERFITTING
# ─────────────────────────────────────────────────────────────────────────────
section("PART 9 — Overfitting & Underfitting")

note("Think of it like studying for an exam:")
print("""
  UNDERFITTING   = You barely studied. Can't answer easy OR hard questions.
                   Model is too simple. Fails on BOTH training and test data.

  OVERFITTING    = You memorised the textbook word-for-word. You answer
                   practice questions perfectly, but fail the real exam
                   because it uses different words.
                   Model does great on training, BAD on test data.

  JUST RIGHT     = You understood the concepts. Do well on both.
""")

# Demonstrate with different tree depths
note("Live demo — notice how training vs test accuracy changes with tree depth:")
print(f"\n  {'Depth':>6}  {'Train Acc':>10}  {'Test Acc':>10}  {'Gap':>8}  Verdict")
print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*15}")
for depth in [1, 2, 3, 5, 10, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train_c, y_train_c)
    tr_a = tree.score(X_train_c, y_train_c)
    te_a = tree.score(X_test_c,  y_test_c)
    gap  = tr_a - te_a
    label = "Depth"
    d_str = str(depth) if depth else "None"
    if gap > 0.15:
        verdict = "⚠️  OVERFITTING"
    elif tr_a < 0.75:
        verdict = "📉 UNDERFITTING"
    else:
        verdict = "✅  Good"
    print(f"  {d_str:>6}  {tr_a:>9.1%}  {te_a:>9.1%}  {gap:>7.1%}  {verdict}")

note("The sweet spot is where test accuracy is highest and the gap is small.")


# ─────────────────────────────────────────────────────────────────────────────
#  PART 10 : FEATURE IMPORTANCE / EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────
section("PART 10 — Feature Importance (Why Did the Model Decide This?)")

note("Black box models are dangerous. Always ask: WHICH features mattered most?")

# Use the Random Forest's built-in feature importance
importances = rf.feature_importances_
feat_importance = pd.DataFrame({
    "Feature":    list(X.columns),
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\n  Feature Importance (Random Forest):")
for _, row in feat_importance.iterrows():
    bar = "█" * int(row["Importance"] * 40)
    print(f"    {row['Feature']:20s}  {row['Importance']:.3f}  {bar}")

note("Interpretation:")
top_feat = feat_importance.iloc[0]["Feature"]
print(f"    '{top_feat}' is the most predictive feature.")
print("    The model mostly makes decisions based on this.")
print("    You can use this to tell a business: 'Focus on THIS to reduce failures.'")

note("Logistic Regression coefficients (also explainable):")
for feat, coef in zip(X.columns, log_reg.coef_[0]):
    direction = "↑ PASS chance" if coef > 0 else "↑ FAIL chance"
    print(f"    1 more {feat:20s} → {direction} by {abs(coef):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
#  PART 11 : REAL-WORLD PITFALLS
# ─────────────────────────────────────────────────────────────────────────────
section("PART 11 — Real-World Pitfalls (Why Projects Fail)")

note("Pitfall 1: DATA LEAKAGE — using future info to predict the future")
print("""
    Bad example:
      Predicting if a student will FAIL, using "score" as a feature.
      The score IS the answer — of course it predicts itself!
      In production, you won't have the score when you need the prediction.

    Always ask: "Will this feature be AVAILABLE at prediction time?"
""")

note("Pitfall 2: WRONG METRICS — optimising for the wrong thing")
print("""
    Bad: "Our fraud detector has 99.9% accuracy!" 
    Truth: Only 0.1% of transactions are fraud.
    Predicting 'not fraud' for everything = 99.9% accuracy = useless.
    
    Fix: Use Precision, Recall, AUC-ROC instead.
""")

note("Pitfall 3: FORGETTING THE BASELINE — claiming 80% is 'good'")
print("""
    If 80% of customers don't churn, predicting 'no churn' always =80%.
    Your model MUST beat the simple baseline to prove its worth.
""")

note("Pitfall 4: DATA DRIFT — model works today, fails next year")
print("""
    Customer behaviour changes. Economic conditions change.
    A model trained on 2020 data may fail badly in 2024.
    
    Fix: Monitor predictions, retrain regularly.
""")

note("Pitfall 5: FORGETTING THE SCALER in deployment")
print("""
    You trained with StandardScaler. You saved the model. You forgot the scaler.
    Production input: raw numbers. Model expects scaled numbers. Predictions = garbage.
    
    Fix: Save the scaler ALONG with the model (next section!).
""")


# ─────────────────────────────────────────────────────────────────────────────
#  PART 12 : SAVE & LOAD A MODEL (DEPLOYMENT)
# ─────────────────────────────────────────────────────────────────────────────
section("PART 12 — Saving & Loading a Model (Making It Usable)")

note("Training happens once. Prediction happens thousands of times a day.")
note("You save the trained model to a file, then load it wherever needed.")

from sklearn.preprocessing import StandardScaler

# Fit a scaler + model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_c)
X_test_scaled  = scaler.transform(X_test_c)

final_model = LogisticRegression()
final_model.fit(X_train_scaled, y_train_c)

final_acc = final_model.score(X_test_scaled, y_test_c)
result(f"Final model accuracy (with scaling): {final_acc:.1%}")

# Save
SAVE_DIR = os.path.join("saved_model")
os.makedirs(SAVE_DIR, exist_ok=True)
joblib.dump(final_model, os.path.join(SAVE_DIR, "student_pass_model.pkl"))
joblib.dump(scaler,      os.path.join(SAVE_DIR, "scaler.pkl"))

note("Saved! Files created:")
print(f"    saved_model/student_pass_model.pkl")
print(f"    saved_model/scaler.pkl")

# Load and use
note("Now pretend this is a DIFFERENT computer (or a web server)...")
loaded_model  = joblib.load(os.path.join(SAVE_DIR, "student_pass_model.pkl"))
loaded_scaler = joblib.load(os.path.join(SAVE_DIR, "scaler.pkl"))

def predict_student(study_hours, sleep_hours, num_absences):
    """
    This is what your production API would call.
    Input: raw student data
    Output: PASS or FAIL + probability
    """
    # Step 1: create feature row
    raw_features = [[study_hours, sleep_hours, num_absences]]

    # Step 2: apply the SAME scaler used in training (this is key!)
    scaled_features = loaded_scaler.transform(raw_features)

    # Step 3: predict
    proba      = loaded_model.predict_proba(scaled_features)[0]
    prediction = loaded_model.predict(scaled_features)[0]

    return {
        "prediction":   "PASS ✅" if prediction == 1 else "FAIL ❌",
        "pass_prob":    f"{proba[1]:.1%}",
        "fail_prob":    f"{proba[0]:.1%}",
    }

note("Testing the production function:")
test_cases = [
    (8, 7, 0),    # high study, good sleep, no absences → should PASS
    (2, 9, 6),    # low study, too much sleep, many absences → should FAIL
    (5, 7, 2),    # average student
]
for study, sleep, absent in test_cases:
    out = predict_student(study, sleep, absent)
    result(f"study={study}h  sleep={sleep}h  absences={absent}  →  {out['prediction']}  "
           f"(Pass: {out['pass_prob']}, Fail: {out['fail_prob']})")


# ─────────────────────────────────────────────────────────────────────────────
#  FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
section("FINAL SUMMARY — What You Just Learned")
print(f"""
  Part 1  ✅  Predictive Analytics = patterns from past to predict future
  Part 2  ✅  3 types: Regression (number), Classification (category), Time Series
  Part 3  ✅  Feature Engineering = creating useful inputs from raw data
  Part 4  ✅  Train/Test split = honest evaluation on unseen data
  Part 5  ✅  Always build a baseline first — your model must beat it
  Part 6  ✅  Linear Reg (numbers) | Logistic Reg (categories + probability)
  Part 7  ✅  Decision Tree (one set of rules) | Random Forest (100 trees vote)
  Part 8  ✅  Use AUC, F1, Precision, Recall — not just accuracy
  Part 9  ✅  Overfitting = memorised. Underfitting = too simple. Find the middle.
  Part 10 ✅  Feature importance = model transparency (why did it decide this?)
  Part 11 ✅  5 pitfalls: leakage, wrong metrics, no baseline, drift, forgotten scaler
  Part 12 ✅  Save model + scaler together. Load them in production. That's deployment.

  ─────────────────────────────────────────────────────────────
  NEXT STEP: Open Project-1-Customer-Churn/ — all of the above,
             but applied to a real business problem end-to-end.
  ─────────────────────────────────────────────────────────────
""")
