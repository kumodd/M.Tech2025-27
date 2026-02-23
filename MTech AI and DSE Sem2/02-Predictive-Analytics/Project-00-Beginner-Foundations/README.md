# Project-00: Beginner Foundations 🎓
### "All 12 concepts. One file. Zero confusion."

---

## 🎯 Who Is This For?

- You know Python basics (variables, loops, functions, lists)
- You've **never** built an ML project before
- You want to understand **WHY** each step exists, not just **what** to type
- You're about to tackle Projects 1–3 and want a warm-up

---

## ▶️ How to Run

```bash
# 1. Install dependencies (if not already)
pip install numpy pandas scikit-learn joblib

# 2. Run the single file
cd Project-00-Beginner-Foundations
python learn_predictive_analytics.py
```

That's it. One command. Everything prints to your terminal.

---

## 📖 What's Inside

The entire script is **one file** (`learn_predictive_analytics.py`) with **12 clearly marked sections** — one for each Part in the main README.

| Section | README Part | What You'll See |
|---------|------------|-----------------|
| Part 1 | What is Predictive Analytics? | Concept + real-world analogy |
| Part 2 | Three types of problems | Regression / Classification / Time Series |
| Part 3 | Feature Engineering | Build features from 10 student records |
| Part 4 | Train / Test Split | Why you MUST split data |
| Part 5 | Baseline Model | The dumbest-correct answer |
| Part 6 | Linear & Logistic Regression | Live predictions printed |
| Part 7 | Decision Tree → Random Forest | Overfitting demo live |
| Part 8 | Evaluation Metrics | Confusion matrix + AUC explained |
| Part 9 | Overfitting & Underfitting | Table showing depth vs accuracy |
| Part 10 | Feature Importance | Bar chart in terminal (text) |
| Part 11 | Real-World Pitfalls | 5 things that kill ML projects |
| Part 12 | Save & Load a Model | `joblib` save → load → predict |

---

## 🧪 Dataset Used

A tiny, hand-made **student score** dataset. Simple on purpose.

```
Features:  study_hours, sleep_hours, num_absences
Target 1:  score (0-100)             → Regression
Target 2:  passed (0 = Fail, 1 = Pass) → Classification
```

Why students? Everyone understands the intuition:
- *More study → higher score*
- *More absences → more likely to fail*

This lets you focus on the **ML concepts**, not the domain knowledge.

---

## 🧠 Key Beginner Insights (Spoilers)

1. **Baseline first** — if predicting "Pass" for everyone gives 80% accuracy, your model must beat 80%
2. **Accuracy can lie** — on imbalanced data, use AUC or F1
3. **Overfitting demo** — a Decision Tree with no depth limit gets 100% train accuracy, but fails on test data
4. **Always save the scaler** — the model is useless without it in production
5. **Feature importance** — the model tells you which inputs mattered most

---

## 📁 Files

```
Project-00-Beginner-Foundations/
├── learn_predictive_analytics.py   ← The ONE file to read and run
├── saved_model/
│   ├── student_pass_model.pkl      ← Auto-generated when you run the script
│   └── scaler.pkl                  ← Auto-generated
└── README.md                       ← You are here
```

---

## ➡️ What's Next?

After this, move to the full project:

```
Project-1-Customer-Churn/   ← Same concepts, real business problem, production code
```

---

*Part of the [Predictive Analytics](../README.md) end-to-end project series.*
