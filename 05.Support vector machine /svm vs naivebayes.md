# ⚖️ Naive Bayes vs SVM for Spam Detection

---

## 🔹 Naive Bayes (NB)

- Assumes **word independence** (Bag-of-Words model).
- Very **fast to train & predict** (counts + probabilities).
- Works **surprisingly well** in text problems because words are nearly conditionally independent in practice.
- Needs less data to generalize well.
- Performs well with **small & medium datasets**.

✅ **Pros**: Simple, fast, interpretable.  
❌ **Cons**: Assumption of independence is not always true, performance plateaus on large datasets.

---

## 🔹 Support Vector Machines (SVM)

- Learns a **hyperplane** that maximizes the margin between spam vs. ham.
- Works extremely well in **high-dimensional spaces** (like TF-IDF text vectors).
- Often achieves **higher accuracy** than NB on large, clean datasets.
- More **robust to correlated features** than NB.

✅ **Pros**: High accuracy, strong generalization, great in high-dimensional sparse text.  
❌ **Cons**: Slower training on very large datasets, more memory intensive.

---

## 🚀 Who Wins?

- On **small datasets** → **Naive Bayes** usually works better (faster + less overfitting).
- On **large datasets (millions of emails)** → **SVM usually outperforms NB** in accuracy.
- In real-world spam filters (Gmail, Outlook), SVM and Logistic Regression are often used (sometimes in ensembles with Naive Bayes).

---

## 📌 Rule of Thumb

- Start with **Naive Bayes** as a **baseline** (simple & quick).
- For **production-level accuracy**, use **Linear SVM (or Logistic Regression)** with TF-IDF.
