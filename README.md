# 🩺 HealthInsight AI — Healthcare Analysis & Risk Prediction

A professional, multi-section **Streamlit** dashboard that uses a trained
`RandomForestClassifier` to estimate a patient's health risk from common
clinical indicators, and turns that result into an easy-to-understand,
personalized diet & lifestyle plan.

🔗 **GitHub Repo:** [github.com/Lovepreetsingh67/HealthInsight-AI](https://github.com/Lovepreetsingh67/HealthInsight-AI)

---

## 📖 About the Project

**HealthInsight AI** is a healthcare analysis and risk prediction system built
around a trained Random Forest classifier. A user enters a patient's basic
clinical details — age, BMI, glucose, blood pressure, cholesterol, heart rate,
and a few other indicators — and the app instantly returns:

- A **risk score** (Low / Moderate / High) with a clear visual gauge
- **KPI cards** showing how each key metric compares to the patient's last check
- **Health tips** written in plain, easy-to-understand language
- A **personalized diet plan** — real, specific foods to eat or limit based on the patient's results
- A **history log** of every past assessment, with trend charts and CSV export
- A downloadable **PDF report** for each result
- **Dark mode / Light mode**, and a layout that adapts to both mobile and desktop

The goal is to make health-risk screening approachable — not just a number,
but clear guidance a patient can actually act on.

---

## ✨ Features

- **AI-Powered Risk Prediction** — Random Forest model trained on real clinical health indicators
- **Interactive Patient Form** — form organized by category (Demographics, Vitals, Lab Results), with a built-in **BMI calculator**
- **Visual Risk Meter** — color-coded gauge plus KPI cards with change-vs-last-check indicators
- **Personalized Diet Plans** — condition-specific, understandable food guidance generated from glucose, blood pressure, cholesterol and BMI
- **Health Tips Panel** — plain-language tips tailored to each metric
- **Trends & History** — charts of past assessments, filterable by patient, with **CSV export**
- **PDF Reports** — download a full, shareable PDF summary of any assessment
- **Dark Mode / Light Mode** — toggle from the sidebar
- **Responsive Layout** — works on both mobile and desktop, with full tooltip hints on tap/hover


## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — web app framework
- [scikit-learn](https://scikit-learn.org/) — Random Forest model (loaded from a pre-trained `.pkl` file)
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) — charts and visualizations
- [pandas](https://pandas.pydata.org/) — data handling & history log
- [fpdf2](https://pyfpdf.github.io/fpdf2/) — PDF report generation
- [streamlit-option-menu](https://github.com/victoryhb/streamlit-option-menu) *(optional)* — sidebar navigation menu

  **DEVELOPED BY**
  Lovepreet Singh

---

## 📂 Project Structure
