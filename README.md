🩺 HealthInsight AI — Healthcare Analysis & Risk Prediction

A professional, multi-section Streamlit dashboard that uses a trained RandomForestClassifier to estimate a patient's health risk from common clinical indicators, and turns that result into an easy-to-understand, personalized diet & lifestyle plan.

🔗 GitHub Repo: github.com/Lovepreetsingh67/HealthInsight-AI

📖 About the Project

HealthInsight AI is a healthcare analysis and risk prediction system built around a trained Random Forest classifier. A user enters a patient's basic clinical details — age, BMI, glucose, blood pressure, cholesterol, heart rate, and a few other indicators — and the app instantly returns:

A risk score (Low / Moderate / High) with a clear visual gauge
KPI cards showing how each key metric compares to the patient's last check
Health tips written in plain, easy-to-understand language
A personalized diet plan — real, specific foods to eat or limit based on the patient's results
A history log of every past assessment, with trend charts and CSV export
A downloadable PDF report for each result
Dark mode / Light mode, and a layout that adapts to both mobile and desktop

The goal is to make health-risk screening approachable — not just a number, but clear guidance a patient can actually act on.

✨ Features
AI-Powered Risk Prediction — Random Forest model trained on real clinical health indicators
Interactive Patient Form — form organized by category (Demographics, Vitals, Lab Results), with a built-in BMI calculator
Visual Risk Meter — color-coded gauge plus KPI cards with change-vs-last-check indicators
Personalized Diet Plans — condition-specific, understandable food guidance generated from glucose, blood pressure, cholesterol and BMI
Health Tips Panel — plain-language tips tailored to each metric
Trends & History — charts of past assessments, filterable by patient, with CSV export
PDF Reports — download a full, shareable PDF summary of any assessment
Dark Mode / Light Mode — toggle from the sidebar
Responsive Layout — works on both mobile and desktop, with full tooltip hints on tap/hover


Home	Risk Prediction	Trends & History
Show Image	Show Image	Show Image
🛠️ Tech Stack
Streamlit — web app framework
scikit-learn — Random Forest model (loaded from a pre-trained .pkl file)
Matplotlib & Seaborn — charts and visualizations
pandas — data handling & history log
fpdf2 — PDF report generation
streamlit-option-menu (optional) — sidebar navigation menu
📂 Project Structure
├── app.py                 # Main Streamlit application
├── healthcare_model.pkl   # Trained RandomForestClassifier (model + feature list)
├── history.csv            # Auto-created — stores past assessments (persists across sessions)
└── README.md


🚀 Running the App
bash
git clone https://github.com/Lovepreetsingh67/HealthInsight-AI.git
cd HealthInsight-AI
streamlit run app.py


📋 Input Fields
Field	Unit	Description
Age	years	Patient's age
BMI	kg/m²	Body Mass Index (or use the built-in calculator)
Glucose	mg/dL	Fasting blood glucose
Blood Pressure	mmHg	Systolic blood pressure
Skin Thickness	mm	Triceps skinfold thickness
Insulin	mu U/mL	2-hour serum insulin
Pregnancies	count	Total number of pregnancies
Diabetes Pedigree	score	Family-history-based diabetes risk score
Cholesterol	mg/dL	Total blood cholesterol
Heart Rate	bpm	Resting heart rate

Patient Name is required for every assessment so results can be tracked correctly per patient across visits.

⚠️ Disclaimer

This application is intended for educational and informational purposes only. It does not provide medical diagnoses and should never replace consultation with a qualified healthcare professional.

👤 Developed By

Lovepreet Singh
