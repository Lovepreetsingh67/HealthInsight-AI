"""
Step 3: Streamlit Frontend Application
This is the main web application built using Streamlit.
It allows users to interactively:
1. Input health metrics to analyze patient risk using our trained Machine Learning model.
2. View historical dashboard trends for the week.
3. Access personalized dietary recommendations based on their health results.

To run this app, execute in your terminal:
streamlit run app.py
install numpy , pandas , matplotlib , scikit-learn , streamlit , seaborn
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# Set page config for a professional dashboard look
st.set_page_config(
    page_title="VitalShield - Healthcare Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom premium CSS styles for visual excellence
st.markdown("""
<style>
    /* Styling headers */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 2rem;
    }
    
    /* Styling sidebar */
    .css-1d391kg {
        background-color: #f3f4f6;
    }
    
    /* Elegant card container */
    .health-card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
        transition: transform 0.2s ease;
    }
    
    .health-card:hover {
        transform: translateY(-2px);
    }
    
    /* Healthy Status Glow Border */
    .status-healthy {
        border-left: 6px solid #10b981;
    }
    
    /* Risk Status Glow Border */
    .status-risk {
        border-left: 6px solid #ef4444;
    }
    
    /* Styling section headers inside cards */
    .section-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Food guide list styling */
    .diet-box {
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .diet-include {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
    }
    
    .diet-avoid {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load the trained model
@st.cache_resource
def load_model():
    model_path = "model/healthcare_model.pkl"
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as file:
                model_data = pickle.load(file)
            return model_data
        except Exception as e:
            st.error(f"Error loading model pickle: {e}")
            return None
    else:
        st.warning("Model file not found! Please run train_model.py first.")
        return None

# Load the model and features
model_data = load_model()

# Setup sidebar navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80)
st.sidebar.markdown("<h2 style='margin-bottom: 0px;'>VitalShield</h2><p style='color:gray; font-size:0.85rem; margin-top:0px;'>Smart Healthcare Analytics</p>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation Menu",
    ["🩺 Analyse Health", "📈 Weekly Analysis Graph", "🥗 Diet Chart"]
)

# ----------------- PAGE 1: ANALYSE HEALTH -----------------
if page == "🩺 Analyse Health":
    st.markdown('<h1 class="main-title">🩺 Health Assessment & Risk Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Enter patient health indicators below to predict potential health risk zones instantly.</p>', unsafe_allow_html=True)
    
    if model_data is None:
        st.error("Please run the `train_model.py` script first to generate and train the ML model.")
    else:
        model = model_data['model']
        feature_names = model_data['features']
        
        # Create an input form split into 2 columns for a clean interface
        with st.form("health_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📋 Basic Demographics")
                age = st.slider("Age (years)", min_value=18, max_value=90, value=45, help="Patient age in years")
                bmi = st.slider("BMI (Body Mass Index)", min_value=15.0, max_value=45.0, value=25.0, step=0.1, help="Weight (kg) / Height^2 (m)")
                pregnancies = st.number_input("Pregnancies", min_value=0, max_value=15, value=2, step=1, help="Number of times pregnant")
                diabetes_pedigree = st.slider("Diabetes Pedigree Function", min_value=0.05, max_value=2.5, value=0.5, step=0.01, help="Family history score of diabetes")
                heart_rate = st.slider("Resting Heart Rate (bpm)", min_value=40, max_value=140, value=72, help="Heart beats per minute at rest")
                
            with col2:
                st.markdown("### 🧪 Clinical Test Vitals")
                glucose = st.slider("Glucose Level (mg/dL)", min_value=50.0, max_value=250.0, value=120.0, step=1.0, help="Fasting blood sugar level")
                blood_pressure = st.slider("Systolic Blood Pressure (mmHg)", min_value=80.0, max_value=200.0, value=125.0, step=1.0, help="Systolic BP")
                cholesterol = st.slider("Cholesterol Level (mg/dL)", min_value=100.0, max_value=350.0, value=195.0, step=1.0, help="Total cholesterol level")
                skin_thickness = st.slider("Skin Thickness (mm)", min_value=5.0, max_value=60.0, value=28.0, step=0.5, help="Triceps skin fold thickness")
                insulin = st.slider("Insulin Level (μU/mL)", min_value=10.0, max_value=400.0, value=150.0, step=1.0, help="2-Hour serum insulin level")
                
            # Submit button
            submit_button = st.form_submit_button("Analyze Patient Health Status")
            
        if submit_button:
            # Prepare features list in the correct order matching the trained model features
            input_features = [
                age, bmi, glucose, blood_pressure, skin_thickness,
                insulin, pregnancies, diabetes_pedigree, cholesterol, heart_rate
            ]
            
            # Predict outcome and probability
            # We wrap input features inside a DataFrame with feature names to avoid warnings
            input_df = pd.DataFrame([input_features], columns=feature_names)
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1] # Probability of being "At Risk" (class 1)
            
            # Save this prediction session to st.session_state so the Diet tab can read it dynamically
            st.session_state['latest_analysis'] = {
                'age': age,
                'bmi': bmi,
                'glucose': glucose,
                'blood_pressure': blood_pressure,
                'cholesterol': cholesterol,
                'risk_level': int(prediction),
                'risk_percentage': float(probability * 100)
            }
            
            # Write query record to history.csv for the weekly metrics page
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_record = pd.DataFrame([{
                'timestamp': timestamp,
                'age': age,
                'bmi': bmi,
                'glucose': glucose,
                'blood_pressure': blood_pressure,
                'skin_thickness': skin_thickness,
                'insulin': insulin,
                'pregnancies': pregnancies,
                'diabetes_pedigree': diabetes_pedigree,
                'cholesterol': cholesterol,
                'heart_rate': heart_rate,
                'risk_level': int(prediction),
                'risk_percentage': round(float(probability * 100), 1)
            }])
            
            # Append record without index or headers
            new_record.to_csv("history.csv", mode='a', header=False, index=False)
            
            # Display results in beautiful customized cards
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")
            
            res_col1, res_col2 = st.columns([2, 1])
            
            with res_col1:
                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class="health-card status-risk">
                            <div class="section-title" style="color:#ef4444;">⚠️ ALERT: High Health Risk Risk Flagged</div>
                            <p>Our machine learning analysis indicates that this patient exhibits multiple clinical risk markers. Further clinical evaluation is recommended.</p>
                            <p><b>Model Risk Probability Score:</b> <span style="font-size:1.5rem; color:#ef4444; font-weight:700;">{probability * 100:.1f}%</span></p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="health-card status-healthy">
                            <div class="section-title" style="color:#10b981;">✅ STATUS: Healthy / Low Risk</div>
                            <p>Good news! The model predicts that this patient is currently within healthy boundaries. Continue maintaining a balanced lifestyle.</p>
                            <p><b>Model Risk Probability Score:</b> <span style="font-size:1.5rem; color:#10b981; font-weight:700;">{probability * 100:.1f}%</span></p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            
            with res_col2:
                st.markdown("### 🔔 What to do next?")
                if prediction == 1:
                    st.info(
                        "👉 **Check out the Diet Chart tab!** We have customized a nutrition plan specifically based on this patient's flagged health risk metrics."
                    )
                else:
                    st.success(
                        "👉 **Maintain wellness!** Review the Diet Chart tab for daily wellness guidance and nutrition maintenance."
                    )
                    
            # Detailed breakdown of individual features
            st.markdown("### 🔍 Specific Metric Insights")
            breakdown_cols = st.columns(4)
            
            with breakdown_cols[0]:
                val_color = "#ef4444" if glucose > 130 else "#10b981"
                st.markdown(f"**Fasting Glucose**  \n<span style='font-size:1.5rem; font-weight:700; color:{val_color};'>{glucose} mg/dL</span>  \n*(Healthy: < 100)*", unsafe_allow_html=True)
                
            with breakdown_cols[1]:
                val_color = "#ef4444" if blood_pressure > 135 else "#10b981"
                st.markdown(f"**Systolic Blood Pressure**  \n<span style='font-size:1.5rem; font-weight:700; color:{val_color};'>{blood_pressure} mmHg</span>  \n*(Healthy: < 120)*", unsafe_allow_html=True)
                
            with breakdown_cols[2]:
                val_color = "#ef4444" if cholesterol > 200 else "#10b981"
                st.markdown(f"**Total Cholesterol**  \n<span style='font-size:1.5rem; font-weight:700; color:{val_color};'>{cholesterol} mg/dL</span>  \n*(Healthy: < 200)*", unsafe_allow_html=True)
                
            with breakdown_cols[3]:
                val_color = "#ef4444" if bmi > 27 else "#10b981"
                st.markdown(f"**BMI**  \n<span style='font-size:1.5rem; font-weight:700; color:{val_color};'>{bmi}</span>  \n*(Healthy: 18.5 - 24.9)*", unsafe_allow_html=True)


# ----------------- PAGE 2: WEEKLY ANALYSIS GRAPH -----------------
elif page == "📈 Weekly Analysis Graph":
    st.markdown('<h1 class="main-title">📈 Weekly Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Visual representation of vital checks and analysis trends recorded this week.</p>', unsafe_allow_html=True)
    
    if not os.path.exists("history.csv"):
        st.warning("No assessment logs found! Run an assessment on the 'Analyse Health' page first.")
    else:
        # Load patient assessment history
        history_df = pd.read_csv("history.csv")
        
        # Ensure timestamp is parsed
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df['date'] = history_df['timestamp'].dt.date
        
        # Display high-level metrics
        total_checks = len(history_df)
        high_risk_count = len(history_df[history_df['risk_level'] == 1])
        healthy_count = len(history_df[history_df['risk_level'] == 0])
        avg_risk_prob = history_df['risk_percentage'].mean()
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Total Vitals Checked", total_checks)
        m_col2.metric("At-Risk Flagged", high_risk_count, delta=f"{high_risk_count/total_checks*100:.1f}%", delta_color="inverse")
        m_col3.metric("Healthy Flagged", healthy_count, delta=f"{healthy_count/total_checks*100:.1f}%")
        m_col4.metric("Avg Risk Probability", f"{avg_risk_prob:.1f}%")
        
        st.markdown("---")
        
        # Columns for side-by-side charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("### 📅 Daily Assessment Volume")
            # Group assessments by date
            daily_assessments = history_df.groupby('date').size().reset_index(name='Patient Count')
            daily_assessments['date'] = daily_assessments['date'].astype(str) # Convert to string to prevent index plotting issues
            
            # Interactive Streamlit Bar Chart
            st.bar_chart(data=daily_assessments, x='date', y='Patient Count', color="#4f46e5")
            st.caption("This chart displays the number of patients analyzed each day of the week.")
            
        with chart_col2:
            st.markdown("### 🩺 Vital Markers Tracking (Average by Day)")
            # Average glucose & cholesterol levels grouped by date
            daily_vitals = history_df.groupby('date')[['glucose', 'cholesterol', 'blood_pressure']].mean().reset_index()
            daily_vitals['date'] = daily_vitals['date'].astype(str)
            
            # Interactive Line Chart for tracking trends
            st.line_chart(data=daily_vitals, x='date', y=['glucose', 'cholesterol', 'blood_pressure'], color=["#10b981", "#f59e0b", "#3b82f6"])
            st.caption("Monitors the trend of average glucose, cholesterol, and systolic BP of tested patients.")

        # Show patient log history table at the bottom
        st.markdown("### 📋 Recent Assessment Log")
        # Format table for cleaner display
        formatted_df = history_df.copy()
        formatted_df['risk_level'] = formatted_df['risk_level'].map({0: "Healthy", 1: "At Risk"})
        formatted_df = formatted_df[['timestamp', 'age', 'bmi', 'glucose', 'blood_pressure', 'cholesterol', 'risk_level', 'risk_percentage']]
        
        # Sort newest first
        formatted_df = formatted_df.sort_values(by='timestamp', ascending=False)
        
        st.dataframe(formatted_df, hide_index=True, use_container_width=True)


# ----------------- PAGE 3: DIET CHART -----------------
elif page == "🥗 Diet Chart":
    st.markdown('<h1 class="main-title">🥗 Tailored Diet and Nutrition Guidance</h1>', unsafe_allow_html=True)
    
    # Check if a user has completed an analysis during this session
    if 'latest_analysis' in st.session_state:
        analysis = st.session_state['latest_analysis']
        st.markdown("<p class='subtitle'>Nutrition chart customized based on the patient's latest analyzed results.</p>", unsafe_allow_html=True)
        
        # Read the values
        glucose = analysis['glucose']
        bp = analysis['blood_pressure']
        chol = analysis['cholesterol']
        bmi = analysis['bmi']
        risk_level = analysis['risk_level']
        
        st.markdown("### 🔎 Latest Patient Health Profile Summary")
        prof_cols = st.columns(4)
        prof_cols[0].metric("Glucose Level", f"{glucose} mg/dL")
        prof_cols[1].metric("Systolic BP", f"{bp} mmHg")
        prof_cols[2].metric("Cholesterol", f"{chol} mg/dL")
        prof_cols[3].metric("BMI Status", f"{bmi} ({'Overweight/Obese' if bmi > 25 else 'Healthy Range'})")
        
        st.markdown("---")
        
        # Determine customized diets
        st.markdown("## 🍽️ Your Personalized Diet Guide")
        
        # 1. Diabetic / Glucose Regulation Diet (if Glucose > 130)
        if glucose > 130:
            with st.expander("🍭 Glucose Regulation Meal Chart (High Fasting Glucose Alert)", expanded=True):
                col_b, col_l, col_d = st.columns(3)
                with col_b:
                    st.markdown("""
                    <div class="diet-box diet-include">
                        <h4>🍳 Breakfast</h4>
                        <ul>
                            <li><b>Oatmeal:</b> Sugar-free oatmeal topped with chia seeds and raw almonds.</li>
                            <li><b>Veggie Omelette:</b> Omelette made with 2 egg whites, spinach, and mushrooms.</li>
                            <li><b>Greek Yogurt:</b> Unsweetened Greek yogurt mixed with a handful of fresh blueberries.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                with col_l:
                    st.markdown("""
                    <div class="diet-box diet-include">
                        <h4>🍛 Lunch</h4>
                        <ul>
                            <li><b>Grilled Chicken/Paneer Salad:</b> Served on spinach, cucumbers, and cherry tomatoes.</li>
                            <li><b>Quinoa Bowl:</b> Quinoa cooked with broccoli, bell peppers, and boiled chickpeas.</li>
                            <li><b>Lentil Salad:</b> Sprouted lentils tossed with onions, cucumber, and lemon juice.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                with col_d:
                    st.markdown("""
                    <div class="diet-box diet-include">
                        <h4>🍲 Dinner</h4>
                        <ul>
                            <li><b>Baked Fish/Tofu:</b> Served with sautéed cauliflower, asparagus, and green beans.</li>
                            <li><b>Lentil Soup (Dal):</b> A bowl of yellow/black dal with a side of stir-fried zucchini.</li>
                            <li><b>Stir-fried Greens:</b> Cabbage, bell peppers, and paneer chunks lightly tossed in olive oil.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
        # 2. BP / DASH Diet (if Blood Pressure > 135)
        if bp > 135:
            with st.expander("🧂 Sodium Restriction & Heart Health Meal Chart (High BP Alert)", expanded=True):
                col_b, col_l, col_d = st.columns(3)
                with col_b:
                    st.markdown("""
                    <div class="diet-box diet-include">
                        <h4>🍳 Breakfast</h4>
                        <ul>
                            <li><b>Banana Oatmeal:</b> Oats cooked in low-fat milk, topped with sliced bananas.</li>
                            <li><b>Avocado Toast:</b> Whole-wheat toast topped with mashed avocado (no salt).</li>
                            <li><b>Smoothie Bowl:</b> Spinach, banana, and almond milk topped with unsalted seeds.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                with col_l:
                    st.markdown("""
                    <div class="diet-box diet-include">
                        <h4>🍛 Lunch</h4>
                        <ul>
                            <li><b>Mixed Bean Salad:</b> Kidney beans, chickpeas, onions, and cilantro with lemon dress.</li>
                            <li><b>Paneer Veggie Wrap:</b> Grilled paneer with shredded lettuce in a whole-wheat wrap.</li>
                            <li><b>Tomato Soup:</b> Unsalted fresh tomato soup with a side of boiled vegetables.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                with col_d:
                    st.markdown("""
                    <div class="diet-box diet-include">
                        <h4>🍲 Dinner</h4>
                        <ul>
                            <li><b>Brown Rice & Dal:</b> Brown rice served with low-sodium lentil dal and broccoli.</li>
                            <li><b>Baked Chicken Breast:</b> Seasoned with garlic and lemon (no salt) and steamed spinach.</li>
                            <li><b>Stir-fried Tofu:</b> Cooked with bell peppers, carrots, and mushrooms in minimal olive oil.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
        # 3. Cholesterol / Low Fat Diet (if Cholesterol > 200)
        if chol > 200:
            with st.expander("🥩 Cholesterol Control Meal Chart (High Cholesterol Alert)", expanded=True):
                col_b, col_l, col_d = st.columns(3)
                with col_b:
                    st.markdown("""
                    <div class="diet-box diet-include">
                        <h4>🍳 Breakfast</h4>
                        <ul>
                            <li><b>Oat Bran:</b> Cooked oatmeal sprinkled with ground flaxseeds and apple slices.</li>
                            <li><b>Berry Smoothie:</b> Made with soy/almond milk, mixed berries, and chia seeds.</li>
                            <li><b>Multigrain Toast:</b> Spread with peanut butter (unsweetened, no palm oil).</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                with col_l:
                    st.markdown("""
                    <div class="diet-box diet-include">
                        <h4>🍛 Lunch</h4>
                        <ul>
                            <li><b>Salmon/Tofu Salad:</b> Tossed with mixed greens, walnuts, and extra virgin olive oil.</li>
                            <li><b>Barley Soup:</b> Vegetable barley soup with a side of steamed spinach.</li>
                            <li><b>Chickpea Bowl:</b> Boiled chickpeas, cucumber, avocado, and bell peppers.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                with col_d:
                    st.markdown("""
                    <div class="diet-box diet-include">
                        <h4>🍲 Dinner</h4>
                        <ul>
                            <li><b>Grilled Mackerel/Sardines:</b> Served with roasted sweet potato and asparagus.</li>
                            <li><b>Dal Palak:</b> Lentils cooked with fresh spinach, served with a small cup of brown rice.</li>
                            <li><b>Vegetable Stir-fry:</b> Broccoli, beans, and sprouts stir-fried in healthy canola oil.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
        # 4. Weight Management (if BMI > 27)
        if bmi > 27:
            with st.expander("⚖️ BMI & Weight Management Meal Chart (Elevated BMI Alert)", expanded=True):
                col_b, col_l, col_d = st.columns(3)
                with col_b:
                    st.markdown("""
                    <div class="diet-box diet-include">
                        <h4>🍳 Breakfast</h4>
                        <ul>
                            <li><b>Boiled Eggs:</b> 2 hard-boiled eggs served with a cup of unsweetened green tea.</li>
                            <li><b>Moong Dal Sprouts:</b> Tossed with chopped cucumber, tomatoes, and lime juice.</li>
                            <li><b>Chia Seed Pudding:</b> Made with skimmed milk and flavored with vanilla/cinnamon.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                with col_l:
                    st.markdown("""
                    <div class="diet-box diet-include">
                        <h4>🍛 Lunch</h4>
                        <ul>
                            <li><b>Chapati & Sabzi:</b> 1 whole-wheat chapati, a bowl of mixed vegetable sabzi, and curd.</li>
                            <li><b>Grilled Tofu Salad:</b> High-protein salad with cucumbers, radish, and lettuce.</li>
                            <li><b>Boiled Chicken Breast:</b> Sliced and served with a side of steamed green peas.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                with col_d:
                    st.markdown("""
                    <div class="diet-box diet-include">
                        <h4>🍲 Dinner</h4>
                        <ul>
                            <li><b>Clear Vegetable Soup:</b> With boiled chicken breast or tofu chunks.</li>
                            <li><b>Paneer Tikka:</b> Grilled paneer chunks with bell peppers and a cup of buttermilk.</li>
                            <li><b>Sautéed Vegetables:</b> Broccoli, mushrooms, and zucchini cooked in minimal spray oil.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
        # 5. General Fitness Wellness (If patient is healthy and all markers are normal)
        if glucose <= 130 and bp <= 135 and chol <= 200 and bmi <= 27:
            st.success("🎉 All your vitals are in the normal range! Here is your daily wellness maintenance diet:")
            col_b, col_l, col_d = st.columns(3)
            with col_b:
                st.markdown("""
                <div class="diet-box diet-include">
                    <h4>🍳 Breakfast</h4>
                    <ul>
                        <li><b>Poha/Upma:</b> Vegetable poha or suji upma cooked with a handful of roasted peanuts.</li>
                        <li><b>Paneer Toast:</b> Multigrain toast topped with low-fat crumbled paneer.</li>
                        <li><b>Fresh Fruits:</b> A bowl of seasonal fresh fruits with warm milk.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            with col_l:
                st.markdown("""
                <div class="diet-box diet-include">
                    <h4>🍛 Lunch</h4>
                    <ul>
                        <li><b>Balanced Thali:</b> 2 multigrain chapatis, seasonal veg curry, dal, and a bowl of salad.</li>
                        <li><b>Chicken Pulao:</b> Brown rice chicken pulao with cucumber raita (yogurt).</li>
                        <li><b>Soy Bean Curry:</b> Served with a cup of steamed basmati rice and sliced cucumber.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            with col_d:
                st.markdown("""
                <div class="diet-box diet-include">
                    <h4>🍲 Dinner</h4>
                    <ul>
                        <li><b>Vegetable Khichdi:</b> Light dal-rice khichdi with a teaspoon of ghee.</li>
                        <li><b>Grilled Paneer/Chicken:</b> Served with steamed carrot, beans, and baby corn.</li>
                        <li><b>Minestrone Soup:</b> Hearty tomato vegetable soup with whole-wheat pasta shells.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
    else:
        # Fallback when no patient has been analyzed yet in the current session
        st.markdown("<p class='subtitle'>No patient analysis found for the current session. Go to <b>Analyse Health</b> first to get a personalized diet card, or read the general wellness chart below.</p>", unsafe_allow_html=True)
        
        st.markdown("### 🥦 General Balanced Wellness Diet Chart")
        
        col_b, col_l, col_d = st.columns(3)
        
        with col_b:
            st.markdown("""
            <div class="health-card status-healthy">
                <div class="section-title">🍳 Breakfast Suggestions</div>
                <ul>
                    <li><b>Oatmeal or Porridge:</b> Cooked with sliced apples and a handful of nuts.</li>
                    <li><b>Paneer/Egg Scramble:</b> Served with multi-grain toast.</li>
                    <li><b>Vegetable Poha:</b> Made with curry leaves, mustard seeds, and roasted peanuts.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col_l:
            st.markdown("""
            <div class="health-card status-healthy">
                <div class="section-title">🍛 Lunch Suggestions</div>
                <ul>
                    <li><b>Roti, Sabzi & Dal:</b> Balanced combination of wheat chapatis, mixed veg, and lentil dal.</li>
                    <li><b>Brown Rice Chicken Curry:</b> Served with a bowl of cooling cucumber raita.</li>
                    <li><b>Healthy Chickpea Salad:</b> Combined with bell peppers, cucumber, paneer, and lime.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col_d:
            st.markdown("""
            <div class="health-card status-healthy">
                <div class="section-title">🍲 Dinner Suggestions</div>
                <ul>
                    <li><b>Light Khichdi:</b> Easy-to-digest rice and yellow lentil soup-stew.</li>
                    <li><b>Grilled Fish or Paneer:</b> Seasoned with herbs and served with steamed vegetables.</li>
                    <li><b>Vegetable Soup:</b> Clear soup loaded with cabbage, spinach, and tofu/chicken chunks.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
