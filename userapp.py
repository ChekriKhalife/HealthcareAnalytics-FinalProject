import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Set page config
st.set_page_config(page_title="Stroke Risk Interactive Guide", layout="wide", initial_sidebar_state="expanded")

# Custom CSS to improve design
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .sidebar .sidebar-content {
        background: #ffffff
    }
    .Widget>label {
        color: #31333F;
        font-weight: bold;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #007bff;
        border-radius: 5px;
    }
    .stProgress .st-bo {
        background-color: #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_data' not in st.session_state:
    st.session_state.user_data = {
        'name': '',
        'age': 0,
        'gender': '',
        'height': 0,
        'weight': 0,
        'bmi': 0,
        'activity_level': '',
        'diet_habits': {},
        'smoking_status': '',
        'blood_pressure': {},
        'cholesterol': {},
        'glucose': 0,
    }

def main():
    st.title("🩺 Stroke Risk Interactive Guide")
    st.write("Explore, learn, and take action to reduce your stroke risk factors!")

    # Sidebar for user info and risk factor selection
    with st.sidebar:
        st.header("👤 Your Profile")
        if st.session_state.user_data['name'] == '':
            st.session_state.user_data['name'] = st.text_input("Enter your name:")
        st.write(f"Welcome, {st.session_state.user_data['name']}!")
        
        if st.session_state.user_data['age'] == 0:
            st.session_state.user_data['age'] = st.slider("Your age:", 18, 100, 30)
        if st.session_state.user_data['gender'] == '':
            st.session_state.user_data['gender'] = st.selectbox("Your gender:", ['Male', 'Female', 'Other'])
        
        st.header("🎯 Risk Factors")
        risk_factor = st.radio(
            "Choose a risk factor to explore:",
            ["Body Mass Index (BMI)", "Diet", "Sodium Intake", "Cholesterol",
             "Blood Glucose", "Blood Pressure", "Physical Activity",
             "Air Pollution", "Smoking"]
        )

    # Main content
    if risk_factor == "Body Mass Index (BMI)":
        bmi_campaign()
    elif risk_factor == "Diet":
        diet_campaign()
    elif risk_factor == "Sodium Intake":
        sodium_campaign()
    elif risk_factor == "Cholesterol":
        cholesterol_campaign()
    elif risk_factor == "Blood Glucose":
        glucose_campaign()
    elif risk_factor == "Blood Pressure":
        blood_pressure_campaign()
    elif risk_factor == "Physical Activity":
        physical_activity_campaign()
    elif risk_factor == "Air Pollution":
        pollution_campaign()
    elif risk_factor == "Smoking":
        smoking_campaign()

def bmi_campaign():
    st.header("🏋 Body Mass Index (BMI) Explorer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Your BMI Calculator")
        height = st.number_input("Enter your height (cm):", min_value=100, max_value=250, value=170)
        weight = st.number_input("Enter your weight (kg):", min_value=30, max_value=300, value=70)
        
        if st.button("Calculate BMI"):
            bmi = weight / ((height/100)**2)
            st.session_state.user_data['bmi'] = bmi
            st.session_state.user_data['height'] = height
            st.session_state.user_data['weight'] = weight
            
            st.metric("Your BMI", f"{bmi:.1f}")
            
            if bmi < 18.5:
                st.warning("You are underweight. Focus on gaining healthy weight.")
            elif 18.5 <= bmi < 25:
                st.success("You have a healthy BMI. Keep up the good work!")
            elif 25 <= bmi < 30:
                st.warning("You are overweight. Consider lifestyle changes to reduce your BMI.")
            else:
                st.error("You are in the obese range. Please consult with a healthcare professional.")
    
    with col2:
        st.subheader("BMI Distribution")
        bmi_data = pd.DataFrame({
            'BMI': np.random.normal(26, 5, 1000)
        })
        fig = px.histogram(bmi_data, x='BMI', nbins=30,
                           title='BMI Distribution in Population',
                           labels={'BMI': 'Body Mass Index', 'count': 'Number of People'},
                           color_discrete_sequence=['#3366cc'])
        fig.add_vline(x=st.session_state.user_data['bmi'], line_dash="dash", line_color="red", annotation_text="Your BMI")
        st.plotly_chart(fig)

    st.subheader("🎯 Personalized BMI Action Plan")
    if st.session_state.user_data['bmi'] > 25:
        st.write("Here's a customized plan to help you achieve a healthier BMI:")
        weeks_to_goal = int((st.session_state.user_data['bmi'] - 24.9) / 0.5)
        goal_weight = 24.9 * ((st.session_state.user_data['height']/100)**2)
        weight_to_lose = st.session_state.user_data['weight'] - goal_weight
        
        st.write(f"1. Your target weight: {goal_weight:.1f} kg")
        st.write(f"2. Weight to lose: {weight_to_lose:.1f} kg")
        st.write(f"3. Estimated time to reach goal: {weeks_to_goal} weeks")
        st.write("4. Weekly goals:")
        st.write("   - Aim to lose 0.5-1 kg per week")
        st.write("   - 150 minutes of moderate aerobic activity or 75 minutes of vigorous aerobic activity")
        st.write("   - Strength training exercises at least 2 days a week")
        st.write("5. Dietary recommendations:")
        st.write("   - Increase intake of fruits, vegetables, and whole grains")
        st.write("   - Choose lean proteins and low-fat dairy products")
        st.write("   - Limit processed foods, sugary drinks, and high-calorie snacks")
        
        if st.button("Generate Workout Plan"):
            st.write("Your personalized workout plan:")
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            exercises = ["Brisk walking", "Cycling", "Swimming", "Strength training", "Yoga", "HIIT workout", "Rest day"]
            workout_plan = pd.DataFrame({
                "Day": days,
                "Exercise": random.sample(exercises, 7),
                "Duration (minutes)": [45, 30, 45, 30, 45, 30, 0]
            })
            st.table(workout_plan)
    else:
        st.write("Great job maintaining a healthy BMI! Here are some tips to stay healthy:")
        st.write("1. Continue with regular physical activity")
        st.write("2. Maintain a balanced diet rich in fruits, vegetables, and whole grains")
        st.write("3. Stay hydrated")
        st.write("4. Get regular health check-ups")
        st.write("5. Manage stress through relaxation techniques or hobbies")

def diet_campaign():
    st.header("🥗 Dietary Habits Explorer")
    
    st.subheader("Your Dietary Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fruit_veg = st.slider("How many servings of fruits and vegetables do you eat daily?", 0, 10, 3)
        whole_grains = st.slider("How many servings of whole grains do you eat daily?", 0, 10, 2)
        processed_meat = st.slider("How many servings of processed meat do you eat weekly?", 0, 21, 3)
        sugary_drinks = st.slider("How many sugary drinks do you consume weekly?", 0, 21, 3)
    
    with col2:
        st.write("Recommended daily intake:")
        st.write("- Fruits and vegetables: 5+ servings")
        st.write("- Whole grains: 3+ servings")
        st.write("- Processed meat: Less than 2 servings per week")
        st.write("- Sugary drinks: Less than 3 per week")
    
    diet_score = (fruit_veg / 5 * 30) + (whole_grains / 3 * 20) + ((7 - processed_meat) / 7 * 25) + ((7 - sugary_drinks) / 7 * 25)
    
    st.subheader("Your Diet Quality Score")
    st.progress(diet_score / 100)
    st.write(f"Your score: {diet_score:.1f}/100")
    
    if diet_score < 50:
        st.error("Your diet quality needs significant improvement.")
    elif 50 <= diet_score < 70:
        st.warning("Your diet quality is moderate. There's room for improvement.")
    else:
        st.success("Great job! You have a high-quality diet.")
    
    st.subheader("Personalized Dietary Recommendations")
    if fruit_veg < 5:
        st.write("🍎 Increase your fruit and vegetable intake:")
        st.write("- Add a serving of fruit to your breakfast")
        st.write("- Include a salad with lunch and dinner")
        st.write("- Keep cut vegetables for easy snacking")
    
    if whole_grains < 3:
        st.write("🌾 Boost your whole grain consumption:")
        st.write("- Switch to whole grain bread and pasta")
        st.write("- Try quinoa or brown rice as a side dish")
        st.write("- Start your day with oatmeal")
    
    if processed_meat > 2:
        st.write("🥩 Reduce processed meat intake:")
        st.write("- Choose fresh, lean meats instead of processed options")
        st.write("- Try plant-based protein sources like beans or tofu")
        st.write("- Limit processed meat to special occasions")
    
    if sugary_drinks > 3:
        st.write("🥤 Cut down on sugary drinks:")
        st.write("- Swap sodas for water or unsweetened tea")
        st.write("- If you crave something sweet, try infused water with fruits")
        st.write("- Gradually reduce intake to make the change sustainable")
    
    st.subheader("Meal Planner")
    if st.button("Generate a Healthy Meal Plan"):
        meals = {
            "Breakfast": ["Oatmeal with berries", "Whole grain toast with avocado", "Greek yogurt with nuts and fruits", "Vegetable omelette", "Smoothie bowl"],
            "Lunch": ["Grilled chicken salad", "Quinoa and vegetable bowl", "Lentil soup with whole grain bread", "Tuna sandwich on whole wheat", "Vegetable stir-fry with brown rice"],
            "Dinner": ["Baked salmon with roasted vegetables", "Turkey chili with mixed beans", "Grilled tofu with quinoa and broccoli", "Whole wheat pasta with tomato sauce and vegetables", "Lean beef stir-fry with brown rice"],
            "Snack": ["Apple slices with peanut butter", "Carrot sticks with hummus", "Mixed nuts", "Greek yogurt with berries", "Whole grain crackers with cheese"]
        }
        
        meal_plan = pd.DataFrame({
            "Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            "Breakfast": random.choices(meals["Breakfast"], k=7),
            "Lunch": random.choices(meals["Lunch"], k=7),
            "Dinner": random.choices(meals["Dinner"], k=7),
            "Snack": random.choices(meals["Snack"], k=7)
        })
        
        st.table(meal_plan)

def sodium_campaign():
    st.header("🧂 Sodium Intake Awareness")
    
    st.subheader("Estimate Your Daily Sodium Intake")
    
    foods = {
        "Bread (1 slice)": 150,
        "Cheese (1 oz)": 175,
        "Ham (3 oz)": 1000,
        "Pickle (1 medium)": 800,
        "Tomato soup (1 cup)": 700,
        "Pizza (1 slice)": 600,
        "Salad dressing (2 tbsp)": 300,
        "Potato chips (1 oz)": 150,
        "Canned vegetables (1/2 cup)": 200,
        "Restaurant meal": 2000
    }
    
    user_intake = {}
    total_sodium = 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        for food, sodium in list(foods.items())[:5]:
            amount = st.number_input(f"How many servings of {food} do you consume daily?", min_value=0.0, max_value=10.0, step=0.5)
            user_intake[food] = amount
            total_sodium += amount * sodium
    
    with col2:
        for food, sodium in list(foods.items())[5:]:
            amount = st.number_input(f"How many servings of {food} do you consume daily?", min_value=0.0, max_value=10.0, step=0.5)
            user_intake[food] = amount
            total_sodium += amount * sodium
    
    st.subheader("Your Estimated Daily Sodium Intake")
    st.metric("Total Sodium", f"{total_sodium:.0f} mg")
    
    recommended = 2300  # mg, recommended daily limit
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = total_sodium,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Daily Sodium Intake (mg)"},
        gauge = {
            'axis': {'range': [None, 5000], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [{'range': [0, 2300], 'color': "lightgreen"},
                {'range': [2300, 3500], 'color': "yellow"},
                {'range': [3500, 5000], 'color': "red"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 2300}}))
    
    st.plotly_chart(fig)
    
    if total_sodium > recommended:
        st.warning(f"Your sodium intake is {total_sodium - recommended:.0f} mg above the recommended daily limit of 2300 mg.")
        st.subheader("Tips to Reduce Sodium Intake")
        st.write("1. Choose fresh fruits and vegetables over canned ones.")
        st.write("2. Look for low-sodium versions of your favorite foods.")
        st.write("3. Use herbs and spices instead of salt to flavor your food.")
        st.write("4. Rinse canned foods to remove excess sodium.")
        st.write("5. Cook more meals at home to control sodium content.")
    else:
        st.success("Great job! Your sodium intake is within the recommended limit.")
    
    st.subheader("Sodium Content Comparison")
    selected_foods = st.multiselect("Select foods to compare sodium content:", list(foods.keys()))
    if selected_foods:
        comparison_data = {food: foods[food] for food in selected_foods}
        fig = px.bar(x=list(comparison_data.keys()), y=list(comparison_data.values()),
                     labels={'x': 'Food', 'y': 'Sodium Content (mg)'},
                     title='Sodium Content Comparison')
        st.plotly_chart(fig)

def cholesterol_campaign():
    st.header("🫀 Cholesterol Management")
    
    st.subheader("Your Cholesterol Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        total_cholesterol = st.number_input("Total Cholesterol (mg/dL):", min_value=100, max_value=300, value=180)
        ldl = st.number_input("LDL Cholesterol (mg/dL):", min_value=30, max_value=200, value=100)
        hdl = st.number_input("HDL Cholesterol (mg/dL):", min_value=20, max_value=100, value=50)
    
    with col2:
        st.write("Healthy Cholesterol Levels:")
        st.write("- Total Cholesterol: Less than 200 mg/dL")
        st.write("- LDL Cholesterol: Less than 100 mg/dL")
        st.write("- HDL Cholesterol: 60 mg/dL or higher")
    
    st.subheader("Cholesterol Risk Assessment")
    
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode = "number+gauge+delta",
        value = total_cholesterol,
        domain = {'x': [0, 0.32], 'y': [0, 1]},
        title = {'text': "Total Cholesterol"},
        delta = {'reference': 200, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 300], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 200], 'color': "lightgreen"},
                {'range': [200, 240], 'color': "yellow"},
                {'range': [240, 300], 'color': "red"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 200}}))
    
    fig.add_trace(go.Indicator(
        mode = "number+gauge+delta",
        value = ldl,
        domain = {'x': [0.34, 0.66], 'y': [0, 1]},
        title = {'text': "LDL Cholesterol"},
        delta = {'reference': 100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 200], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 100], 'color': "lightgreen"},
                {'range': [100, 160], 'color': "yellow"},
                {'range': [160, 200], 'color': "red"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 100}}))
    
    fig.add_trace(go.Indicator(
        mode = "number+gauge+delta",
        value = hdl,
        domain = {'x': [0.68, 1], 'y': [0, 1]},
        title = {'text': "HDL Cholesterol"},
        delta = {'reference': 60, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': "red"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "lightgreen"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60}}))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig)
    
    st.subheader("Personalized Cholesterol Management Plan")
    if total_cholesterol > 200 or ldl > 100 or hdl < 60:
        st.write("Here are some steps to improve your cholesterol levels:")
        st.write("1. Increase physical activity: Aim for at least 30 minutes of moderate exercise most days.")
        st.write("2. Improve your diet:")
        st.write("   - Reduce saturated and trans fats")
        st.write("   - Eat more fiber-rich foods like fruits, vegetables, and whole grains")
        st.write("   - Include foods rich in omega-3 fatty acids, like fish, nuts, and seeds")
        st.write("3. Maintain a healthy weight")
        st.write("4. Quit smoking if you smoke")
        st.write("5. Limit alcohol consumption")
        
        if st.button("Generate a Cholesterol-Friendly Meal Plan"):
            meals = {
                "Breakfast": ["Oatmeal with berries and nuts", "Whole grain toast with avocado and egg whites", "Greek yogurt parfait with fruits and granola", "Smoothie bowl with chia seeds", "Whole grain cereal with low-fat milk and banana"],
                "Lunch": ["Grilled chicken salad with olive oil dressing", "Lentil soup with whole grain bread", "Tuna sandwich on whole wheat with carrot sticks", "Quinoa bowl with mixed vegetables and grilled tofu", "Turkey and avocado wrap with apple slices"],
                "Dinner": ["Baked salmon with roasted vegetables", "Grilled lean steak with sweet potato and broccoli", "Stir-fried tofu and vegetables with brown rice", "Grilled chicken breast with quinoa and asparagus", "Vegetable and bean chili with corn bread"],
                "Snack": ["Apple slices with almond butter", "Carrot sticks with hummus", "Mixed nuts (unsalted)", "Greek yogurt with berries", "Edamame"]
            }
            
            meal_plan = pd.DataFrame({
                "Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                "Breakfast": random.choices(meals["Breakfast"], k=7),
                "Lunch": random.choices(meals["Lunch"], k=7),
                "Dinner": random.choices(meals["Dinner"], k=7),
                "Snack": random.choices(meals["Snack"], k=7)
            })
            
            st.table(meal_plan)
    else:
        st.success("Great job! Your cholesterol levels are within the healthy range. Keep up the good work!")

def glucose_campaign():
    st.header("🍬 Blood Glucose Management")
    
    st.subheader("Your Blood Glucose Profile")
    
    fasting_glucose = st.number_input("Fasting Blood Glucose (mg/dL):", min_value=70, max_value=300, value=100)
    hba1c = st.number_input("HbA1c (%):", min_value=4.0, max_value=14.0, value=5.7, step=0.1)
    
    st.write("Healthy Blood Glucose Levels:")
    st.write("- Fasting Blood Glucose: Less than 100 mg/dL")
    st.write("- HbA1c: Less than 5.7%")
    
    st.subheader("Blood Glucose Risk Assessment")
    
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode = "number+gauge+delta",
        value = fasting_glucose,
        domain = {'x': [0, 0.5], 'y': [0, 1]},
        title = {'text': "Fasting Blood Glucose"},
        delta = {'reference': 100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 200], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 100], 'color': "lightgreen"},
                {'range': [100, 125], 'color': "yellow"},
                {'range': [125, 200], 'color': "red"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 100}}))
    
    fig.add_trace(go.Indicator(
        mode = "number+gauge+delta",
        value = hba1c,
        domain = {'x': [0.5, 1], 'y': [0, 1]},
        title = {'text': "HbA1c"},
        delta = {'reference': 5.7, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 14], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 5.7], 'color': "lightgreen"},
                {'range': [5.7, 6.4], 'color': "yellow"},
                {'range': [6.4, 14], 'color': "red"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 5.7}}))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig)
    
    st.subheader("Personalized Blood Glucose Management Plan")
    if fasting_glucose >= 100 or hba1c >= 5.7:
        st.write("Here are some steps to improve your blood glucose levels:")
        st.write("1. Monitor your carbohydrate intake")
        st.write("2. Choose foods with a low glycemic index")
        st.write("3. Increase fiber intake")
        st.write("4. Stay hydrated")
        st.write("5. Exercise regularly")
        st.write("6. Manage stress")
        st.write("7. Get adequate sleep")
        
        if st.button("Generate a Glucose-Friendly Meal Plan"):
            meals = {
                "Breakfast": ["Scrambled eggs with spinach and whole grain toast", "Greek yogurt with berries and chia seeds", "Overnight oats with cinnamon and apple", "Whole grain pancakes with almond butter", "Veggie and egg white omelette"],
                "Lunch": ["Grilled chicken salad with mixed greens and vinaigrette", "Lentil soup with a side salad", "Turkey and avocado wrap on whole wheat", "Quinoa bowl with grilled vegetables and chickpeas", "Tuna salad with whole grain crackers"],
                "Dinner": ["Baked salmon with roasted Brussels sprouts", "Grilled lean steak with sweet potato and broccoli", "Stir-fried tofu and vegetables with brown rice", "Grilled chicken breast with quinoa and asparagus", "Vegetable and bean chili"],
                "Snack": ["Apple slices with peanut butter", "Carrot sticks with hummus", "Mixed nuts (unsalted)", "Greek yogurt with cinnamon", "Celery with almond butter"]
            }
            
            meal_plan = pd.DataFrame({
                "Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                "Breakfast": random.choices(meals["Breakfast"], k=7),
                "Lunch": random.choices(meals["Lunch"], k=7),
                "Dinner": random.choices(meals["Dinner"], k=7),
                "Snack": random.choices(meals["Snack"], k=7)
            })
            
            st.table(meal_plan)
    else:
        st.success("Great job! Your blood glucose levels are within the healthy range. Keep up the good work!")

def blood_pressure_campaign():
    st.header("🩸 Blood Pressure Management")
    
    st.subheader("Your Blood Pressure Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        systolic = st.number_input("Systolic Blood Pressure (mmHg):", min_value=70, max_value=220, value=120)
        diastolic = st.number_input("Diastolic Blood Pressure (mmHg):", min_value=40, max_value=120, value=80)
    
    with col2:
        st.write("Blood Pressure Categories:")
        st.write("- Normal: Less than 120/80 mmHg")
        st.write("- Elevated: 120-129/<80 mmHg")
        st.write("- Hypertension Stage 1: 130-139/80-89 mmHg")
        st.write("- Hypertension Stage 2: 140+/90 + mmHg")
    
    st.subheader("Blood Pressure Risk Assessment")
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = systolic,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Blood Pressure: {systolic}/{diastolic} mmHg"},
        delta = {'reference': 120, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 220], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 120], 'color': "lightgreen"},
                {'range': [120, 130], 'color': "yellow"},
                {'range': [130, 180], 'color': "orange"},
                {'range': [180, 220], 'color': "red"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 140}}))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig)
    
    if systolic < 120 and diastolic < 80:
        st.success("Your blood pressure is in the normal range. Great job!")
    elif 120 <= systolic <= 129 and diastolic < 80:
        st.warning("Your blood pressure is elevated. Consider lifestyle changes to lower it.")
    elif 130 <= systolic <= 139 or 80 <= diastolic <= 89:
        st.warning("Your blood pressure is in the Hypertension Stage 1 range. Consult with your doctor about lifestyle changes or medication.")
    elif systolic >= 140 or diastolic >= 90:
        st.error("Your blood pressure is in the Hypertension Stage 2 range. Consult with your doctor immediately about treatment options.")
    
    st.subheader("Personalized Blood Pressure Management Plan")
    st.write("Here are some steps to improve or maintain healthy blood pressure:")
    st.write("1. Maintain a healthy weight")
    st.write("2. Exercise regularly (at least 150 minutes of moderate activity per week)")
    st.write("3. Reduce sodium intake (aim for less than 2,300 mg per day)")
    st.write("4. Limit alcohol consumption")
    st.write("5. Quit smoking if you smoke")
    st.write("6. Manage stress through relaxation techniques or meditation")
    st.write("7. Monitor your blood pressure regularly at home")
    
    if st.button("Generate a Blood Pressure-Friendly Meal Plan"):
        meals = {
            "Breakfast": ["Oatmeal with berries and flaxseeds", "Whole grain toast with avocado and egg whites", "Greek yogurt parfait with fruits and unsalted nuts", "Smoothie bowl with spinach and chia seeds", "Whole grain cereal with low-fat milk and banana"],
            "Lunch": ["Grilled chicken salad with olive oil and lemon dressing", "Lentil soup with whole grain bread", "Tuna sandwich on whole wheat with carrot sticks", "Quinoa bowl with mixed vegetables and grilled tofu", "Turkey and avocado wrap with apple slices"],
            "Dinner": ["Baked salmon with roasted vegetables", "Grilled lean steak with sweet potato and steamed broccoli", "Stir-fried tofu and vegetables with brown rice", "Grilled chicken breast with quinoa and asparagus", "Vegetable and bean chili with corn bread"],
            "Snack": ["Apple slices with unsalted almond butter", "Carrot sticks with homemade hummus", "Mixed unsalted nuts", "Greek yogurt with berries", "Celery sticks with low-fat cream cheese"]
        }
        
        meal_plan = pd.DataFrame({
            "Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            "Breakfast": random.choices(meals["Breakfast"], k=7),
            "Lunch": random.choices(meals["Lunch"], k=7),
            "Dinner": random.choices(meals["Dinner"], k=7),
            "Snack": random.choices(meals["Snack"], k=7)
        })
        
        st.table(meal_plan)

def physical_activity_campaign():
    st.header("🏃‍♂ Physical Activity Tracker")
    
    st.subheader("Your Activity Profile")
    
    activity_days = st.slider("How many days per week do you engage in moderate to vigorous physical activity?", 0, 7, 3)
    activity_minutes = st.number_input("On average, how many minutes do you spend on physical activity on those days?", 0, 180, 30)
    
    weekly_minutes = activity_days * activity_minutes
    
    st.write(f"Your total weekly physical activity: {weekly_minutes} minutes")
    
    recommended_minutes = 150
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = weekly_minutes,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Weekly Physical Activity (minutes)"},
        delta = {'reference': recommended_minutes, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 300], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 150], 'color': "lightcoral"},
                {'range': [150, 300], 'color': "lightgreen"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 150}}))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig)
    
    if weekly_minutes < recommended_minutes:
        st.warning(f"You're {recommended_minutes - weekly_minutes} minutes short of the recommended 150 minutes of moderate-intensity aerobic activity per week.")
    else:
        st.success(f"Great job! You're meeting or exceeding the recommended 150 minutes of moderate-intensity aerobic activity per week.")
    
    st.subheader("Personalized Activity Plan")
    st.write("Here's a suggested weekly activity plan to help you reach or maintain the recommended level of physical activity:")
    
    activities = [
        "Brisk walking", "Jogging", "Swimming", "Cycling", "Dancing",
        "Yoga", "Strength training", "Tennis", "Basketball", "Hiking"
    ]
    
    plan = []
    remaining_minutes = max(recommended_minutes - weekly_minutes, 0)
    days = list(range(1, 8))
    random.shuffle(days)
    
    for day in days:
        if remaining_minutes > 0:
            duration = min(random.randint(20, 60), remaining_minutes)
            activity = random.choice(activities)
            plan.append({"Day": f"Day {day}", "Activity": activity, "Duration (minutes)": duration})
            remaining_minutes -= duration
    
    if plan:
        plan_df = pd.DataFrame(plan)
        st.table(plan_df)
    else:
        st.write("You're already meeting the recommended level of physical activity. Keep up the good work!")
    
    st.write("Remember to:")
    st.write("1. Start slowly and gradually increase your activity level")
    st.write("2. Choose activities you enjoy to stay motivated")
    st.write("3. Mix cardio exercises with strength training for optimal health benefits")
    st.write("4. Stay hydrated and listen to your body")
    st.write("5. Consult with your doctor before starting a new exercise program, especially if you have any health concerns")

def pollution_campaign():
    st.header("🌬 Air Pollution Awareness")
    
    st.subheader("Local Air Quality Index (AQI)")
    
    aqi = st.slider("What's the current Air Quality Index (AQI) in your area?", 0, 500, 50)
    
    st.write("AQI Categories:")
    st.write("- 0-50: Good")
    st.write("- 51-100: Moderate")
    st.write("- 101-150: Unhealthy for Sensitive Groups")
    st.write("- 151-200: Unhealthy")
    st.write("- 201-300: Very Unhealthy")
    st.write("- 301-500: Hazardous")
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = aqi,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Air Quality Index (AQI)"},
        gauge = {
            'axis': {'range': [None, 500], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [51, 100], 'color': "yellow"},
                {'range': [101, 150], 'color': "orange"},
                {'range': [151, 200], 'color': "red"},
                {'range': [201, 300], 'color': "purple"},
                {'range': [301, 500], 'color': "maroon"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 301}}))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig)
    
    st.subheader("Health Recommendations Based on AQI")
    if aqi <= 50:
        st.success("Air quality is good. It's a great day for outdoor activities!")
    elif 51 <= aqi <= 100:
        st.info("Air quality is moderate. Unusually sensitive people should consider reducing prolonged or heavy exertion.")
    elif 101 <= aqi <= 150:
        st.warning("Air quality is unhealthy for sensitive groups. They should reduce prolonged or heavy exertion.")
    elif 151 <= aqi <= 200:
        st.error("Air quality is unhealthy. Everyone should reduce prolonged or heavy exertion.")
    elif 201 <= aqi <= 300:
        st.error("Air quality is very unhealthy. Everyone should avoid prolonged or heavy exertion.")
    else:
        st.error("Air quality is hazardous. Everyone should avoid all physical activity outdoors.")
    
    st.subheader("Tips to Reduce Exposure to Air Pollution")
    st.write("1. Check daily air quality forecasts in your area")
    st.write("2. Avoid exercising outdoors when pollution levels are high")
    st.write("3. If you must go outside, try to go out in the morning when ozone levels are lower")
    st.write("4. Avoid exercising near high-traffic areas")
    st.write("5. Use air filters and purifiers indoors")
    st.write("6. Keep windows closed on high pollution days")
    st.write("7. Wear a mask (N95 or P100) when pollution levels are high")
    
    st.subheader("How You Can Help Reduce Air Pollution")
    st.write("1. Use public transportation, carpool, bike, or walk instead of driving")
    st.write("2. Reduce electricity usage at home")
    st.write("3. Don't burn leaves, trash, or other materials")
    st.write("4. Use environmentally safe paints and cleaning products")
    st.write("5. Plant trees and increase green spaces in your community")

def smoking_campaign():
    st.header("🚭 Smoking Cessation Journey")
    
    smoking_status = st.radio("What's your current smoking status?", ["Non-smoker", "Former smoker", "Current smoker"])
    
    if smoking_status == "Current smoker":
        cigarettes_per_day = st.number_input("How many cigarettes do you smoke per day?", 1, 100, 10)
        years_smoking = st.number_input("For how many years have you been smoking?", 1, 70, 5)
        
        pack_years = (cigarettes_per_day / 20) * years_smoking
        
        st.write(f"Your pack-year history: {pack_years:.1f} pack-years")
        
        st.subheader("Health Risks Based on Pack-Years")
        if pack_years < 10:
            st.warning("Your risk of smoking-related diseases is elevated. Quitting now can significantly reduce your risk.")
        elif 10 <= pack_years < 20:
            st.warning("Your risk of smoking-related diseases is high. Quitting now is crucial for your health.")
        else:
            st.error("Your risk of smoking-related diseases is very high. Immediate smoking cessation is strongly advised.")
        
        st.subheader("Benefits of Quitting Smoking")
        benefits = {
            "20 minutes": "Heart rate and blood pressure drop",
            "12 hours": "Carbon monoxide level in blood drops to normal",
            "2-12 weeks": "Circulation improves and lung function increases",
            "1-9 months": "Coughing and shortness of breath decrease",
            "1 year": "Risk of coronary heart disease is half that of a smoker",
            "5 years": "Stroke risk reduced to that of a non-smoker",
            "10 years": "Lung cancer death rate is about half that of a smoker",
            "15 years": "Risk of coronary heart disease is back to that of a non-smoker"
        }
        
        for time, benefit in benefits.items():
            st.write(f"*After {time}:* {benefit}")
        
        st.subheader("Quitting Plan")
        quit_date = st.date_input("Set a quit date")
        
        st.write("Steps to prepare for your quit date:")
        st.write("1. Inform friends and family about your decision to quit")
        st.write("2. Remove all cigarettes and ashtrays from your home, car, and workplace")
        st.write("3. Stock up on oral substitutes - sugarless gum, carrot sticks, hard candy")
        st.write("4. Set up a support system - a group program or a friend to call")
        st.write("5. Ask your healthcare provider about nicotine replacement therapy or other medications")
        
        st.write("On your quit day:")
        st.write("1. Do not smoke at all")
        st.write("2. Stay busy - try to distract yourself from urges to smoke")
        st.write("3. Drink lots of water and other fluids")
        st.write("4. Start using nicotine replacement if that's your choice")
        st.write("5. Attend a stop-smoking group or follow a self-help plan")
        st.write("6. Avoid situations where the urge to smoke is strong")
        st.write("7. Avoid people who are smoking")
        st.write("8. Exercise to relieve stress and improve mood")
        
        if st.button("Calculate Potential Savings"):
            cigarette_cost = 0.50  # Assume $0.50 per cigarette
            daily_cost = cigarettes_per_day * cigarette_cost
            yearly_cost = daily_cost * 365
            
            st.write(f"By quitting smoking, you could save approximately:")
            st.write(f"- ${daily_cost:.2f} per day")
            st.write(f"- ${yearly_cost:.2f} per year")
            st.write(f"- ${yearly_cost * 5:.2f} over 5 years")
            
            fig = go.Figure(go.Indicator(
                mode = "number+delta",
                value = yearly_cost * 5,
                number = {'prefix': "$"},
                delta = {'position': "top", 'reference': 0},
                domain = {'x': [0, 1], 'y': [0, 1]}
            ))
            
            fig.update_layout(
                title={
                    'text': "Potential 5-Year Savings",
                    'y':0.9,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                height=300
            )
            
            st.plotly_chart(fig)
        
    elif smoking_status == "Former smoker":
        st.success("Congratulations on quitting smoking! Your health has already started to improve.")
        years_quit = st.number_input("How many years ago did you quit smoking?", 0, 50, 1)
        
        st.subheader("Health Improvements Since Quitting")
        if years_quit < 1:
            st.write("- Your heart rate and blood pressure have dropped")
            st.write("- The carbon monoxide level in your blood has dropped to normal")
            st.write("- Your circulation has improved and your lung function has increased")
        elif 1 <= years_quit < 5:
            st.write("- Your risk of coronary heart disease is now half that of a smoker")
            st.write("- Your lung function has significantly improved")
            st.write("- Your risk of stroke has greatly decreased")
        elif 5 <= years_quit < 10:
            st.write("- Your risk of stroke has reduced to that of a non-smoker")
            st.write("- Your risk of cancers of the mouth, throat, and esophagus has halved")
        else:
            st.write("- Your risk of lung cancer has dropped to about half that of a smoker")
            st.write("- Your risk of coronary heart disease is now similar to that of a non-smoker")
        
        st.write("Keep up the great work! Your body thanks you for quitting smoking.")
        
    else:  # Non-smoker
        st.success("Great job on being a non-smoker! You're avoiding numerous health risks associated with smoking.")
        
        st.subheader("Benefits of Remaining Smoke-Free")
        st.write("1. Lower risk of lung cancer and many other types of cancer")
        st.write("2. Reduced risk of heart disease and stroke")
        st.write("3. Lower risk of lung diseases such as COPD")
        st.write("4. Reduced risk of infertility in women of childbearing age")
        st.write("5. Lower risk of having a low-birth-weight baby")
        st.write("6. Improved overall health and quality of life")
        
        st.subheader("How You Can Help Others")
        st.write("As a non-smoker, you can play a crucial role in helping others quit:")
        st.write("1. Offer support and encouragement to friends or family trying to quit")
        st.write("2. Share information about the health risks of smoking")
        st.write("3. Advocate for smoke-free policies in your community")
        st.write("4. Don't start smoking - be a positive role model for others")

# Main execution
if _name_ == "_main_":
    st.sidebar.title("Stroke Risk Factor Explorer")
    st.sidebar.write("Explore different risk factors and learn how to reduce your risk of stroke.")
    
    # User profile in sidebar
    with st.sidebar.expander("Your Profile"):
        if 'name' not in st.session_state:
            st.session_state.name = st.text_input("Your Name")
        if 'age' not in st.session_state:
            st.session_state.age = st.slider("Your Age", 18, 100, 30)
        if 'gender' not in st.session_state:
            st.session_state.gender = st.selectbox("Your Gender", ["Male", "Female", "Other"])
    
    # Risk factor selection
    risk_factor = st.sidebar.radio(
        "Choose a risk factor to explore:",
        ["Body Mass Index (BMI)", "Diet", "Sodium Intake", "Cholesterol",
         "Blood Glucose", "Blood Pressure", "Physical Activity",
         "Air Pollution", "Smoking"]
    )
    
    # Main content
    st.title(f"Exploring {risk_factor}")
    
    if risk_factor == "Body Mass Index (BMI)":
        bmi_campaign()
    elif risk_factor == "Diet":
        diet_campaign()
    elif risk_factor == "Sodium Intake":
        sodium_campaign()
    elif risk_factor == "Cholesterol":
        cholesterol_campaign()
    elif risk_factor == "Blood Glucose":
        glucose_campaign()
    elif risk_factor == "Blood Pressure":
        blood_pressure_campaign()
    elif risk_factor == "Physical Activity":
        physical_activity_campaign()
    elif risk_factor == "Air Pollution":
        pollution_campaign()
    elif risk_factor == "Smoking":
        smoking_campaign()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("This app is for educational purposes only. Always consult with a healthcare professional for medical advice.")