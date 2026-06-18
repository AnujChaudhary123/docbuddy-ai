# importing the required libraries here.
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle
import firebase_admin
from firebase_admin import credentials
import requests
import json
import ast
from dotenv import load_dotenv
import os
import warnings
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from typing import Generator
from groq import Groq
from langdetect import detect
from translate import Translator

warnings.filterwarnings("ignore")
load_dotenv()

# init firebase app here.
cred = credentials.Certificate(os.getenv('FIREBASE_JSON_PATH'))
try:
    firebase_admin.get_app()
except ValueError as e:
    firebase_admin.initialize_app(cred)

# setting up the page config here.
st.set_page_config(
    page_title="SmartBuddy",
    page_icon=r"static\\favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/',
        'Report a bug': "https://www.github.com/",
        'About': """
### 🤖 Welcome to SmartBuddy AI — Your 24x7 Health Companion! 🩺

SmartBuddy AI is your intelligent, always-available virtual health buddy.  
Whether you're feeling under the weather or just curious about your symptoms,  
our AI is here to provide insights, suggestions, and support — instantly. 💬

"""

    }
)

# Load data
symptom_data = pd.read_csv("Data\\symptoms_df.csv")
precautions_data = pd.read_csv("Data\\precautions_df.csv")
workout_data = pd.read_csv("Data\\workout_df.csv")
desc_data = pd.read_csv("Data\\description.csv")
diets_data = pd.read_csv("Data\\diets.csv")
medication_data = pd.read_csv("Data\\medications.csv")

precautions_data.replace('nan', None, inplace=True)
precautions_data = precautions_data.where(pd.notnull(precautions_data), None)

# State
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if 'disease' not in st.session_state:
    st.session_state.disease = None
if 'description' not in st.session_state:
    st.session_state.description = None
if 'precautions' not in st.session_state:
    st.session_state.precautions = None
if 'workout' not in st.session_state:
    st.session_state.workout = None
if 'diets' not in st.session_state:
    st.session_state.diets = None
if 'medications' not in st.session_state:
    st.session_state.medications = None

# Prediction and processing functions
# Create symptom-to-index map from all known symptoms in the model
symptoms_dict = {
    'itching': 0,
    'skin_rash': 1,
    'nodal_skin_eruptions': 2,
    'continuous_sneezing': 3,
    'shivering': 4,
    'chills': 5,
    'joint_pain': 6,
    'stomach_pain': 7,
    'acidity': 8,
    'ulcers_on_tongue': 9,
    'muscle_wasting': 10,
    'vomiting': 11,
    'burning_micturition': 12,
    'spotting_urination': 13,
    'fatigue': 14,
    'weight_gain': 15,
    'anxiety': 16,
    'cold_hands_and_feets': 17,
    'mood_swings': 18,
    'weight_loss': 19,
    'restlessness': 20,
    'lethargy': 21,
    'patches_in_throat': 22,
    'irregular_sugar_level': 23,
    'cough': 24,
    'high_fever': 25,
    'sunken_eyes': 26,
    'breathlessness': 27,
    'sweating': 28,
    'dehydration': 29,
    'indigestion': 30,
    'headache': 31,
    'yellowish_skin': 32,
    'dark_urine': 33,
    'nausea': 34,
    'loss_of_appetite': 35,
    'pain_behind_the_eyes': 36,
    'back_pain': 37,
    'constipation': 38,
    'abdominal_pain': 39,
    'diarrhoea': 40,
    'mild_fever': 41,
    'yellow_urine': 42,
    'yellowing_of_eyes': 43,
    'acute_liver_failure': 44,
    'fluid_overload': 45,
    'swelling_of_stomach': 46,
    'swelled_lymph_nodes': 47,
    'malaise': 48,
    'blurred_and_distorted_vision': 49,
    'phlegm': 50,
    'throat_irritation': 51,
    'redness_of_eyes': 52,
    'sinus_pressure': 53,
    'runny_nose': 54,
    'congestion': 55,
    'chest_pain': 56,
    'weakness_in_limbs': 57,
    'fast_heart_rate': 58,
    'pain_during_bowel_movements': 59,
    'pain_in_anal_region': 60,
    'bloody_stool': 61,
    'irritation_in_anus': 62,
    'neck_pain': 63,
    'dizziness': 64,
    'cramps': 65,
    'bruising': 66,
    'obesity': 67,
    'swollen_legs': 68,
    'swollen_blood_vessels': 69,
    'puffy_face_and_eyes': 70,
    'enlarged_thyroid': 71,
    'brittle_nails': 72,
    'swollen_extremeties': 73,
    'excessive_hunger': 74,
    'extra_marital_contacts': 75,
    'drying_and_tingling_lips': 76,
    'slurred_speech': 77,
    'knee_pain': 78,
    'hip_joint_pain': 79,
    'muscle_weakness': 80,
    'stiff_neck': 81,
    'swelling_joints': 82,
    'movement_stiffness': 83,
    'spinning_movements': 84,
    'loss_of_balance': 85,
    'unsteadiness': 86,
    'weakness_of_one_body_side': 87,
    'loss_of_smell': 88,
    'bladder_discomfort': 89,
    'foul_smell_of_urine': 90,
    'continuous_feel_of_urine': 91,
    'passage_of_gases': 92,
    'internal_itching': 93,
    'toxic_look_(typhos)': 94,
    'depression': 95,
    'irritability': 96,
    'muscle_pain': 97,
    'altered_sensorium': 98,
    'red_spots_over_body': 99,
    'belly_pain': 100,
    'abnormal_menstruation': 101,
    'dischromic_patches': 102,
    'watering_from_eyes': 103,
    'increased_appetite': 104,
    'polyuria': 105,
    'family_history': 106,
    'mucoid_sputum': 107,
    'rusty_sputum': 108,
    'lack_of_concentration': 109,
    'visual_disturbances': 110,
    'receiving_blood_transfusion': 111,
    'receiving_unsterile_injections': 112,
    'coma': 113,
    'stomach_bleeding': 114,
    'distention_of_abdomen': 115,
    'history_of_alcohol_consumption': 116,
    'fluid_overload.1': 117,
    'blood_in_sputum': 118,
    'prominent_veins_on_calf': 119,
    'palpitations': 120,
    'painful_walking': 121,
    'pus_filled_pimples': 122,
    'blackheads': 123,
    'scurring': 124,
    'skin_peeling': 125,
    'silver_like_dusting': 126,
    'small_dents_in_nails': 127,
    'inflammatory_nails': 128,
    'blister': 129,
    'red_sore_around_nose': 130,
    'yellow_crust_ooze': 131
}

diseases_list = {
    15: 'Fungal infection',
    4: 'Allergy',
    16: 'GERD',
    9: 'Chronic cholesterol',
    14: 'Drug Reaction',
    33: 'Peptic ulcer disease',
    1: 'Acute Disease',
    12: 'Diabetes',
    17: 'Gastroenteritis',
    6: 'Bronchial Asthma',
    23: 'Hypertension',
    30: 'Migraine',
    7: 'Cervical spondylosis',
    32: 'Paralysis (brain hemorrhage)',
    28: 'Jaundice',
    29: 'Malaria',
    8: 'Chicken pox',
    11: 'Dengue',
    37: 'Typhoid',
    40: 'Hepatitis A',
    19: 'Hepatitis B',
    20: 'Hepatitis C',
    21: 'Hepatitis D',
    22: 'Hepatitis E',
    3: 'Alcoholic hepatitis',
    36: 'Tuberculosis',
    10: 'Common Cold',
    34: 'Pneumonia',
    13: 'Dimorphic hemorrhoids (piles)',
    18: 'Heart attack',
    39: 'Varicose veins',
    26: 'Hypothyroidism',
    24: 'Hyperthyroidism',
    25: 'Hypoglycemia',
    31: 'Osteoarthritis',
    5: 'Arthritis',
    0: '(vertigo) Paroxysmal Positional Vertigo',
    2: 'Acne',
    38: 'Urinary tract infection',
    35: 'Psoriasis',
    27: 'Impetigo'
}
def get_predicted_values(patient_symptoms):
    st.session_state.predicted = True
    model = pickle.load(open('Model\\model.pkl', 'rb'))

    input_vector = [0] * len(symptoms_dict)
    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    prediction_index = model.predict([input_vector])[0]
    return diseases_list.get(prediction_index, "Unknown Disease")


def get_desc(predicted_value):
    match = desc_data[desc_data["Disease"] == predicted_value]
    if not match.empty:
        return match["Description"].values[0]
    else:
        return "No description found for this disease."

def get_precautions(predicted_value):
    match = precautions_data[precautions_data['Disease'] == predicted_value]
    if not match.empty:
        return match.values[0][2:]
    else:
        return ["No precautions found."]

def get_medication(predicted_value):
    match = medication_data[medication_data['Disease'] == predicted_value]
    if not match.empty:
        return ast.literal_eval(match['Medication'].values[0])
    else:
        return ["No medication found."]

def get_workout(predicted_value):
    match = workout_data[workout_data['disease'] == predicted_value]
    return match["workout"].values if not match.empty else ["No workout found."]

def get_diet(predicted_value):
    match = diets_data[diets_data['Disease'] == predicted_value]
    return ast.literal_eval(match['Diet'].values[0]) if not match.empty else ["No diet found."]


# Removed login — simplified account function
def account():
    st.image("static/Login-DocBuddy.png")
    st.title("👥 Meet the Minds Behind SmartBuddy")

    st.markdown("""
        ### 🤖 About SmartBuddy  
        **SmartBuddy** is your intelligent virtual health assistant — here to guide you through symptom analysis, wellness recommendations, and mental support using the power of AI.

        ---

        ### 👨‍💻 Developed By:
        - **Anuj Chaudhary**  
        - **Pranjal Singh**  
        - **Aditya Singh**

        ---""")

       
       

    st.success("You're ready to explore SmartBuddy — your health partner in AI form 💡")


# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="SmartBuddy",
        options=["Home", "Recommendations", "Generate Report", "Chat With Me", "WorkFlow", "Account"],
        icons=["house", "magic", "book", "chat", "activity", "gear"],
        menu_icon="app-indicator",
        default_index=0,
    )

# Home tab
if selected == "Home":
    st.title("🤖 Welcome to SmartBuddy")
    st.subheader("Your Personal AI Health Companion 🩺")

    


    st.markdown("""
        ###  Why Choose SmartBuddy?
        - 🧠 **AI-powered** disease prediction based on your symptoms  
        - 🍎 Personalized **diet and wellness** guidance  
        - 💊 Instant **medication suggestions**  
        - 🧘 Tailored **workout routines**  
        - 🧾 Downloadable health **reports** in one click  
        - 💬 Friendly **AI chatbot** to answer your health questions  
        """)

    st.markdown("---")
    st.info("🚀 Get started using the sidebar")
    st.success("Your health journey starts now 💪")

    

# Recommendations tab
elif selected == "Recommendations":
    st.title("SmartBuddy Recommendation Center 🔮")

    symptoms = st.multiselect("#### Select Symptoms:", sorted(symptoms_dict.keys()))

    if st.button("Predict Disease"):
        if not symptoms:
            st.warning("Please select at least one symptom.")
        else:
            disease = get_predicted_values(symptoms)
            st.session_state.disease = disease
            st.session_state.description = get_desc(disease)
            st.session_state.precautions = get_precautions(disease)
            st.session_state.workout = get_workout(disease)
            st.session_state.diets = get_diet(disease)
            st.session_state.medications = get_medication(disease)

            st.subheader(f"Predicted Disease: {disease}")
            st.write(st.session_state.description)
            st.subheader("Precautions")
            for item in st.session_state.precautions:
                if item:
                    st.write(f"⚠️ {item}")
            st.subheader("Workouts")
            for item in st.session_state.workout:
                st.write(f"💪 {item}")
            st.subheader("Diets")
            for item in st.session_state.diets:
                st.write(f"🥗 {item}")
            st.subheader("Medications")
            for item in st.session_state.medications:
                st.write(f"💊 {item}")

# Report tab
# PDF report generation function
def generate_report(name, age, disease, description, precautions, workouts, diets, medications, file_path):
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("DocBuddy Health Report", styles['Heading1']), Spacer(1, 12)]
    story.append(Paragraph(f"Patient: {name}, Age: {age}", styles['Normal']))
    story.append(Paragraph(f"Disease: {disease}", styles['Normal']))
    story.append(Paragraph(f"Description: {description}", styles['Normal']))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Precautions:", styles['Heading2']))
    story += [Paragraph(f"- {p}", styles['Normal']) for p in precautions if p]
    story.append(Spacer(1, 12))
    story.append(Paragraph("Workouts:", styles['Heading2']))
    story += [Paragraph(f"- {w}", styles['Normal']) for w in workouts]
    story.append(Spacer(1, 12))
    story.append(Paragraph("Diets:", styles['Heading2']))
    story += [Paragraph(f"- {d}", styles['Normal']) for d in diets]
    story.append(Spacer(1, 12))
    story.append(Paragraph("Medications:", styles['Heading2']))
    story += [Paragraph(f"- {m}", styles['Normal']) for m in medications]

    doc.build(story)
    print(f"PDF saved to {file_path}")

if selected == "Generate Report":
    st.title("Generate Health Report 📄")
    name = st.text_input("Patient Name")
    age = st.number_input("Patient Age", min_value=1, max_value=120)

    if st.button("Generate Report"):
        if not st.session_state.predicted:
            st.warning("Please run prediction in Recommendations tab first.")
        elif name and age:
            filename = f"SmartBuddy_{name}_Report.pdf"
            generate_report(
                name, age,
                disease=st.session_state.disease,
                description=st.session_state.description,
                precautions=st.session_state.precautions,
                workouts=st.session_state.workout,
                diets=st.session_state.diets,
                medications=st.session_state.medications,
                file_path=filename
            )
            with open(filename, "rb") as file:
                st.download_button("Download Report", data=file, file_name=filename, mime="application/pdf")
        else:
            st.warning("Please provide valid name and age.")


# Chatbot tab
elif selected == "Chat With Me":
    st.title("Chat With SmartBuddy 🤖")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model_option = st.selectbox("Choose model", ["llama3-8b-8192", "llama2-70b-4096", "mixtral-8x7b-32768"])

    if prompt := st.chat_input("Ask your health question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            chat_completion = client.chat.completions.create(
                model=model_option,
                messages=st.session_state.messages,
                max_tokens=1024,
                stream=True
            )

            response_text = ""
            with st.chat_message("assistant"):
                response_placeholder = st.empty()

                for chunk in chat_completion:
                    if chunk.choices[0].delta.content:
                        response_text += chunk.choices[0].delta.content
                        response_placeholder.markdown(response_text)

            st.session_state.messages.append({"role": "assistant", "content": response_text})

        except Exception as e:
            st.error(str(e))

# ✅ Note: This must be OUTSIDE the previous block (unindented)
elif selected == "WorkFlow":
    st.title("🔄 How SmartBuddy AI Works")
    st.subheader("A Step-by-Step Guide to Your AI-Powered Health Journey 🧠💪")

    st.markdown("### 🩺 Step 1: Symptom Input")
    st.info("""
    Use the **Recommendations** tab to select your symptoms from a dynamic list.  
    You can choose multiple symptoms based on how you feel.
    """)

    st.markdown("### 🧠 Step 2: AI-Powered Disease Prediction")
    st.success("""
    Our trained ML model processes your symptoms instantly and predicts the most likely disease.  
    This gives you a quick, accurate starting point for awareness.
    """)

    st.markdown("### 🍽️ Step 3: Personalized Advice")
    st.markdown("""
    Once a prediction is made, you'll receive:
    - 💊 Recommended **medications**
    - 🧘‍♂️ Suitable **workouts**
    - 🥗 Balanced **diet suggestions**
    - ⚠️ Health **precautions** to follow
    """)

    st.markdown("### 🧾 Step 4: Download Report")
    st.info("""
    Go to the **Generate Report** tab and enter your name and age.  
    You’ll get a downloadable PDF health summary — great for keeping records or sharing with a doctor!
    """)

    st.markdown("### 🗣️ Step 5: Chat with the AI Assistant")
    st.success("""
    Use the **Chat With Me** tab to ask follow-up questions, get emotional support,  
    or clarify doubts about symptoms and wellness — powered by Groq AI models.
    """)

    st.markdown("---")
    st.balloons()
    st.success("Ready to explore? Use the sidebar to begin your wellness journey 🚀")

# ✅ Also correctly placed
elif selected == "Account":
    account()


# PDF report generation function
def generate_report(name, age, disease, description, precautions, workouts, diets, medications, file_path):
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("SmartBuddy Health Report", styles['Heading1']), Spacer(1, 12)]
    story.append(Paragraph(f"Patient: {name}, Age: {age}", styles['Normal']))
    story.append(Paragraph(f"Disease: {disease}", styles['Normal']))
    story.append(Paragraph(f"Description: {description}", styles['Normal']))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Precautions:", styles['Heading2']))
    story += [Paragraph(f"- {p}", styles['Normal']) for p in precautions if p]
    story.append(Spacer(1, 12))
    story.append(Paragraph("Workouts:", styles['Heading2']))
    story += [Paragraph(f"- {w}", styles['Normal']) for w in workouts]
    story.append(Spacer(1, 12))
    story.append(Paragraph("Diets:", styles['Heading2']))
    story += [Paragraph(f"- {d}", styles['Normal']) for d in diets]
    story.append(Spacer(1, 12))
    story.append(Paragraph("Medications:", styles['Heading2']))
    story += [Paragraph(f"- {m}", styles['Normal']) for m in medications]

    doc.build(story)
    print(f"PDF saved to {file_path}")
