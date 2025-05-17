# DermaDetect
import streamlit as st
import sqlite3
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="DermaDetect", page_icon="🩺", layout="wide")

# --- Database Setup ---
@st.cache_resource
def get_connection():
    conn = sqlite3.connect("users.db", check_same_thread=False)
    return conn

conn = get_connection()
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)")
c.execute("""CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    disease TEXT,
    confidence REAL,
    timestamp TEXT
)""")
conn.commit()

# --- Load Model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('derma_detect_mobilenet.h5')
model = load_model()

# --- Disease info ---
disease_labels = {
    0: "Benign-Keratosis",
    1: "Melanocytic-Nevus",
    2: "Melanoma",
    3: "Normal",
}
disease_descriptions = {
    0: """**Benign Keratosis**  
Non-cancerous growths caused by aging or sun exposure.  
Typically brown or tan patches with rough or scaly texture.  
Common on face, chest, back.""",

    1: """**Melanocytic Nevus**  
Pigment cell clusters appearing as brown/black spots.  
Can be congenital or acquired.  
Usually round, smooth, uniform color.""",

    2: """**Melanoma**  
Serious skin cancer from melanocytes.  
New or changing dark spots or moles.  
Irregular borders, varied color, fast growth.""",

    3: """**Normal Skin**  
Even tone, no visible lesions or discoloration.  
Healthy moisture and oil balance.""",
}
disease_prevention = {
    0: """Use sunscreen daily, avoid tanning beds, wear protective clothing.""",
    1: """Monitor moles, avoid UV exposure, get regular skin checks.""",
    2: """Apply high SPF sunscreen, avoid peak sun hours, self-examine monthly.""",
    3: """Maintain skincare routine, hydrate, protect from sun.""",
}

st.warning("📝 This tool provides AI-based assistance. Consult a dermatologist for diagnosis.")

# --- CSS Styling for Centered Login/Signup ---
st.markdown("""
<style>
body {
    margin: 0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(-45deg, #ff9a9e, #fad0c4, #a1c4fd, #c2e9fb, #fbc2eb, #a6c1ee);
    background-size: 400% 400%;
    animation: rainbowBG 15s ease infinite;
    height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
}
@keyframes rainbowBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
#login-container {
    background: white;
    padding: 2rem 3rem;
    border: 4px solid #3b82f6; /* blue border */
    border-radius: 15px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    max-width: 400px;
    width: 90vw;
    text-align: center;
}
input[type="text"], input[type="password"] {
    width: 100%;
    padding: 0.5rem 0.7rem;
    margin: 0.6rem 0 1.2rem 0;
    border: 2px solid #3b82f6;
    border-radius: 8px;
    font-size: 1rem;
    outline-color: #2563eb;
    transition: border-color 0.3s ease;
}
input[type="text"]:focus, input[type="password"]:focus {
    border-color: #1d4ed8;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    font-weight: 700;
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    width: 100%;
    font-size: 1.1rem;
    transition: background-color 0.3s ease;
}
.stButton>button:hover {
    background-color: #1e40af;
}
h1 {
    margin-bottom: 1rem;
    color: #1e40af;
    font-weight: 900;
    letter-spacing: 1.5px;
}
.radio > label {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

def create_user(username, password):
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        st.error("⚠️ Username already exists.")
        return False

def authenticate_user(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return c.fetchone() is not None

def show_login():
    st.markdown('<div id="login-container">', unsafe_allow_html=True)
    st.markdown("<h1>DermaDetect</h1>", unsafe_allow_html=True)
    st.markdown("### 🔐 Login or Sign Up")
    choice = st.radio("", ["Login", "Sign Up"], horizontal=True, key="auth_choice")

    input_username = st.text_input("Username", key="input_username")
    input_password = st.text_input("Password", type="password", key="input_password")

    if choice == "Sign Up":
        input_confirm_password = st.text_input("Confirm Password", type="password", key="input_confirm_password")
        if st.button("Sign Up"):
            if not input_username or not input_password or not input_confirm_password:
                st.error("Please fill in all fields.")
            elif input_password != input_confirm_password:
                st.error("Passwords do not match.")
            else:
                created = create_user(input_username, input_password)
                if created:
                    st.success("✅ Account created! Please login.")
    else:  # Login
        if st.button("Login"):
            if authenticate_user(input_username, input_password):
                st.session_state.logged_in = True
                st.session_state.username = input_username
                st.success("✅ Login successful!")
                st.experimental_rerun()
            else:
                st.error("❌ Invalid username or password.")
    st.markdown('</div>', unsafe_allow_html=True)

def main_app():
    st.sidebar.header(f"👤 User: {st.session_state.username}")
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

    st.title("👩🏻‍🔬 DermaDetect: Automated Diagnosis of Skin Diseases with Image Recognition")
    st.write("Upload an image of a skin lesion and click 'Analyze' to get a prediction.")

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        if st.button("🔍 Analyze"):
            try:
                img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
                img_array = np.expand_dims(np.array(img), axis=0)
                img_array = preprocess_input(img_array)

                predictions = model.predict(img_array)
                predicted_class = int(np.argmax(predictions))
                confidence = predictions[0][predicted_class] * 100

                c.execute("INSERT INTO predictions (username, disease, confidence, timestamp) VALUES (?, ?, ?, ?)",
                          (st.session_state.username, disease_labels[predicted_class], confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()

                st.success(f"🧾 **Prediction**: {disease_labels[predicted_class]} ({confidence:.2f}%)")
                st.markdown("### 🧠 Description")
                st.info(disease_descriptions[predicted_class])
                st.markdown("### 🛡️ Prevention")
                st.success(disease_prevention[predicted_class])
            except Exception as e:
                st.error(f"⚠️ Error during prediction: {e}")

    with st.expander("📊 View Prediction History and Graphs"):
        c.execute("SELECT disease, confidence, timestamp FROM predictions WHERE username = ? ORDER BY timestamp DESC", (st.session_state.username,))
        records = c.fetchall()
        if records:
            df = pd.DataFrame(records, columns=["Disease", "Confidence", "Timestamp"])
            st.dataframe(df)

            st.subheader("Bar Chart of Prediction Confidence")
            st.bar_chart(df.set_index("Disease")["Confidence"])

            st.subheader("Line Chart of Confidence Over Time")
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df.sort_values("Timestamp", inplace=True)
            df.set_index("Timestamp", inplace=True)
            st.line_chart(df["Confidence"])

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()
else:
    show_login()
