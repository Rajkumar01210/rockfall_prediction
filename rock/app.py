import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle, smtplib
from email.mime.text import MIMEText
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"

# ---------- Email Alert Function ----------
def send_email_alert(risk_score):
    sender = "yourmail@gmail.com"
    password = "your-app-password"   # generate Gmail app password
    receiver = "receiver@gmail.com"  # your phone mail id

    subject = "⚠️ Rockfall Risk Alert!"
    body = f"High Risk Detected! Risk Score = {risk_score:.2f}. Immediate action required."

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver

    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(sender, password)
        server.sendmail(sender, [receiver], msg.as_string())
        server.quit()
        st.success("📩 Email alert sent successfully!")
    except Exception as e:
        st.error(f"❌ Error sending email: {e}")

# ---------- Load Models ----------
IMG_MODEL_PATH = "models/rockfall_image_model.h5"
SENSOR_MODEL_PATH = "models/sensor_model.pkl"

img_model = None
sensor_model = None

if os.path.exists(IMG_MODEL_PATH):
    try:
        img_model = tf.keras.models.load_model(IMG_MODEL_PATH)
    except:
        pass
if os.path.exists(SENSOR_MODEL_PATH):
    try:
        with open(SENSOR_MODEL_PATH,"rb") as f:
            sensor_model = pickle.load(f)
    except:
        pass

# ---------- Streamlit UI ----------
st.title("🪨 Rockfall Prediction & Alert System")

# User enters email for alerts
st.sidebar.header("⚙️ Settings")
receiver_email = st.sidebar.text_input("Enter email for alerts:", "test@gmail.com")

# Image input
st.header("Image Prediction")
uploaded = st.file_uploader("Upload slope image", type=["jpg","png"])
image_prob = 0.0
if uploaded:
    img = Image.open(uploaded).convert("RGB").resize((128,128))
    st.image(img, caption="Uploaded")
    arr = np.array(img)/255.0
    if img_model:
        image_prob = float(img_model.predict(arr.reshape(1,128,128,3))[0][0])
    else:
        image_prob = arr.mean()  # fallback approx
    st.write(f"Image Risk Probability: {image_prob:.2f}")

# Sensor input
st.header("Sensor Prediction")
rain = st.number_input("Rainfall (mm)",0,250,50)
vib = st.number_input("Vibration",0.0,10.0,0.2)
temp = st.number_input("Temperature (°C)",-20,60,25)
sensor_prob = 0.0
if st.button("Predict Sensor Risk"):
    if sensor_model:
        sensor_prob = float(sensor_model.predict_proba([[rain,vib,temp]])[0][1])
    else:
        # fallback logic
        score=0
        if rain>200: score+=0.5
        if vib>1.0: score+=0.4
        if temp<10: score+=0.1
        sensor_prob = min(1.0,score)
    st.write(f"Sensor Risk Probability: {sensor_prob:.2f}")

# Combined output
st.header("Final Risk & Alert")
combined = image_prob*0.6 + sensor_prob*0.4
st.write(f"Combined Risk Score: {combined:.2f}")

if combined >= 0.7:
    st.error("⚠️ HIGH RISK – Immediate action needed!")
    if receiver_email:
        send_email_alert(combined, receiver_email)   # send to entered mail
elif combined >= 0.4:
    st.warning("⚠️ MEDIUM RISK – Monitor closely.")
else:
    st.success("✅ LOW RISK – Safe conditions.")

