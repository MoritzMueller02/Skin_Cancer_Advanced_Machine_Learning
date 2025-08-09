import os
import streamlit as st
from PIL import Image, ImageEnhance
import pandas as pd
from datetime import datetime
from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    Qwen2VLForConditionalGeneration, Qwen2VLProcessor,
)
import requests
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
import cv2
import numpy as np

st.set_page_config(page_title="Skin Cancer Dashboard", layout="wide")

# --- Configuration ---
# Ensure you have set your Hugging Face token as an environment variable:
# export HF_TOKEN="YOUR_TOKEN_HERE"
MODEL_NAME = "Anwarkh1/Skin_Cancer-Image_Classification"
LLM_NAME = "google/flan-t5-large"
HF_TOKEN   = "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"
DATA_DIR = "data/harvard_dataset"  # Path where you download and unpack the Harvard Dataverse dataset
DIARY_CSV = "diary.csv"
gemini_api_secret_name = 'AIzaSyDA3nwO6265GZCDaBqTxTAmKWwrSCfSqnc'
CANCER_DIR = r"D:\Models\googleflan-t5-xl" #you 
LLM_DIR = r"D:\Models\SkinCancer"

# Initialize session state defaults
if 'initialized' not in st.session_state:
    st.session_state['label'] = None
    st.session_state['score'] = None
    st.session_state['mole_id'] = ''
    st.session_state['geo_location'] = ''
    st.session_state['chat_history'] = []
    st.session_state['initialized'] = True

# Initialize geolocator for free geocoding
geolocator = Nominatim(user_agent="skin-dashboard", timeout = 10)

# --- Load Model & Feature Extractor ---
@st.cache_resource
def load_image_model(token: str):
    extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_NAME,
        model_dir = LLM_DIR,
        use_auth_token=token
    )
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        use_auth_token=token
    )
    return pipeline(
        "image-classification",
        model=model,
        feature_extractor=extractor,
        device=0  # set to GPU index or -1 for CPU
    )


@st.cache_resource
def load_llm(token: str):

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_NAME,
        model_dir = CANCER_DIR,
        use_auth_token=token
    )
    # Use Seq2SeqLM for T5-style (text2text) models:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        LLM_NAME,
        use_auth_token=token,
    )
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",    # or device=0 for single GPU / -1 for CPU
        max_length=10000,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,

    )

# Load the Gemini model
classifier = load_image_model(HF_TOKEN) if HF_TOKEN else None
explainer = load_llm(HF_TOKEN) if HF_TOKEN else None


# --- Diary Init ----

def save_entry(img_path: str, mole_id: str, geo_location: str,
               label: str, score: float,
               body_location: str, prior_consult: str, pain: str, itch: str):
    # Ensure that the DataFrame is being read correctly and entry is saved
    if not os.path.exists(DIARY_CSV):
        df = pd.DataFrame(columns=["timestamp", "image_path", "mole_id", "geo_location", 
                                   "label", "score", "body_location", "prior_consultation", 
                                   "pain", "itch"])
    else:
        df = pd.read_csv(DIARY_CSV)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "image_path": img_path,
        "mole_id": mole_id,
        "geo_location": geo_location,
        "label": label,
        "score": float(score),
        "body_location": body_location,
        "prior_consultation": prior_consult,
        "pain": pain,
        "itch": itch
    }

    entry_df = pd.DataFrame([entry])
    df = pd.concat([df, entry_df], ignore_index=True)
    df.to_csv(DIARY_CSV, index=False)
    # Debugging: Confirm the entry was saved
    st.write(f"Entry saved for mole ID: {mole_id}")

# Image Preprocessing

def preprocess_and_detect_mole(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    h, w = V.shape

    # --- 2) Segment original mole via Otsu on Value ---
    _, mask = cv2.threshold(V, 0, 255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # --- 3) Randomly distort border ---
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.dilate(mask, kern, iterations=np.random.randint(1,4))
    mask = cv2.erode (mask, kern, iterations=np.random.randint(1,4))

    # --- 4) Sprinkle extra dark blotches ---
    for _ in range(np.random.randint(5, 15)):
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        ax = np.random.randint(5, 20)
        ay = np.random.randint(5, 20)
        ang = np.random.randint(0, 360)
        cv2.ellipse(mask, (cx, cy), (ax, ay), ang, 0, 360, 255, -1)

    # --- 5) Add variegated noise in V channel inside mask ---
    noise = np.random.normal(loc=0, scale=30, size=(h, w)).astype(np.int16)
    V2 = V.astype(np.int16) + noise
    V2 = np.clip(V2, 0, 255).astype(np.uint8)
    V = np.where(mask==255, V2, V)

    # --- 6) Random hue shift + saturation boost inside mask ---
    hue_shift = np.random.randint(-15, 15)
    H2 = (H.astype(np.int16) + hue_shift) % 180
    H = np.where(mask==255, H2.astype(np.uint8), H)

    sat_boost = np.random.randint(20, 70)
    S2 = np.clip(S.astype(np.int16) + sat_boost, 0, 255)
    S = np.where(mask==255, S2.astype(np.uint8), S)

    # --- 7) Recombine & convert back to RGB ---
    hsv_mod = cv2.merge([H, S, V])
    rgb_mod = cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2RGB)

    return Image.fromarray(rgb_mod)


# -----Streamlit layout ---- 
st.title("🩺 Skin Cancer Recognition Dashboard")
menu = ["Scan Mole","Chat","Diary", "Dataset Explorer"]
choice = st.sidebar.selectbox("Navigation", menu)

# --- Initialize Scan a Mole ---

if choice == "Scan Mole":
    st.header("🔍 Scan a Mole")
    if not classifier:
        st.error("Missing HF_TOKEN.")
        st.stop()

    upload = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

    if upload is not None:
        # Preprocess the image and detect mole automatically
        preprocessed_image = preprocess_and_detect_mole(upload)

        st.image(preprocessed_image, caption="Preprocessed", use_container_width=True)

    # Collect user inputs
    mole = st.text_input("Mole ID")
    city = st.text_input("Geographic location")
    body = st.selectbox(
        "Body location",
        ["Face", "Scalp", "Neck", "Chest", "Back", "Arm", "Hand", "Leg", "Foot", "Other"]
    )
    prior = st.radio("Prior consult?", ["Yes", "No"], horizontal=True)
    pain = st.radio("Pain?", ["Yes", "No"], horizontal=True)
    itch = st.radio("Itch?", ["Yes", "No"], horizontal=True)

    if st.button("Classify"):
        if not mole or not city:
            st.error("Enter ID and location.")
        else:
            # Reset map visibility when new classification occurs
            st.session_state.show_map = False

            with st.spinner("Analyzing..."):
                # Reset file pointer and classify
                upload.seek(0)
                img = Image.open(upload).convert("RGB")
                out = classifier(preprocessed_image)

            lbl, scr = out[0]["label"], out[0]["score"]

            # Store label and geo_location in session_state
            st.session_state["label"] = lbl
            st.session_state["score"] = scr
            st.session_state['mole_id'] = mole
            st.session_state["geo_location"] = city  # Store geo_location

            # Save the classification and user inputs in the DataFrame
            save_entry(
                img_path="path/to/mole/image.jpg",  # Replace with actual image path
                mole_id=mole,
                geo_location=city,
                label=lbl,
                score=scr,
                body_location=body,
                prior_consult=prior,
                pain=pain,
                itch=itch
            )

            st.success(f"Classification result: {lbl} with score {scr:.4f}")

    # Handle label and recommendation display
    if st.session_state.get("label"):
        lbl = st.session_state["label"]
        score = st.session_state["score"]

        # Hardcoded explanations and recommendations
        recommendations = {
            "benign_keratosis-like_lesions": (
                "These are non-cancerous growths often due to sun exposure. "
                "They typically do not transform into cancer.",
                "Recommendation: Routine self-monitoring; clinical check every 12 months."
            ),
            "basal_cell_carcinoma": (
                "The most common form of skin cancer. It grows slowly and rarely spreads but can damage surrounding tissue.",
                "Recommendation: See a dermatologist within 2 weeks for evaluation and treatment."
            ),
            "actinic_keratoses": (
                "Rough, scaly patches caused by long-term sun exposure. They can progress to squamous cell carcinoma.",
                "Recommendation: Schedule dermatology visit within 1 month; follow-up every 6 months."
            ),
            "vascular_lesions": (
                "Benign growths of blood vessels such as hemangiomas or spider angiomas. "
                "Usually harmless but monitor changes.",
                "Recommendation: Self-monitor; clinical review if changes occur or annually."
            ),
            "melanocytic_Nevi": (
                "Common moles composed of melanocyte cells. Most are benign but can rarely develop into melanoma.",
                "Recommendation: Monthly self-exams using the ABCDE rule; dermatology check every 12 months."
            ),
            "melanoma": (
                "A serious form of skin cancer that can spread quickly to other organs if not caught early.",
                "Recommendation: Urgent dermatologist appointment within 1 week; regular follow-up as advised."
            ),
            "dermatofibroma": (
                "A benign, firm skin nodule usually caused by minor skin injury. Remains stable over time.",
                "Recommendation: No immediate treatment needed; check annually or if it changes."
            )  
        }

        if lbl in recommendations:
            st.subheader(f"About {lbl.replace('_', ' ').title()}")
            st.write(recommendations[lbl][0])
            st.write(recommendations[lbl][1])
        else:
            st.write("No specific information available for this diagnosis.")


    if st.session_state.get("geo_location"):
        loc = geolocator.geocode(st.session_state.get("geo_location"))
        if loc:
            # Allow user to adjust search radius
            radius = st.slider(
                "Search radius (meters)",
                min_value=1000,
                max_value=20000,
                value=5000,
                step=500,
                key="search_radius"
            )

            # Center the button in the layout
            cols = st.columns([1, 2, 1])
            with cols[1]:
                if st.button("Recommend Nearby Doctors", key="show_map_btn"):
                    st.session_state.show_map = True

            if st.session_state.get("show_map"):
                # Initialize map at user's location
                m = folium.Map([loc.latitude, loc.longitude], zoom_start=12)
                folium.CircleMarker(
                    location=[loc.latitude, loc.longitude],
                    radius=8,
                    popup="You",
                    weight=1,
                    fill=True
                ).add_to(m)

                # Use MarkerCluster for better visualization
                from folium.plugins import MarkerCluster
                cluster = MarkerCluster().add_to(m)

                # Query Overpass API for nearby clinics/doctors
                resp = requests.post(
                    "https://overpass-api.de/api/interpreter",
                    data={
                        "data": (
                            f"[out:json];node(around:{radius},{loc.latitude},{loc.longitude})"
                            "[~\"^(amenity|healthcare)$\"~\"clinic|doctors\"];out;"
                        )
                    }
                )
                elements = resp.json().get("elements", [])

                if elements:
                    for el in elements:
                        tags = el.get("tags", {})
                        lat = el.get("lat") or el.get("center", {}).get("lat")
                        lon = el.get("lon") or el.get("center", {}).get("lon")
                        name = tags.get("name", "Clinic/Doctor")
                        folium.Marker(
                            location=[lat, lon],
                            popup=name,
                            icon=folium.Icon(icon="plus-sign", prefix="glyphicon")
                        ).add_to(cluster)
                else:
                    st.warning("No clinics or doctors found in this radius.")

                # Display the map
                st.markdown("### Nearby Clinics & Doctors")
                st_folium(m, width="100%", height=500)



# --- Chat ---
elif choice == "Chat":
    st.header("💬 Follow-Up Chat")
    if not st.session_state['label']:
        st.info("Please perform a scan first in the 'Scan Mole' tab.")
    else:
        lbl = st.session_state['label']
        scr = st.session_state['score']
        mid = st.session_state['mole_id']
        gloc = st.session_state['geo_location']
        st.markdown(f"**Context:** prediction for **{mid}** at **{gloc}** is **{lbl}** (confidence {scr:.2f}).")

        # New user message comes first for immediate loop
        user_q = st.chat_input("Ask a follow-up question:", key="chat_input")
        if user_q and explainer:
            st.session_state['chat_history'].append({'role':'user','content':user_q})
            system_p = "You are a dermatology assistant. Provide concise medical advice without clarifying questions."
            tpl = (
                f"{system_p}\nContext: prediction is {lbl} with confidence {scr:.2f}.\n"
                f"User: {user_q}\nAssistant:"
            )
            with st.spinner("Generating response..."):
                reply = explainer(tpl)[0]['generated_text']
            st.session_state['chat_history'].append({'role':'assistant','content':reply})

        # Display the updated chat history
        for msg in st.session_state['chat_history']:
            prefix = 'You' if msg['role']=='user' else 'AI'
            st.markdown(f"**{prefix}:** {msg['content']}")

# --- Diary Page ---
elif choice == "Diary":
    st.header("📖 Skin Cancer Diary")
    df = pd.read_csv(DIARY_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df.empty:
        st.info("No diary entries yet.")
    else:
        mole_ids = sorted(df['mole_id'].unique())
        sel = st.selectbox("Select Mole to View", ['All'] + mole_ids, key="diary_sel")
        if sel == 'All':
            # Display moles in columns (max 3 per row)
            chunks = [mole_ids[i:i+3] for i in range(0, len(mole_ids), 3)]
            for group in chunks:
                cols = st.columns(len(group))
                for col, mid in zip(cols, group):
                    with col:
                        st.subheader(mid)
                        entries = df[df['mole_id'] == mid].sort_values('timestamp')
                        # Show image timeline
                        for _, row in entries.iterrows():
                            if os.path.exists(row['image_path']):
                                st.image(
                                    row['image_path'],
                                    width=150,
                                    caption=f"{row['timestamp'].strftime('%Y-%m-%d')} — {row['score']:.2f}"
                                )
                        st.write(f"Total scans: {len(entries)}")
        else:
            # Detailed view for a single mole
            entries = df[df['mole_id'] == sel].sort_values('timestamp')
            if entries.empty:
                st.warning(f"No entries for {sel}.")
            else:
                # Score over time
                st.line_chart(entries.set_index('timestamp')['score'])

                # Organized image timeline in 4 columns
                st.markdown("#### Image Timeline")
                chunks = [entries.iloc[i:i+4] for i in range(0, len(entries), 4)]
                for chunk in chunks:
                    cols = st.columns(4)
                    for col, (_, row) in zip(cols, chunk.iterrows()):
                        with col:
                            if os.path.exists(row['image_path']):
                                st.image(
                                    row['image_path'],
                                    width=200,
                                    caption=(
                                        f"{row['timestamp'].strftime('%Y-%m-%d %H:%M')} — "
                                        f"Score: {row['score']:.2f}"
                                    )
                                )

                # Details table
                st.markdown("#### Details")
                st.dataframe(
                    entries[
                        ['timestamp','geo_location','label','score',
                         'body_location','prior_consultation','pain','itch']
                    ]
                    .rename(columns={
                        'timestamp':'Time','geo_location':'Location',
                        'label':'Diagnosis','score':'Confidence',
                        'body_location':'Body Part','prior_consultation':'Prior Consult',
                        'pain':'Pain','itch':'Itch'
                    })
                    .sort_values('Time', ascending=False)
                )

else:
    st.header("📂 Dataset Explorer")
    st.write("Preview images from the Harvard Skin Cancer Dataset")

    # pick up to 15 image files
    image_files = [
    f for f in os.listdir(DATA_DIR)
    if os.path.isfile(os.path.join(DATA_DIR, f))
    and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ][:15]

    for i in range(0, len(image_files), 3):
        cols = st.columns(3)
        for col, fn in zip(cols, image_files[i : i + 3]):
            path = os.path.join(DATA_DIR, fn)
            img = Image.open(path)
            col.image(img, use_container_width=True)
            col.caption(fn)

st.sidebar.markdown("---")
st.sidebar.write("Dataset powered by Harvard Dataverse [DBW86T]")
st.sidebar.write(f"Model: {MODEL_NAME}")
st.sidebar.write(f"LLM: {LLM_NAME}")

if __name__ == '__main__':
    st.write()
