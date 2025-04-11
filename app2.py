import streamlit as st
import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image

# Load YOLO model
model = YOLO("D:\\Medical_plant\\best.pt")

# ---------- CSS Styling ----------
import streamlit as st

def local_css():
    st.markdown("""
    <style>
    /* 🌿 General Layout */
    html, body {
        background-color: #f0fdf4;
        font-family: 'Segoe UI', sans-serif;
        overflow-y: scroll !important;
    }

    ::-webkit-scrollbar {
        width: 10px !important;
    }
    ::-webkit-scrollbar-track {
        background: #e0f2f1 !important;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #66bb6a !important;
        border-radius: 10px;
    }

    .main {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0px 6px 20px rgba(0,0,0,0.1);
    }

    /* 🌿 Titles and Text */
    h1, h2, h3 {
        color: #2e7d32;
        font-weight: bold;
    }

    .upload-header {
        font-size: 26px;
        color: #2c3e50 !important;
        text-align: center;
        font-weight: 600;
    }

    .plant-name {
        color: #1b5e20 !important;
        font-size: 28px;
        font-weight: bold;
        background-color: #e8f5e9;
        border-left: 6px solid #43a047;
        border-radius: 8px;
        padding: 1rem;
    }

    .prediction-title {
        font-size: 24px;
        color: #00796b !important;
        margin: 1.5rem 0 0.5rem 0;
    }

    .info-box {
        background-color: #f9f9f9;
        color: #2c3e50;
        padding: 1.5rem;
        border-left: 5px solid #43a047;
        border-radius: 10px;
        font-size: 17px;
        line-height: 1.6;
        margin: 1.5rem 0;
        max-height: 300px;
        overflow-y: auto;
    }

    /* 🌿 Tables */
    table {
        font-size: 16px !important;
        color: #2c3e50 !important;
        background-color: #f1f8e9 !important;
    }

    th, td {
        padding: 12px;
        border: 1px solid #c8e6c9;
    }

    /* 🌿 FileUploader & Markdown */
    .stFileUploader {
        margin: 1rem 0;
    }

    .stMarkdown, .stText, .stAlert {
        color: #2c3e50;
    }

    .block-container {
        padding: 2rem 1rem;
    }

    .resource-link {
        display: block;
        padding: 8px 0;
        color: #1b5e20 !important;
        text-decoration: none;
    }

    .resource-link:hover {
        color: #43a047 !important;
        text-decoration: underline;
    }

    /* 🌿 Expander Styling - Dark with plant theme */
    .streamlit-expander {
        background-color: #1e293b !important;
        border-radius: 10px;
        border: 1px solid #334155;
        margin-bottom: 15px;
    }

    .streamlit-expanderHeader {
        color: #ffffff !important;
        font-size: 20px !important;
        font-weight: bold;
        background-color: #1e293b !important;
    }

    .streamlit-expanderContent {
        color: #e0f2f1 !important;
        font-size: 18px;
    }

    /* 🌿 Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #134e4a !important;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    section[data-testid="stSidebar"] .streamlit-expander {
        border: 1px solid #1b5e20 !important;
        border-radius: 8px;
        background-color: #14532d !important;
    }

    section[data-testid="stSidebar"] div[role="button"][aria-expanded] {
        font-weight: bold;
        font-size: 18px;
        color: white !important;
    }

    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        font-weight: bold;
        font-size: 18px;
        background-color: #14532d !important;
    }

    section[data-testid="stSidebar"] .streamlit-expanderContent {
        padding: 10px;
        background-color: #14532d !important;
        color: white !important;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

    # 🌿 Sidebar logo and name
    st.sidebar.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <img src="https://img.icons8.com/color/96/basil.png" width="100" style="margin-right: 10px;">
        <div>
            <h3 style="margin: 0; color: white; font-family: 'Segoe UI', sans-serif;">
                AyurLeaf
            </h3>
            <span style="font-size: 17px; color: white;">ஆயுர் லீஃப்</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
# ---------- Class Details ----------
class_names = [
    'Azadirachta Indica-Neem(வேம்பு)',
    'Calotropis(எருக்கு)',
    'Ficus Religiosa(அரச மரம்)',
    'Oleander(அரளி)'
]

class_details = {
    "Azadirachta Indica-Neem(வேம்பு)": {
        "botanical_name": "Azadirachta indica",
        "tamil_name": "வேம்பு",
        "common_name": "Neem",
        "family": "Meliaceae",
        "habitat": "Native to the Indian subcontinent, widely cultivated in tropical and semi-tropical regions",
        "medicinal_properties": "Antibacterial, antifungal, anti-inflammatory, and anti-diabetic properties. Used to treat skin diseases, fever, and various other ailments.",
        "description":"Azadirachta indica, commonly known as neem, is a versatile tree native to the Indian subcontinent and parts of Southeast Asia. It belongs to the mahogany family, Meliaceae. Neem is renowned for its numerous medicinal properties and has been used in traditional medicine for centuries.\n\nThe neem tree typically grows up to 15-20 meters in height and has dense foliage with dark green, pinnate leaves that are serrated along the edges. Its small, white fragrant flowers bloom in clusters, and the tree bears olive-like fruits that are yellow when ripe.\n\nVarious parts of the neem tree, including its leaves, bark, seeds, and oil, are used for medicinal, cosmetic, and agricultural purposes. Neem is known for its antibacterial, antiviral, antifungal, and antiseptic properties, making it a valuable ingredient in skincare products, herbal remedies, and agricultural pesticides.\n\nNeem oil, extracted from the seeds, is particularly famous for its insecticidal properties and is widely used in organic farming as a natural pesticide. Additionally, neem leaves are used in traditional medicine to treat various ailments such as skin disorders, diabetes, digestive issues, and infections.\n\nOverall, neem is celebrated for its multifaceted benefits and continues to be an integral part of traditional healing practices and sustainable agriculture."
    },
    "Calotropis(எருக்கு)": {
        "botanical_name": "Calotropis Gigantea",
        "tamil_name": "எருக்கு",
        "common_name": "Giant Milkweed,Apple of Sodom",
        "family": "Apocynaceae",
        "habitat": "Native to the Indian subcontinent, also found in other tropical and subtropical regions",
        "medicinal_properties": "Traditionally used to treat various ailments, including diarrhea, fever, and skin diseases. Contains cardiac glycosides and needs to be used with caution,Anti-inflammatory,Antimicrobial,Wound Healing,Analgesic (Pain Relief),Gastrointestinal Disorders,Anti-parasitic",
        "description": "Calotropis gigantea, also known as crown flower or giant milkweed, is a large shrub or small tree native to tropical regions of Asia and Africa. It belongs to the Apocynaceae family. Here's a description of Calotropis gigantea:\n\n1. **Appearance**: Calotropis gigantea typically grows up to 2-4 meters tall, though it can reach heights of up to 6 meters. It has thick, succulent stems with grayish-green, elliptical leaves arranged spirally along the stems. The leaves are leathery and may have a slightly fuzzy texture.\n\n2. **Flowers**: The most distinctive feature of Calotropis gigantea is its large, showy flowers. These flowers are star-shaped and occur in clusters at the tips of the branches. They come in shades of white, purple, or lavender, and have a sweet fragrance, attracting various pollinators such as bees and butterflies.\n\n3. **Fruits**: After flowering, Calotropis gigantea produces distinctive seed pods that are large, elongated, and covered with soft spines. When mature, the pods split open to release numerous seeds, each attached to a silky parachute-like structure, aiding in wind dispersal.\n\n4. **Cultural Significance**: In addition to its medicinal properties, Calotropis gigantea holds cultural significance in various regions where it grows. It is often used in religious ceremonies, floral decorations, and garlands, especially in South Asian countries like India.\n\nOverall, Calotropis gigantea is a striking plant with notable flowers and cultural importance, though its toxic properties warrant careful handling and usage"
    },
    "Ficus Religiosa(அரச மரம்)": {
        "botanical_name": "Ficus religiosa",
        "tamil_name": "அரச மரம்",
        "common_name": "Sacred Fig,Bodhi tree",
        "family": "Moraceae",
        "habitat": "Native to the Indian subcontinent and widely cultivated in tropical and semi-tropical regions",
        "medicinal_properties": "Used in traditional medicine to treat various disorders, including diabetes, asthma, and skin diseases. The bark and leaves have antimicrobial and anti-inflammatory properties.",
        "description":"Ficus religiosa, commonly known as the sacred fig or bodhi tree, is a revered species native to the Indian subcontinent and Southeast Asia. This large, deciduous tree can reach heights of up to 30 meters, boasting a spreading canopy adorned with heart-shaped leaves.\n\nIts distinctive aerial roots descend from the branches to form additional trunk-like structures upon reaching the ground. Producing small, purple fig fruits, it offers sustenance to various birds and animals. Culturally, Ficus religiosa holds profound significance in Hinduism, Buddhism, and Jainism, representing knowledge, enlightenment, and spiritual awakening.\n\nSacred rituals and prayers often take place beneath its shade. Ecologically, the tree plays a vital role, providing habitat and food for diverse species, while its expansive canopy offers shelter and helps regulate the local microclimate.\n\nFurthermore, its aerial roots contribute to soil stabilization and erosion control. Beyond its physical attributes, Ficus religiosa embodies spiritual beliefs and ecological harmony, serving as a timeless symbol of reverence and interconnectedness in the cultural landscape of Asia"
    },
    "Oleander(அரளி)": {
        "botanical_name": "Apocynaceae",
        "tamil_name": "அரளி",
        "common_name": "Indian Jujube,Nerium",
        "family": "Nerium indicum Linn",
        "habitat": "Native to the Indian subcontinent, also found in parts of Africa and Southeast Asia",
        "medicinal_properties": "Rich in vitamin C, antioxidants, and laxative properties. Used to treat various digestive disorders and skin diseases.",
        "description":"Nerium oleander, commonly known as oleander, is a perennial shrub or small tree native to the Mediterranean region and parts of Asia. It belongs to the dogbane family, Apocynaceae. Oleander is widely cultivated for its attractive flowers and ornamental value, but it's important to note that all parts of the plant are highly toxic if ingested.\n\nOleander typically grows 2-6 meters in height, though it can reach up to 10 meters under favorable conditions. The leaves are narrow, lance-shaped, and leathery, arranged in whorls along the stems. The flowers are showy and fragrant, occurring in clusters at the ends of branches. They come in various colors including white, pink, red, or yellow, depending on the cultivar.\n\nDespite its toxicity, oleander has historical and cultural significance. It has been used in traditional medicine, though extreme caution is advised due to its poisonous nature. In landscaping, oleander is valued for its ability to tolerate heat, drought, and poor soil conditions, making it a popular choice for coastal gardens and arid regions.\n\nOverall, while oleander is prized for its beauty and resilience, it's essential to handle it with care and be aware of its toxic properties, especially if there are children or pets present."
    }
}

# Trending Resources (7 links)
Resources = [
    {"name": "Medicinal Plants Database", "url": "https://www.mpbd.info/"},
    {"name": "Kew Science - Plants of the World", "url": "https://powo.science.kew.org/"},
    {"name": "Indian Medicinal Plants Database", "url": "https://www.medicinalplants.in/"},
    {"name": "Tamil Nadu Medicinal Plants Board", "url": "https://tnmpb.tn.gov.in/"},
    {"name": "Ayurvedic Pharmacopoeia", "url": "https://www.ayurveda.hu/api/API-Vol-1.pdf"},
    {"name": "World Health Organization - Traditional Medicine", "url": "https://www.who.int/health-topics/traditional-complementary-and-integrative-medicine"},
    {"name": "National Medicinal Plants Board", "url": "https://nmpb.nic.in/"}
]

# ---------- Main App ----------
def app():
    local_css()
    
    # Enhanced Sidebar with clear white background and dark text
    with st.sidebar:
        






        st.sidebar.markdown("<h2 style='color:black;'>🌿 Medical Plant Identifier</h2>", unsafe_allow_html=True)
        
        # About section
        st.sidebar.markdown("<h3 style='color:black;'>ℹ️ About This App</h3>", unsafe_allow_html=True)

# Supporting text
        st.sidebar.markdown("""
<b >Identify medicinal plants and discover their:</b><br>
• Botanical properties<br>
• Traditional uses<br>
• Medicinal benefits
""", unsafe_allow_html=True)
        
        # User guide
        with st.sidebar.expander("📘 How to Use", expanded=True):
            st.markdown("""
    1. Upload a clear plant image  
    2. Get AI identification  
    3. Explore detailed information  
    4. Learn medicinal uses  
    """)
        
      
        
        # Resources
        with st.expander("📚 Resources"):
            st.markdown("#### Explore these helpful resources:")
            for resource in Resources:
                st.markdown(f'<a href="{resource["url"]}" target="_blank" class="resource-link">🔗 {resource["name"]}</a>', unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
        ---
        **Contact**: [thameemansarivrn@gmail.com](mailto:thameemansarivrn@gmail.com)    
        """)

    # Main content
    st.markdown('<h1 style="text-align: center; color: #2c3e50; margin-bottom: 1.5rem;">🌱 Medical Plant Identification System</h1>', unsafe_allow_html=True)
    
    # Upload section with reduced header size
    st.markdown("""
    <div class="upload-header">
        Upload a plant image to identify medicinal properties
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload Plant Image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        numpy_image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

        results = model.predict(source=numpy_image)
        st.image(results[0].plot(), caption="🧠 YOLOv8 Model Prediction", use_column_width=True)

        predictions = results[0].boxes.xyxy.cpu().numpy()
        predicted_classes = results[0].boxes.cls.cpu().numpy().astype(int)

        predicted_class_index = predicted_classes[0]
        predicted_plant_name = class_names[predicted_class_index]
        predicted_key = list(class_details.keys())[predicted_class_index]
        plant_details = class_details[predicted_key]

        st.markdown('<div class="prediction-title">✅ Identified Plant:</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="plant-name">🌿 {predicted_plant_name}</div>', unsafe_allow_html=True)

        # Display Info Table (respecting sidebar filters)
        table_data = [
    ["Botanical Name", plant_details['botanical_name']],
    ["Tamil Name", plant_details['tamil_name']],
    ["Common Name", plant_details['common_name']],
    ["Family", plant_details['family']],
    ["Habitat", plant_details['habitat']],
    ["Medicinal Properties", plant_details['medicinal_properties']]]
        
        df = pd.DataFrame(table_data, columns=["Description", "Value"])
        st.dataframe(df, use_container_width=True, height=300)

        # Description box
        st.markdown(f'<div class="info-box"><b>📝 Description:</b><br>{plant_details["description"]}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Run App ----------
if __name__ == "__main__":
    app()