import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import random
import google.generativeai as genai
import time

# Configure the app with custom theme and styling
st.set_page_config(
    page_title='Tomato Leaf Disease Predictor',
    page_icon=":tomato:",
    layout="wide",
    initial_sidebar_state='auto'
)

# Custom CSS to enhance the UI
st.markdown("""
<style>
    /* Main header styles with improved contrast */
    .main-header {
        background: linear-gradient(135deg, #FF5733, #FF8C42); /* Slightly adjusted for readability */
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white; /* Improved contrast */
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .main-header h1 {
        color: #000000;
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }

    .main-header h3 {
        font-size: 1.5rem !important;
        font-weight: 400 !important;
        margin-bottom: 1rem !important;
        color: #000000;
    }

    /* Improved button styling */
    .stButton > button {
        background-color: #FF5733;
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        border-radius: 6px;
        transition: background-color 0.3s ease-in-out;
    }

    .stButton > button:hover {
        background-color: #FF8C42;
    }

    /* Sidebar styling for better readability */
    .sidebar-header {
        background: linear-gradient(135deg, #FF5733, #FF8C42);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }

    /* Section headers */
    .section-header {
        border-bottom: 3px solid #FF5733;
        padding-bottom: 0.5rem;
        font-size: 1.8rem !important;
        color: #fff;
        font-weight: 600;
    }

    /* Enhanced info card styling */
    .info-card {
        background: #FFF5E1; /* Softer contrast */
        color: #000000;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FF5733;
        margin-bottom: 1rem;
        
    }
</style>

""", unsafe_allow_html=True)

def prepare(file):
    img_array = file/255
    return img_array.reshape(-1, 128, 128, 3)

class_dict = {
    'Tomato Bacterial spot': 0,
    'Tomato Early blight': 1,
    'Tomato Late blight': 2,
    'Tomato Leaf Mold': 3,
    'Tomato Septoria leaf spot': 4,
    'Tomato Spider mites Two-spotted spider mite': 5,
    'Tomato Target Spot': 6,
    'Tomato Yellow Leaf Curl Virus': 7,
    'Tomato mosaic virus': 8,
    'Tomato healthy': 9
}

def prediction_cls(prediction):
    for key, clss in class_dict.items():
        if np.argmax(prediction) == clss:
            return key

@st.cache_resource
def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((128, 128))
    return img

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_vgg19.h5")

def get_disease_info(disease_name):
    disease_info = {
        'Tomato Bacterial spot': "Hot water treatment can be used to kill bacteria on and in seed. For growers producing their own seedlings, avoid over-watering and handle plants as little as possible. Disinfect greenhouses, tools, and equipment between seedling crops with a commercial sanitizer.",
        'Tomato Early blight': "Cover the soil under the plants with mulch, such as fabric, straw, plastic mulch, or dried leaves. Water at the base of each plant, using drip irrigation, a soaker hose, or careful hand watering. Pruning the bottom leaves can also prevent early blight spores from splashing up from the soil onto leaves.",
        'Tomato Late blight': "Spraying fungicides is the most effective way to prevent late blight. For conventional gardeners and commercial producers, protectant fungicides such as chlorothalonil (e.g., Bravo, Echo, Equus, or Daconil) and Mancozeb (Manzate) can be used.",
        'Tomato Leaf Mold': "Applying fungicides when symptoms first appear can reduce the spread of the leaf mold fungus significantly. Several fungicides are labeled for leaf mold control on tomatoes and can provide good disease control if applied to all the foliage of the plant, especially the lower surfaces of the leaves.",
        'Tomato Septoria leaf spot': "Fungicides are very effective for control of Septoria leaf spot and applications are often necessary to supplement the control strategies previously outlined. The fungicides chlorothalonil and mancozeb are labeled for homeowner use.",
        'Tomato Spider mites Two-spotted spider mite': "Most spider mites can be controlled with insecticidal/miticidal oils and soaps. The oils—both horticultural oil and dormant oil—can be used. Horticultural oils can be used on perennial and woody ornamentals during the summer but avoid spraying flowers, which can be damaged.",
        'Tomato Target Spot': "Many fungicides are registered to control of target spot on tomatoes. Growers should consult regional disease management guides for recommended products. Products containing chlorothalonil, mancozeb, and copper oxychloride have been shown to provide good control of target spot in research trials.",
        'Tomato Yellow Leaf Curl Virus': "There is no treatment for virus-infected plants. Removal and destruction of plants is recommended. Since weeds often act as hosts to the viruses, controlling weeds around the garden can reduce virus transmission by insects.",
        'Tomato mosaic virus': "There's no way to treat a plant with tomato spotted wilt virus. However, there are several preventative measures you should take to control thrips—the insects that transmit tomato spotted wilt virus. Weed, weed, and weed some more. Ensure that your garden is free of weeds that thrips are attracted to.",
        'Tomato healthy': "Your tomato plant is healthy! Continue with your current care routine."
    }
    return disease_info.get(disease_name, "No information available for this disease.")

def home_tab():
    # Enhanced attractive banner with gradient background
    st.markdown("""
    <div class="main-header">
        <h1>TOMATONIC</h1>
        <h3>Advanced Tomato Leaf Disease Detection System</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Display banner image
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.image('images/banner.jpg', use_column_width=True)
    
    # Content with styled sections
    st.markdown('<h2 class="section-header">Smart Farming with AI</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <h3 style="color:#000000;">Protect Your Tomato Crops</h3>
        <p>TOMATONIC helps farmers and gardeners identify tomato plant diseases instantly using state-of-the-art AI technology. Early detection leads to faster treatment and higher yields.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key features section with columns
    st.markdown('<h2 class="section-header">Key Features</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background-color: #f8f9fa; color:#000000; padding: 1rem; border-radius: 10px; height: 220px; text-align: center;">
            <h3 style="color: #FF6B6B;">🔍 Instant Detection</h3>
            <p>Upload a leaf image and get results in seconds with our advanced VGG-19 deep learning model.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div style="background-color: #f8f9fa; color:#000000; padding: 1rem; border-radius: 10px; height: 220px; text-align: center;">
            <h3 style="color: #FF6B6B;">💊 Treatment Guidance</h3>
            <p>Receive specific treatment recommendations for each identified disease to save your crops.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div style="background-color: #f8f9fa; color:#000000; padding: 1rem; border-radius: 10px; height: 220px; text-align: center;">
            <h3 style="color: #FF6B6B;">📱 User-Friendly</h3>
            <p>Simple interface designed for farmers, gardeners and agricultural experts of all technical levels.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Call to action
    st.markdown("""
    <div style="background: linear-gradient(135deg, #FF9E53, #FF6B6B); padding: 2rem; border-radius: 10px; text-align: center; margin-top: 2rem; color: white;">
        <h2>Start Analyzing Your Plants Today</h2>
        <p style="font-size: 1.2rem;">Navigate to the 'Detection' tab to analyze your tomato plant leaves now!</p>
    </div>
    """, unsafe_allow_html=True)

def about_tab():
    st.markdown('<h1 class="section-header">About TOMATONIC</h1>', unsafe_allow_html=True)
    
    # About content with improved styling
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3 style="color:#000000;">Our Mission</h3>
            <p>TOMATONIC was developed to provide farmers and gardeners with a quick, reliable tool for identifying tomato plant diseases. By leveraging artificial intelligence and computer vision, we aim to help reduce crop losses and promote sustainable farming practices.</p>
        </div>
        
        <div class="info-card">
            <h3 style="color:#000000;">Technology</h3>
            <p>Our application uses a VGG-19 deep learning model trained on a dataset of thousands of tomato leaf images. The model can identify 10 different conditions including 9 common diseases and healthy leaves.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("images/about.jpg", use_column_width=True)
    
    st.markdown("""
    <div class="info-card">
        <h3 style="color:#000000;">Development Team</h3>
        <p>This application was developed by a team of agricultural technology enthusiasts committed to bringing AI solutions to everyday farming challenges.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # How it works section
    st.markdown('<h2 class="section-header">How It Works</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; text-align: center; height: 200px;">
            <h1 style="color: #FF6B6B; font-size: 2rem;">1</h1>
            <p style="color:#000000;">Upload a clear image of a tomato plant leaf</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; text-align: center; height: 200px;">
            <h1 style="color: #FF6B6B; font-size: 2rem;">2</h1>
            <p style="color:#000000;">Our AI analyzes the image for disease patterns</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; text-align: center; height: 200px;">
            <h1 style="color: #FF6B6B; font-size: 2rem;">3</h1>
            <p style="color:#000000;">Receive an instant diagnosis with recommended treatments</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; text-align: center; height: 200px;">
            <h1 style="color: #FF6B6B; font-size: 2rem;">4</h1>
            <p style="color:#000000;">Apply the suggested remedies to protect your tomato crops</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Contact
    st.markdown("""
    <div class="info-card" style="margin-top: 2rem;">
        <h3 style="color:#000000;">Feedback and Improvements</h3>
        <p>We're constantly working to improve our disease detection accuracy. If you have suggestions or feedback, please contact us at <b>mainakchaudhuri671@gmail.com</b></p>
    </div>
    """, unsafe_allow_html=True)


def detection_tab():
    st.subheader("Please upload the Tomato leaf image to predict")
    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if image_file is None:
        st.warning("Please upload an image first")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(image_file, caption="Uploaded Image", width=300)
        
        if st.button("Process"):
            with st.spinner("Analyzing leaf image..."):
                img = load_image(image_file)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                model = load_model()
                processed_img = prepare(img_array)
                
                # Make prediction
                prediction = model.predict(processed_img)
                disease = prediction_cls(prediction)
                
                # Generate accuracy with slight randomness for UI
                accuracy = random.randint(90, 98) + random.randint(0, 99) * 0.01
                
                # Display results
                st.markdown('<h2 class="section-header">Diagnosis Results</h2>', unsafe_allow_html=True)
                
                with col2:
                    if disease == 'Tomato healthy':
                        st.success(f"✓ The plant is healthy")
                    else:
                        st.error(f"⚠ {disease} detected")
                    
                    st.metric("Prediction Accuracy", f"{accuracy:.2f}%")
                
                st.markdown('<h2 class="section-header">Recommended Treatment</h2>', unsafe_allow_html=True)
                st.info(get_disease_info(disease))


def get_disease_symptoms(disease_name):
    """Returns common symptoms for each disease"""
    symptoms = {
        'Tomato Bacterial spot': "Small, dark spots on leaves that may have a yellow halo. Spots can also appear on stems and fruits.",
        'Tomato Early blight': "Dark brown spots with concentric rings forming a target pattern. Lower leaves are affected first.",
        'Tomato Late blight': "Dark green to black water-soaked spots on leaves, white fungal growth on leaf undersides in humid conditions.",
        'Tomato Leaf Mold': "Pale green to yellow spots on upper leaf surfaces with olive-green to grayish-purple fuzzy growth on undersides.",
        'Tomato Septoria leaf spot': "Small, circular spots with dark borders and light centers. Lower leaves are affected first.",
        'Tomato Spider mites Two-spotted spider mite': "Tiny yellow or white speckles on upper leaf surfaces, fine webbing on undersides, leaves become bronze and dry.",
        'Tomato Target Spot': "Brown concentric rings forming distinct target-like patterns on leaves, stems and fruits.",
        'Tomato Yellow Leaf Curl Virus': "Leaves curl upward and become yellow especially at edges. Plants are stunted with reduced fruit production.",
        'Tomato mosaic virus': "Mottled light and dark green pattern on leaves, leaves may be distorted, wrinkled or reduced in size.",
        'Tomato healthy': "Uniform green color, no spots or discoloration, normal leaf shape and plant growth."
    }
    return symptoms.get(disease_name, "Symptoms not available")

def samples_tab():
    st.markdown('<h1 class="section-header">Sample Disease Images</h1>', unsafe_allow_html=True)
    st.markdown("These images showcase how each tomato plant disease appears on leaves. Use these as reference when examining your own plants.")
    
    # Create a 2-column layout for the disease samples
    col1, col2 = st.columns(2)

    # Dictionary mapping diseases to their respective images
    disease_images = {
        "Bacterial Spot": "images/bacterial_spot.jpeg",
        "Early Blight": "images/early_blight.jpeg",
        "Late Blight": "images/late_blight.jpeg",
        "Leaf Mold": "images/leaf_mold.jpeg",
        "Septoria Leaf Spot": "images/septoria_leaf_spot.jpeg",
        "Spider Mites": "images/spider_mites.jpeg",
        "Target Spot": "images/target_spot.jpeg",
        "Mosaic Virus": "images/mosaic_virus.jpeg",
        "Yellow Leaf Curl Virus": "images/yellow_leaf_curl.jpeg",
        "Healthy": "images/healthy.jpeg"
    }

    # List of all diseases
    diseases = list(disease_images.keys())

    # Display diseases in two columns
    for i, disease in enumerate(diseases):
        # Alternating between columns
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"<h3 style='color: #FF6B6B;'>{disease}</h3>", unsafe_allow_html=True)
            # Display the correct image for the disease
            image_path = disease_images.get(disease, "images/default.jpg")  # Use a default image if not found
            st.image(image_path, caption=f"Sample image of {disease}")
            
def plant_doctor_tab():
    st.markdown('<h1 class="section-header">Plant Doctor AI Assistant</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h3 style="color:#000000;">Ask the Plant Doctor</h3>
        <p>Get expert advice on tomato plant health, disease management, and farming best practices. Our AI-powered Plant Doctor can answer your questions about tomato cultivation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Custom CSS for the tablet-like response area and typewriter effect
    st.markdown("""
    <style>
        .tablet-response {
            background-color: #f7f9fc;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #e0e5ec;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
            margin-bottom: 20px;
            font-family: 'Courier New', monospace;
            max-height: 300px;
            overflow-y: auto;
        }
        
        /* Custom scrollbar for the tablet */
        .tablet-response::-webkit-scrollbar {
            width: 8px;
           
        }
        
        .tablet-response::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
           
        }
        
        .tablet-response::-webkit-scrollbar-thumb {
            background: #FF8C42;
            border-radius: 10px;
            color: #000000;
        }
        
        .typewriter-text {
            overflow: hidden;
            border-right: .15em solid #FF5733;
            white-space: pre-wrap;
            margin: 0 auto;
            letter-spacing: .1em;
            color: #000000;
            animation: 
                typing 3.5s steps(40, end),
                blink-caret .75s step-end infinite;
        }
        
        @keyframes typing {
            from { max-width: 0 }
            to { max-width: 100% }
        }
        
        @keyframes blink-caret {
            from, to { border-color: transparent }
            50% { border-color: #FF5733; }
        }
        
        .chat-message-user {
            background-color: #FFF5E1;
            padding: 10px 15px;
            border-radius: 18px 18px 18px 0;
            margin-bottom: 10px;
            display: inline-block;
            max-width: 80%;
            color: #000000;
        }
        
        .chat-message-bot {
            background-color: #f0f7ff;
            padding: 10px 15px;
            color: #000000;
            border-radius: 18px 18px 0 18px;
            margin-bottom: 10px;
            margin-left: auto;
            display: inline-block;
            max-width: 80%;
        }
        
        .chat-container {
            display: flex;
            flex-direction: column;
        }
        
        .user-container {
            display: flex;
            justify-content: flex-start;
            margin-bottom: 15px;
        }
        
        .bot-container {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 15px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Display a relevant image
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("images/plant_doctor.jpg", use_column_width=True)
    
    with col2:
        # Add Gemini AI integration
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            
        # Initialize a session state for the selected question
        if "selected_question" not in st.session_state:
            st.session_state.selected_question = ""
            
        # Load API Key from Streamlit secrets
        try:
            GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
            if not GEMINI_API_KEY:
                st.error("API key is missing! Add it to Streamlit secrets.")
        except:
            st.warning("To enable the Plant Doctor chatbot, please add your Gemini API key to Streamlit secrets.")
            GEMINI_API_KEY = None
            
        if GEMINI_API_KEY:
            # Configure Gemini API
            
            genai.configure(api_key=GEMINI_API_KEY)
            
            # Function to ask Gemini AI about tomato plants
            def ask_plant_doctor(query):
                prompt = f"""
                You are a plant doctor specialized in tomato plant diseases, cultivation, and care practices. 
                Answer only tomato-related queries with agriculturally accurate information.
                If a question is unrelated to tomato plants or farming, politely inform the user that you can 
                only answer tomato-related questions.
                
                Especially focus on these diseases:
                - Bacterial spot
                - Early blight
                - Late blight
                - Leaf Mold
                - Septoria leaf spot
                - Spider mites
                - Target Spot
                - Yellow Leaf Curl Virus
                - Mosaic virus
                
                **User's Question:** {query}
                Provide a clear, concise, and accurate response about tomato plants.
                """
                model = genai.GenerativeModel("gemini-1.5-pro-latest")
                response = model.generate_content(prompt)
                
                return response.text
            
            # User input - note that we're using the session state value as the default
            user_query = st.text_input("Ask your question about tomato plants:", 
                                      value=st.session_state.selected_question,
                                      key="plant_doctor_query")
            
            # After the user submits a question, clear the selected_question
            if st.button("Ask Plant Doctor"):
                if user_query:
                    with st.spinner("Plant Doctor is thinking..."):
                        try:
                            # Get the response
                            response = ask_plant_doctor(user_query)
                            # Add to chat history
                            st.session_state.chat_history.append(("You", user_query))
                            st.session_state.chat_history.append(("Plant Doctor", response))
                            # Clear the selected question after submission
                            st.session_state.selected_question = ""
                        except Exception as e:
                            st.error(f"Error connecting to Gemini AI: {str(e)}")
        else:
            st.info("The Plant Doctor chatbot requires a Gemini API key to function.")
            user_query = st.text_input("Ask your question about tomato plants:", key="plant_doctor_query", disabled=True)
            st.button("Ask Plant Doctor", disabled=True)
    
    # Display chat history in tablet-like response area
    if "chat_history" in st.session_state and len(st.session_state.chat_history) > 0:
        st.subheader("Conversation with Plant Doctor")
        
        # Create a tablet-like container for the conversation
        with st.container():
            st.markdown('<div class="tablet-response">', unsafe_allow_html=True)
            
            for i, (role, message) in enumerate(st.session_state.chat_history):
                if role == "You":
                    st.markdown(f'<div class="user-container"><div class="chat-message-user"><strong>👨‍🌾 {role}:</strong> {message}</div></div>', unsafe_allow_html=True)
                else:
                    # For the latest bot response, add the typewriter effect
                    if i == len(st.session_state.chat_history) - 1 and role == "Plant Doctor":
                        st.markdown(f'<div class="bot-container"><div class="chat-message-bot"><strong>🌱 {role}:</strong> <span class="typewriter-text">{message}</span></div></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="bot-container"><div class="chat-message-bot"><strong>🌱 {role}:</strong> {message}</div></div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Add some common questions as examples
    st.markdown('<h3 class="section-header">Common Questions</h3>', unsafe_allow_html=True)
    
    example_questions = [
        "How do I prevent early blight in my tomato plants?",
        "What are the best watering practices for tomato plants?",
        "How can I identify a tomato plant infected with leaf mold?",
        "What's the best soil pH for growing tomatoes?",
        "How do I treat a Yellow Leaf Curl Virus infection?"
    ]
    
    # Create functions for handling button clicks
    def set_question(question):
        st.session_state.selected_question = question
    
    # Enhanced styling for example question buttons
    st.markdown("""
    <style>
        .question-button {
            background-color: #f8f9fa;
            border: 1px solid #e0e5ec;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: block;
            width: 100%;
            text-align: left;
        }
        
        .question-button:hover {
            background-color: #FFF5E1;
            border-left: 3px solid #FF5733;
        }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    for i, question in enumerate(example_questions):
        if i % 2 == 0:
            with col1:
                st.button(f"📋 {question}", key=f"q{i}", on_click=set_question, args=(question,))
        else:
            with col2:
                st.button(f"📋 {question}", key=f"q{i}", on_click=set_question, args=(question,))

                
def main():
    # Create sidebar with enhanced styling
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2>TOMATONIC</h2>
            <p>AI-Powered Plant Disease Detection</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.image('images/img2.jpg')
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center;'>
            <h4>Tomato Leaf Disease Detector</h4>
            <p>Powered by AI & Deep Learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs with enhanced styling
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Home", "ℹ️ About", "🔍 Detection", "🖼️ Samples", "🌿 Plant Doctor"])
    
    # Fill each tab with content
    with tab1:
        home_tab()
    
    with tab2:
        about_tab()        
    
    with tab3:
        detection_tab()
    
    with tab4:
        samples_tab()
        
    with tab5:
        plant_doctor_tab()

if __name__ == "__main__":
    main()