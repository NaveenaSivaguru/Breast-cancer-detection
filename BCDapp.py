import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
import pickle
from torchvision import transforms
import torch.nn.functional as F
import os
from pathlib import Path
import datetime
import streamlit as st
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pathlib
from pathlib import Path
import base64
import requests

model1_url="https://github.com/NaveenaSivaguru/Breast-cancer-detection/releases/download/v1.0.0/BC_RES1.pth"
model2_url="https://github.com/NaveenaSivaguru/Breast-cancer-detection/releases/download/v1.0.0/BGMEG_RES1.pth"

# Set page config
st.set_page_config(
    page_title="BreastScan AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


import base64
from pathlib import Path

# --- Configuration ---
class AppConfig:
    # File paths - use relative paths and place these files in your project directory
    BACKGROUND_IMAGE =os.path.join(os.getcwd(), "Background.jpg") 
    COMPANY_LOGO = os.path.join(os.getcwd(),"saslogo.jpg")         
    FIRST_STAGE_MODEL=os.path.join(os.getcwd(), "BC_RES1.pth")
    if not os.path.exists(FIRST_STAGE_MODEL):
        with st.spinner("Downloading model..."):
            response = requests.get(model1_url)
            with open(FIRST_STAGE_MODEL, "wb") as f:
                f.write(response.content)
            st.success("Model1 downloaded successfully!")
    FIRST_STAGE_ENCODER = os.path.join(os.getcwd(),"BC_3CLASS_ENCODER.pkl")
    SECOND_STAGE_MODEL = os.path.join(os.getcwd(), "BGMEG_RES1.pth")
    if not os.path.exists(SECOND_STAGE_MODEL):
        with st.spinner("Downloading model..."):
            response = requests.get(model2_url)
            with open(SECOND_STAGE_MODEL, "wb") as f:
                f.write(response.content)
            st.success("Model2 downloaded successfully!")
    SECOND_STAGE_ENCODER = os.path.join(os.getcwd(),"BC_BEMG_ENCODER.pkl")

    @staticmethod
    def get_image_as_base64(path):
        """Convert image to base64 for embedding in HTML"""
        try:
            with open(path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            return ""
    
    # Image settings
    IMAGE_SIZE = (224, 224)
    MEAN = [0.3684, 0.3684, 0.3684]
    STD = [0.3346, 0.3346, 0.3346]
    
    # Colors
    PRIMARY_COLOR = "#3498db"
    SECONDARY_COLOR = "#2c3e50"
    SUCCESS_COLOR = "#2ecc71"
    WARNING_COLOR = "#f39c12"
    DANGER_COLOR = "#e74c3c"
    
    @staticmethod
    def verify_paths():
        """Verify that required files exist."""
        required_files = [
            AppConfig.BACKGROUND_IMAGE,
            AppConfig.COMPANY_LOGO,
            AppConfig.FIRST_STAGE_MODEL,
            AppConfig.FIRST_STAGE_ENCODER,
            AppConfig.SECOND_STAGE_MODEL,
            AppConfig.SECOND_STAGE_ENCODER
        ]
        
        missing_files = [f for f in required_files if not Path(f).exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")

# --- CSS Styling ---
def set_custom_css():
    bg_image = AppConfig.get_image_as_base64(AppConfig.BACKGROUND_IMAGE)
    logo_image = AppConfig.get_image_as_base64(AppConfig.COMPANY_LOGO)
    
    css = f"""
    <style>
        /* Main background */
        .stApp {{
            background-image: url("data:image/png;base64,{bg_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        /* Full-width logo banner */
        .logo-banner {{
            width: 100%;
            margin: 0;
            padding: 0;
            display: block;
            background-color: white;  /* Fallback if logo has transparency */
        }}
        
        .logo-img {{
            width: 100%;
            height: auto;
            max-height: 200px;  /* Adjust this value as needed */
            object-fit: contain;
        }}
        
        /* Title box */
        .title-box {{
            background-color: #1E90FF;
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            margin: 1rem auto 2rem auto;
            max-width: 800px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        
        .title {{
            font-size: 2.5rem;
            font-weight: bold;
            margin: 0;
        }}
        
        /* Upload section */
        .upload-section {{
            background-color: rgba(255, 255, 255, 0.9);
            padding: 2rem;
            border-radius: 15px;
            margin: 0 auto;
            max-width: 800px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}

        
        /* Card styling */
        .card {{
            background-color: rgba(255, 255, 255, 0.98);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border-left: 4px solid {AppConfig.PRIMARY_COLOR};
        }}
        
        .card-normal {{
            border-left: 4px solid {AppConfig.SUCCESS_COLOR};
            background-color: rgba(46, 204, 113, 0.08);
        }}
        
        .card-abnormal {{
            border-left: 4px solid {AppConfig.DANGER_COLOR};
            background-color: rgba(231, 76, 60, 0.08);
        }}
        
        .card-info {{
            border-left: 4px solid {AppConfig.WARNING_COLOR};
            background-color: rgba(241, 196, 15, 0.08);
        }}
        
        /* Title within cards */
        .card-title {{
            color: {AppConfig.SECONDARY_COLOR};
            border-bottom: 2px solid {AppConfig.PRIMARY_COLOR};
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
            font-size: 1.3rem;
        }}
        
        /* Button styling */
        .stButton>button {{
            background-color: {AppConfig.PRIMARY_COLOR};
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: bold;
            border: none;
            transition: all 0.3s ease;
        }}
        
        .stButton>button:hover {{
            background-color: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        /* Custom progress bar */
        .progress-container {{
            width: 100%;
            background-color: #e0e0e0;
            border-radius: 10px;
            margin: 10px 0;
        }}
        
        .progress-bar {{
            border-radius: 10px;
            height: 24px;
            text-align: center;
            color: white;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: width 0.5s ease;
        }}
        
        /* Responsive adjustments */
        @media (max-width: 768px) {{
            .header-container {{
                flex-direction: column;
                text-align: center;
            }}
            
            .logo {{
                margin-right: 0;
                margin-bottom: 1rem;
            }}
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def create_header():
    logo_image = AppConfig.get_image_as_base64(AppConfig.COMPANY_LOGO)
    
    st.markdown(f"""
    <div class="logo-banner">
        <img src="data:image/png;base64,{logo_image}" class="logo-img" alt="Company Logo">
    </div>
    
    <div class="title-box">
        <h1 class="title">Breast Cancer Detection System</h1>
    </div>
    """, unsafe_allow_html=True)

# --- Model Classes ---
class ResNet18WithGradCAM(nn.Module):
    """Custom ResNet18 model with Grad-CAM capabilities."""
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # Grad-CAM storage
        self.feature_maps = None
        self.gradients = None
        self.hook_forward = None
        self.hook_backward = None

    def save_feature_maps(self, module, input, output):
        """Hook to save feature maps during forward pass."""
        self.feature_maps = output

    def save_gradients(self, module, grad_input, grad_output):
        """Hook to save gradients during backward pass."""
        self.gradients = grad_output[0]

    def forward(self, x, enable_gradcam=False):
        """Forward pass with optional Grad-CAM hooks."""
        if enable_gradcam:
            if self.hook_forward is None:
                self.hook_forward = self.model.layer4.register_forward_hook(self.save_feature_maps)
            if self.hook_backward is None:
                self.hook_backward = self.model.layer4.register_full_backward_hook(self.save_gradients)
        else:
            if self.hook_forward is not None:
                self.hook_forward.remove()
                self.hook_forward = None
            if self.hook_backward is not None:
                self.hook_backward.remove()
                self.hook_backward = None

        return self.model(x)

    def get_cam_features_and_grads(self):
        """Get stored features and gradients for Grad-CAM."""
        return self.feature_maps, self.gradients

# --- Utility Functions ---
@st.cache_resource
def load_model(model_path, encoder_path):
    """Load a model and its corresponding encoder."""
    try:
        # Load encoder
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        class_names = list(encoder.classes_)
        
        # Load model
        model = ResNet18WithGradCAM(num_classes=len(class_names))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def load_and_resize(image):
    """Load and resize an image to the required dimensions."""
    image = image.convert('RGB')
    return image.resize(AppConfig.IMAGE_SIZE)

def predict_with_gradcam(model, image_tensor, original_image, class_names):
    """Perform prediction with Grad-CAM visualization."""
    # Convert to NumPy array (HxWxC) in BGR format
    original_np = np.array(original_image)
    
    # Remove alpha channel if present
    if original_np.shape[2] == 4:
        original_np = original_np[:, :, :3]
    
    original_np = original_np[:, :, ::-1].copy()  # Convert RGB to BGR

    # Hook to capture gradients and features
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks to final conv layer
    target_layer = model.model.layer4[-1]
    forward = target_layer.register_forward_hook(forward_hook)
    backward = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(image_tensor)
    probs = F.softmax(output, dim=1)
    pred_class = output.argmax(dim=1).item()
    prob_score = probs[0][pred_class].item()

    # Backward pass for Grad-CAM
    model.zero_grad()
    output[0, pred_class].backward()

    # Grad-CAM calculation
    grad = gradients[0].detach().cpu()[0]
    act = activations[0].detach().cpu()[0]
    weights = grad.mean(dim=(1, 2), keepdim=True)
    gradcam = (weights * act).sum(dim=0).clamp(min=0)

    # Normalize and resize Grad-CAM
    gradcam -= gradcam.min()
    gradcam /= gradcam.max()
    gradcam = cv2.resize(gradcam.numpy(), (original_np.shape[1], original_np.shape[0]))

    # Convert to uint8 and apply color map
    gradcam_uint8 = np.uint8(255 * gradcam)
    heatmap = cv2.applyColorMap(gradcam_uint8, cv2.COLORMAP_JET)
    
    # Resize heatmap to exactly match original image dimensions
    heatmap = cv2.resize(heatmap, (original_np.shape[1], original_np.shape[0]))
    
    # Overlay heatmap
    overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)

    # Release hooks
    forward.remove()
    backward.remove()

    return class_names[pred_class], prob_score, overlay[:, :, ::-1]  # Convert BGR to RGB

def split_image(image):
    """Split an image into left and right halves."""
    width = image.size[0]
    left = image.crop((0, 0, width//2, image.size[1]))
    right = image.crop((width//2, 0, width, image.size[1]))
    return left, right

def create_progress_bar(probability):
    """Create a styled progress bar for confidence scores."""
    if probability < 0.3:
        color = AppConfig.SUCCESS_COLOR
    elif probability < 0.7:
        color = AppConfig.WARNING_COLOR
    else:
        color = AppConfig.DANGER_COLOR
    
    return f"""
    <div class="progress-container">
        <div class="progress-bar" style="width: {probability*100}%; background-color: {color};">
            {probability*100:.1f}% Confidence
        </div>
    </div>
    """

def create_result_card(title, prediction, probability, card_type="normal"):
    """Create a styled result card WITHOUT stray HTML tags"""
    if probability < 0.3:
        color = "#2ecc71"  # Green
    elif probability < 0.7:
        color = "#f39c12"  # Orange
    else:
        color = "#e74c3c"  # Red
    
    return f"""
    <div class="card card-{card_type}">
        <h3 class="card-title">{title}</h3>
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <div style="font-size: 1.3rem; font-weight: bold; margin-right: 1.5rem;">
                {prediction}
            </div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {probability*100}%; background-color: {color};">
                    {probability*100:.1f}% Confidence
                </div>
            </div>
        </div>
    """  # Removed the closing </div> that was here before



# --- Main Application ---
def main():
    # Set up CSS and header
    set_custom_css()
    create_header()
    
    try:
        # Verify required files exist
        AppConfig.verify_paths()
        
        
        # Load models
        with st.spinner("🔬 Loading AI models... Please wait"):
            first_stage_model, first_class_names = load_model(
                AppConfig.FIRST_STAGE_MODEL, 
                AppConfig.FIRST_STAGE_ENCODER
            )
            second_stage_model, second_class_names = load_model(
                AppConfig.SECOND_STAGE_MODEL,
                AppConfig.SECOND_STAGE_ENCODER
            )
        
        # Image upload section
        with st.container():
            st.markdown("""
            <div class="upload-container">
                <h2 style="color: #2c3e50; margin-top: 0;">Upload Medical Image</h2>
                <p style="color: #7f8c8d;">Please upload a mammogram or breast ultrasound image for analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose an image...", 
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    
                    # Display original image
                    st.subheader("📷 Uploaded Image Preview")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.image(image, caption="Original Image", use_container_width=True)
                    
                    # Analysis settings
                    with col2:
                        st.subheader("⚙️ Analysis Settings")
                        is_split = st.radio(
                            "Image Type:",
                            ("Full mammogram (will be split)", "Already cropped (single image)"),
                            index=0
                        )
                        show_gradcam = st.checkbox("Show AI attention heatmap (Grad-CAM)", value=True)
                        st.info("ℹ️ The heatmap shows which areas influenced the AI's decision the most")
                        
                        if st.button("🔍 Analyze Image", use_container_width=True, type="primary"):
                            with st.spinner("🧠 Analyzing image with AI... This may take a moment"):
                                transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=AppConfig.MEAN, std=AppConfig.STD)
                                ])
                                
                                if "Already cropped" in is_split:
                                    # Process as single image
                                    original = load_and_resize(image)
                                    image_tensor = transform(original).unsqueeze(0).to(device)
                                    
                                    # First stage prediction
                                    first_pred, first_prob, first_overlay = predict_with_gradcam(
                                        first_stage_model, image_tensor, original, first_class_names
                                    )
                                    
                                    # Display results
                                    st.markdown("---")
                                    st.markdown("""
                                    <div style="background-color: #3498db; padding: 1rem; border-radius: 10px;">
                                        <h2 style="color: white; margin: 0;">🔬 Analysis Results</h2>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # First stage results
                                    card_type = "normal" if first_pred.lower() == 'normal' else "abnormal"
                                    st.markdown(create_result_card(
                                        "First Stage Classification",
                                        first_pred,
                                        first_prob,
                                        card_type
                                    ), unsafe_allow_html=True)
                                    
                                    if show_gradcam:
                                        st.image(first_overlay, 
                                                caption=f"AI Attention Map: {first_pred} ({first_prob*100:.1f}% confidence)", 
                                                use_container_width=True)
                                    
                                    # Second stage if abnormal
                                    if first_pred.lower() == 'abnormal':
                                        second_pred, second_prob, second_overlay = predict_with_gradcam(
                                            second_stage_model, image_tensor, original, second_class_names
                                        )
                                        
                                        st.markdown(create_result_card(
                                            "Second Stage Classification",
                                            second_pred,
                                            second_prob,
                                            "abnormal"
                                        ), unsafe_allow_html=True)
                                        
                                        if show_gradcam:
                                            st.image(second_overlay, 
                                                    caption=f"AI Attention Map: {second_pred} ({second_prob*100:.1f}% confidence)", 
                                                    use_container_width=True)
                                    else:
                                        st.markdown("""
                                        <div class="card card-info">
                                            <h3 class="card-title">Additional Information</h3>
                                            <p>No further analysis needed as image is classified as Normal</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    # Process as full image (split into halves)
                                    left_img, right_img = split_image(image)
                                    
                                    # Process left side
                                    left_original = load_and_resize(left_img)
                                    left_tensor = transform(left_original).unsqueeze(0).to(device)
                                    left_first_pred, left_first_prob, left_first_overlay = predict_with_gradcam(
                                        first_stage_model, left_tensor, left_original, first_class_names
                                    )
                                    
                                    # Process right side
                                    right_original = load_and_resize(right_img)
                                    right_tensor = transform(right_original).unsqueeze(0).to(device)
                                    right_first_pred, right_first_prob, right_first_overlay = predict_with_gradcam(
                                        first_stage_model, right_tensor, right_original, first_class_names
                                    )
                                    
                                    # Display results
                                    st.markdown("---")
                                    st.markdown("""
                                    <div style="background-color: #3498db; padding: 1rem; border-radius: 10px;">
                                        <h2 style="color: white; margin: 0;">🔬 Bilateral Analysis Results</h2>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Left side results
                                    st.subheader("🫲 RIGHT Breast Analysis")
                                    col1, col2 = st.columns([1, 1])
                                    
                                    with col1:
                                        st.image(left_original, caption="RIGHT Side", use_container_width=True)
                                    
                                    with col2:
                                        if show_gradcam:
                                            st.image(left_first_overlay, 
                                                    caption="AI Attention Map", 
                                                    use_container_width=True)
                                    
                                    left_card_type = "normal" if left_first_pred.lower() == 'normal' else "abnormal"
                                    st.markdown(create_result_card(
                                        "RIGHT Breast - First Stage",
                                        left_first_pred,
                                        left_first_prob,
                                        left_card_type
                                    ), unsafe_allow_html=True)
                                    
                                    if left_first_pred.lower() == 'abnormal':
                                        left_second_pred, left_second_prob, left_second_overlay = predict_with_gradcam(
                                            second_stage_model, left_tensor, left_original, second_class_names
                                        )
                                        
                                        st.markdown(create_result_card(
                                            "RIGHT Breast - Second Stage",
                                            left_second_pred,
                                            left_second_prob,
                                            "abnormal"
                                        ), unsafe_allow_html=True)
                                        
                                        if show_gradcam:
                                            st.image(left_second_overlay, 
                                                    caption=f"AI Attention Map: {left_second_pred}", 
                                                    use_container_width=True)
                                    
                                    # Right side results
                                    st.subheader("🫱 LEFT Breast Analysis")
                                    col1, col2 = st.columns([1, 1])
                                    
                                    with col1:
                                        st.image(right_original, caption="LEFT Side", use_container_width=True)
                                    
                                    with col2:
                                        if show_gradcam:
                                            st.image(right_first_overlay, 
                                                    caption="AI Attention Map", 
                                                    use_container_width=True)
                                    
                                    right_card_type = "normal" if right_first_pred.lower() == 'normal' else "abnormal"
                                    st.markdown(create_result_card(
                                        "LEFT Breast - First Stage",
                                        right_first_pred,
                                        right_first_prob,
                                        right_card_type
                                    ), unsafe_allow_html=True)
                                    
                                    if right_first_pred.lower() == 'abnormal':
                                        right_second_pred, right_second_prob, right_second_overlay = predict_with_gradcam(
                                            second_stage_model, right_tensor, right_original, second_class_names
                                        )
                                        
                                        st.markdown(create_result_card(
                                            "LEFT Breast - Second Stage",
                                            right_second_pred,
                                            right_second_prob,
                                            "abnormal"
                                        ), unsafe_allow_html=True)
                                        
                                        if show_gradcam:
                                            st.image(right_second_overlay, 
                                                    caption=f"AI Attention Map: {right_second_pred}", 
                                                    use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                
                # Disclaimer
                st.markdown("---")
                st.warning("""
                **Disclaimer:** This AI system is designed to assist healthcare professionals and should not be used as a sole diagnostic tool. 
                Always consult with a qualified radiologist or physician for clinical decisions.
                """)
                
                # Footer
                st.markdown("---")
                st.markdown(f"""
                <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem; margin-top: 2rem;">
                    <p>© {datetime.datetime.now().year} BreastScan AI | Powered by Deep Learning</p>
                </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.stop()

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run the app
    main()
