import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import cv2
import io
import os
import torch
from torchvision.transforms.functional import rgb_to_grayscale
import zipfile
import base64
import hashlib # For hash generation

# Import rembg for background removal
import rembg

# 🔧 Patch basicsr to fix torchvision import error - simplified
import sys
import types
sys.modules['torchvision.transforms.functional_tensor'] = types.SimpleNamespace(
    rgb_to_grayscale=rgb_to_grayscale
)

# Import the necessary components from basicsr and realesrgan
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from basicsr.utils.download_util import load_file_from_url

# --- Constants ---
MODEL_NAME = 'RealESRGAN_x4plus.pth'
MODEL_URL = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'

# Global variable to store device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Helper Functions ---

@st.cache_resource
def load_model():
    """
    Loads the Real-ESRGAN model. Downloads it if not present locally.
    Caches the model to prevent re-loading on every Streamlit rerun.
    """
    if not os.path.exists(MODEL_NAME):
        st.info(f"Downloading model file '{MODEL_NAME}'. This may take a moment...", icon="⏳")
        try:
            model_path = load_file_from_url(url=MODEL_URL, model_dir='.', progress=True, file_name=MODEL_NAME)
            st.success("Model downloaded successfully!", icon="✅")
        except Exception as e:
            st.error(f"Failed to download model: {e}. Please ensure you have an internet connection and the URL is correct.", icon="❌")
            return None
    else:
        st.success(f"Model file '{MODEL_NAME}' found.", icon="✅")

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)

    try:
        loadnet = torch.load(MODEL_NAME, map_location=DEVICE)
        if 'params_ema' in loadnet:
            model.load_state_dict(loadnet['params_ema'], strict=False)
        else:
            model.load_state_dict(loadnet, strict=False)
        st.success("Model weights loaded successfully!", icon="✅")
    except Exception as e:
        st.error(f"Error loading model weights: {e}. Please check the model file integrity and compatibility.", icon="❌")
        return None

    model.eval()
    model = model.to(DEVICE)
    st.info(f"Model loaded on: {DEVICE.upper()}")

    upsampler = RealESRGANer(
        scale=4,
        model_path=MODEL_NAME,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=(DEVICE == "cuda")
    )
    return upsampler

def compress_image_by_quality_func(image, quality):
    """
    Compresses a PIL Image to a target file size in KB.
    """
    buf = io.BytesIO()
    try:
        image.save(buf, format="JPEG", quality=quality)
        compressed_image = Image.open(io.BytesIO(buf.getvalue()))
        return compressed_image
    except Exception as e:
        st.error(f"An error occurred during compression: {e}", icon="❗")
        return None

def get_image_bytes(img, mime_type, quality=95):
    """Generates image bytes for download with format control."""
    buffered = io.BytesIO()
    if mime_type == "image/png":
        img.save(buffered, format="PNG")
    elif mime_type == "image/jpeg" or mime_type == "image/jpg": # Handle both "jpeg" and "jpg" as JPEG
        img.save(buffered, format="JPEG", quality=quality)
    elif mime_type == "image/webp":
        img.save(buffered, format="WEBP", quality=quality)
    return buffered.getvalue()

def enhance_single_image_helper(image_pil, enhancement_strength, original_filename):
    """Encapsulates single image enhancement logic."""
    upsampler = load_model()
    if upsampler is None:
        return None

    img_np = np.array(image_pil)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    with st.spinner(f"🚀 Enhancing {original_filename}..."):
        try:
            output, _ = upsampler.enhance(img_cv, outscale=4)
            result_np = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            enhanced_image = Image.fromarray(result_np)

            if enhancement_strength > 0:
                sharpness_percent = int(enhancement_strength * 2)
                enhanced_image = enhanced_image.filter(
                    ImageFilter.UnsharpMask(radius=2, percent=sharpness_percent, threshold=3)
                )
            return enhanced_image
        except torch.cuda.OutOfMemoryError:
            st.error(f"GPU ran out of memory for {original_filename}. Try a smaller image or consider setting `tile` in `RealESRGANer`.", icon="🚨")
            return None
        except Exception as e:
            st.error(f"An error occurred during enhancement of {original_filename}: {e}", icon="❗")
            return None

def remove_background_helper(image_pil):
    """Removes background from a PIL Image using rembg."""
    try:
        # rembg expects a BytesIO object
        img_byte_arr = io.BytesIO()
        # Save as PNG to preserve potential transparency of original if it had it
        image_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0) # Rewind to the beginning

        # Run rembg
        output_bytes = rembg.remove(img_byte_arr.read())
        
        # Open the result bytes as a PIL Image
        removed_bg_image = Image.open(io.BytesIO(output_bytes))
        
        return removed_bg_image
    except ImportError:
        st.error("The 'rembg' library is not installed. Please install it using: `pip install rembg` and `pip install onnxruntime`", icon="❌")
        return None
    except Exception as e:
        st.error(f"An error occurred during background removal: {e}", icon="❗")
        return None

def _add_to_history(image_pil_to_add, specific_caption_for_this_step=None):
    """
    Adds the current image state to history and updates index.
    Manages history steps and determines the caption for the processed image display.
    """
    if 'image_history' not in st.session_state:
        st.session_state['image_history'] = []
        st.session_state['history_index'] = -1
    
    # If not at the end of history (i.e., user has undone actions), clear future history
    if st.session_state['history_index'] < len(st.session_state['image_history']) - 1:
        st.session_state['image_history'] = st.session_state['image_history'][:st.session_state['history_index'] + 1]

    # If history is empty, this is the very first modification after an upload/restore
    # So, the original image is implicitly the "initial" state in our history model.
    if not st.session_state['image_history']:
        if st.session_state['original_uploaded_image_pil'] is not None:
            st.session_state['image_history'].append((st.session_state['original_uploaded_image_pil'].copy(), "Original Image"))
            st.session_state['history_index'] = 0 # Point to the original image in history

    # Determine caption for this new history entry (the processed image)
    # If this is the first *actual edit* after the original, use the specific caption.
    # Otherwise, it's a subsequent edit or part of an undo/redo chain, use generic.
    caption_to_store = specific_caption_for_this_step if len(st.session_state['image_history']) == 1 and specific_caption_for_this_step is not None else "Processed Image"
    
    st.session_state['image_history'].append((image_pil_to_add.copy(), caption_to_store))
    st.session_state['history_index'] += 1 # Move to the newly added processed image

    # Keep history length manageable (e.g., last 10 states including original)
    max_history = 10
    if len(st.session_state['image_history']) > max_history:
        st.session_state['image_history'] = st.session_state['image_history'][-max_history:]
        st.session_state['history_index'] = max_history - 1 # Adjust index after trimming

    # Update the *displayed* current image and caption
    st.session_state['current_edited_image_pil'] = st.session_state['image_history'][st.session_state['history_index']][0]
    st.session_state['processed_image_for_download'] = st.session_state['current_edited_image_pil']
    st.session_state['current_display_caption'] = st.session_state['image_history'][st.session_state['history_index']][1]

    # Any edit turns off ELA display
    st.session_state['display_ela_in_processed_area'] = False

def _undo_redo_logic(direction):
    """Handles undo/redo operations and updates display state."""
    if 'image_history' not in st.session_state or not st.session_state['image_history']:
        return # No history to undo/redo

    new_index = st.session_state['history_index']
    if direction == "undo":
        new_index = st.session_state['history_index'] - 1
        if new_index < 0: # Cannot undo past the very first state (original image if it exists)
            return
    elif direction == "redo":
        new_index = st.session_state['history_index'] + 1
        if new_index >= len(st.session_state['image_history']): # Cannot redo beyond current history
            return

    st.session_state['history_index'] = new_index
    
    # Get the image and caption from history at the new index
    current_image_state, current_caption_state = st.session_state['image_history'][st.session_state['history_index']]
    
    # If the current state in history is the "Original Image" (which is always at index 0)
    # and there are other processed images in history (meaning we undid edits),
    # then clear the processed panel to show the "Perform an edit" info.
    if st.session_state['history_index'] == 0 and len(st.session_state['image_history']) > 1:
        st.session_state['current_edited_image_pil'] = None 
        st.session_state['processed_image_for_download'] = None
        st.session_state['current_display_caption'] = "Processed Image" # Default caption
    else:
        st.session_state['current_edited_image_pil'] = current_image_state
        st.session_state['processed_image_for_download'] = current_image_state
        st.session_state['current_display_caption'] = current_caption_state

    st.session_state['display_ela_in_processed_area'] = False # Always reset ELA display on undo/redo
    st.rerun()


# --- Streamlit App ---

st.set_page_config(
    page_title="PixelPerfect: Image Resolution Optimizer & Editor",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📸 PixelPerfect: Image Resolution Optimizer & Editor")
st.markdown("✨ Enhance, Compress, and Adjust Your Images.")
st.markdown("---")

# Initialize session state variables
if 'original_uploaded_image_pil' not in st.session_state:
    st.session_state['original_uploaded_image_pil'] = None
if 'original_uploaded_image_raw' not in st.session_state: # Stores the bytes of the initially uploaded file
    st.session_state['original_uploaded_image_raw'] = None 
# current_edited_image_pil is initialized to None for conditional display
if 'current_edited_image_pil' not in st.session_state: 
    st.session_state['current_edited_image_pil'] = None
if 'current_image_name' not in st.session_state:
    st.session_state['current_image_name'] = None
if 'uploaded_files_list' not in st.session_state:
    st.session_state['uploaded_files_list'] = []
if 'processed_image_for_download' not in st.session_state:
    st.session_state['processed_image_for_download'] = None
if 'processed_image_filename_for_download' not in st.session_state:
    st.session_state['processed_image_filename_for_download'] = "processed_image.png"

# History initialization
if 'image_history' not in st.session_state:
    st.session_state['image_history'] = []
    st.session_state['history_index'] = -1
if 'current_display_caption' not in st.session_state:
    st.session_state['current_display_caption'] = "Processed Image" # Default caption
if 'display_ela_in_processed_area' not in st.session_state:
    st.session_state['display_ela_in_processed_area'] = False
if 'ela_result_image' not in st.session_state:
    st.session_state['ela_result_image'] = None


# Initialize session state for custom resize inputs
if 'custom_resize_new_width' not in st.session_state:
    st.session_state.custom_resize_new_width = 100 
if 'custom_resize_new_height' not in st.session_state:
    st.session_state.custom_resize_new_height = 100 


# --- Sidebar for Global Settings / Upload ---
st.sidebar.header("📂 Upload Your Images")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more images", 
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) == 1:
        current_image_file = uploaded_files[0]
        image_bytes = current_image_file.getvalue()

        # Check if a new file is uploaded or if the file content has changed
        if st.session_state['original_uploaded_image_raw'] != image_bytes:
            image = Image.open(io.BytesIO(image_bytes))
            # Ensure image is in RGB or RGBA mode for consistency with operations
            if image.mode not in ['RGB', 'RGBA']:
                image = image.convert('RGB')
            
            st.session_state['original_uploaded_image_raw'] = image_bytes # Store raw bytes for comparison
            st.session_state['original_uploaded_image_pil'] = image.copy() # Store original for restore
            
            # Reset processed image display and history for new upload
            st.session_state['current_edited_image_pil'] = None 
            st.session_state['processed_image_for_download'] = None 
            st.session_state['current_display_caption'] = "Processed Image" # Reset caption to default
            st.session_state['current_image_name'] = current_image_file.name
            st.session_state['uploaded_files_list'] = [] # Clear batch list if single image selected
            
            # Reset history and ELA display for new upload
            st.session_state['image_history'] = []
            st.session_state['history_index'] = -1
            st.session_state['display_ela_in_processed_area'] = False
            st.session_state['ela_result_image'] = None

            # Initialize custom resize inputs with current image dimensions
            st.session_state.custom_resize_new_width = image.width
            st.session_state.custom_resize_new_height = image.height
            
            st.session_state['processed_image_filename_for_download'] = f"original_{current_image_file.name}"
            st.rerun() # Force rerun to update all displays with new image
    else: # Batch Processing
        st.session_state['uploaded_files_list'] = uploaded_files
        st.session_state['original_uploaded_image_pil'] = None # No single original for batch
        st.session_state['current_edited_image_pil'] = None # No single current image for batch
        st.session_state['current_display_caption'] = "Processed Image" # Reset caption to default
        st.session_state['current_image_name'] = "batch_process_output"
        st.session_state['processed_image_for_download'] = None # No single image for global download in batch mode
        st.session_state['image_history'] = [] # Clear history for batch mode
        st.session_state['history_index'] = -1
        st.session_state['display_ela_in_processed_area'] = False
        st.session_state['ela_result_image'] = None

st.sidebar.markdown("---")
st.sidebar.header("⚙️ General Info")
st.sidebar.info(f"Using **{DEVICE.upper()}** for model processing. On CPU, enhancement can be very slow.", icon="ℹ️")

# --- Global Download Options ---
st.sidebar.markdown("---")
st.sidebar.header("⬇️ Download Image/s")
if st.session_state['processed_image_for_download']:
    download_format = st.sidebar.selectbox(
        "Select Download Format:",
        ["png", "jpeg", "webp"],
        key="global_download_format"
    )
    download_mime_type = f"image/{download_format}" if download_format != "jpeg" else "image/jpeg"
    
    base_name = os.path.splitext(st.session_state['processed_image_filename_for_download'])[0]
    final_download_filename = f"{base_name}.{download_format}" if download_format != "jpeg" else f"{base_name}.jpg"

    st.sidebar.download_button(
        label="Download Current Image",
        data=get_image_bytes(st.session_state['processed_image_for_download'], download_mime_type),
        file_name=final_download_filename,
        mime=download_mime_type,
        use_container_width=True
    )
else:
    st.sidebar.info("Perform an edit to enable download of the processed image.", icon="ℹ️")

# --- History, Undo, Redo, Restore ---
st.sidebar.markdown("---")
st.sidebar.header("↩️ History & Actions")

if st.session_state['original_uploaded_image_pil'] is not None:
    # Display history size and current step only if a single image is loaded
    if not st.session_state['uploaded_files_list']: # Only for single image mode
        st.sidebar.write(f"History Size: {len(st.session_state['image_history'])}")
        st.sidebar.write(f"Current Step: {st.session_state['history_index'] + 1}")

    col_undo, col_redo = st.sidebar.columns(2)

    with col_undo:
        if st.button(
            "Undo Last Action", 
            key="undo_button", 
            disabled=(st.session_state['history_index'] <= 0 or not st.session_state['image_history']),
            use_container_width=True
        ):
            _undo_redo_logic("undo")

    with col_redo:
        if st.button(
            "Redo Last Action", 
            key="redo_button", 
            disabled=(st.session_state['history_index'] >= len(st.session_state['image_history']) - 1 or not st.session_state['image_history']),
            use_container_width=True
        ):
            _undo_redo_logic("redo")
    
    st.sidebar.markdown("---") # Separator for restore button

    is_current_image_original = False
    if st.session_state['original_uploaded_image_pil'] is not None:
        if st.session_state['current_edited_image_pil'] is None or \
           (st.session_state['current_edited_image_pil'] is not None and 
            st.session_state['current_edited_image_pil'].tobytes() == st.session_state['original_uploaded_image_pil'].tobytes() and
            st.session_state['history_index'] == 0): # Check if it's the original image at history index 0
            is_current_image_original = True

    if st.sidebar.button(
        "Restore to Original Image",
        key="restore_original_button",
        disabled=(st.session_state['original_uploaded_image_pil'] is None or is_current_image_original),
        use_container_width=True
    ):
        st.session_state['current_edited_image_pil'] = None # Clear processed image display
        st.session_state['processed_image_for_download'] = None
        st.session_state['current_display_caption'] = "Processed Image" # Reset caption
        st.session_state['image_history'] = [] # Clear history
        st.session_state['history_index'] = -1
        st.session_state['display_ela_in_processed_area'] = False # Reset ELA display
        st.session_state['ela_result_image'] = None
        st.success("Image restored to original state (processed image cleared).", icon="✅")
        st.rerun()

elif st.session_state['original_uploaded_image_pil'] is not None and st.session_state['current_edited_image_pil'] is None:
    st.sidebar.info("Perform an edit to start recording history and enable undo/redo/restore.", icon="ℹ️")
else:
    st.sidebar.info("History available for single image processing.", icon="ℹ️")

# --- Generate Hash Button ---
st.sidebar.markdown("---")
st.sidebar.header("🔍 Image Forensics Tools")
if st.sidebar.button("Generate Image Hash", key="generate_hash_button", use_container_width=True):
    if st.session_state['original_uploaded_image_raw']:
        image_bytes_raw = st.session_state['original_uploaded_image_raw']
        md5_hash = hashlib.md5(image_bytes_raw).hexdigest()
        sha256_hash = hashlib.sha256(image_bytes_raw).hexdigest()

        st.sidebar.info("--- Original Image Hashes ---")
        st.sidebar.markdown(f"**MD5:** `{md5_hash}`")
        st.sidebar.markdown(f"**SHA-256:** `{sha256_hash}`")
        st.sidebar.success("Hashes generated for the original uploaded image.", icon="✅")
    else:
        st.sidebar.warning("Please upload an image first to generate its hash.", icon="⚠️")


# --- Main Content Area - Top Level Image Displays ---

col1_orig, col2_processed = st.columns(2)

if st.session_state['original_uploaded_image_pil'] is not None:
    with col1_orig:
        st.subheader("Original Uploaded Image")
        st.image(st.session_state['original_uploaded_image_pil'], caption=f"Name: {st.session_state['current_image_name']}", use_container_width=True)
        original_image_size_kb = len(get_image_bytes(st.session_state['original_uploaded_image_pil'], "image/png")) / 1024
        st.metric(label="Dimensions", value=f"{st.session_state['original_uploaded_image_pil'].width}x{st.session_state['original_uploaded_image_pil'].height} px")
        st.metric(label="Size", value=f"{original_image_size_kb:.2f} KB")

    with col2_processed:
        # Dynamic display for processed image or ELA result
        if st.session_state.get('display_ela_in_processed_area', False) and st.session_state.get('ela_result_image') is not None:
            st.subheader("ELA Result Image") # Specific subheader for ELA
            st.image(st.session_state['ela_result_image'], caption="ELA Result Image", use_container_width=True)
            # No size metrics for ELA as it's an analysis output
        elif st.session_state['current_edited_image_pil'] is not None:
            st.subheader("Processed Image") # Generic subheader for processed image
            st.image(st.session_state['current_edited_image_pil'], caption=st.session_state['current_display_caption'], use_container_width=True)
            current_edited_size_kb = len(get_image_bytes(st.session_state['current_edited_image_pil'], "image/png")) / 1024
            st.metric(label="Dimensions", value=f"{st.session_state['current_edited_image_pil'].width}x{st.session_state['current_edited_image_pil'].height} px")
            st.metric(label="Size", value=f"{current_edited_size_kb:.2f} KB")
        else:
            st.subheader("Processed Image") # Placeholder subheader
            st.info("Perform an edit to see the processed image here.", icon="✨")

elif st.session_state['uploaded_files_list']: # Batch Processing
    st.subheader(f"Batch Processing: {len(st.session_state['uploaded_files_list'])} Images")
    st.warning("Batch processing applies selected operations to all uploaded images.", icon="📦")
    with col1_orig:
        st.info("Original images will be processed in batch mode.", icon="ℹ️")
    with col2_processed:
        st.info("Batch results will be available for download below, once processed.", icon="✅")
else:
    st.info("⬆️ Please upload an image or multiple images in the sidebar to begin. Then select a tab to apply operations.", icon="⬆️")
    st.image("https://assets-global.website-files.com/5f72674e1d13755b41a3884b/60b9432f22b07e86e74b338d_Streamlit-logo.png", width=300)


st.markdown("---")

# --- Tabs for Operations ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🖼️ Image Enhancement", 
    "🗜️ Image Compression", 
    "🔄 Rotate & Resize", 
    "✨ Adjustments & Filters",
    "⚙️ Denoising",
    "🚶‍♀️ Background Removal",
    "🕵️ Forensics (ELA)"
])

# --- Tab 1: Image Enhancement ---
with tab1:
    st.header("Image Resolution Enhancement")
    st.write("Increase the resolution of your images.")
    
    enhancement_strength = st.slider(
        "Sharpening After Enhancement (%)", 
        0, 100, 100, 
        help="Controls the amount of sharpening applied after super-resolution. 0% means no extra sharpening."
    )
    
    if st.button("🚀 Enhance Image(s)", use_container_width=True, key="enhance_button"):
        if st.session_state['original_uploaded_image_pil'] is not None: # Check if a single image is uploaded
            image_to_enhance = st.session_state['current_edited_image_pil'] if st.session_state['current_edited_image_pil'] is not None else st.session_state['original_uploaded_image_pil']

            enhanced_image = enhance_single_image_helper(
                image_to_enhance, 
                enhancement_strength, 
                st.session_state['current_image_name']
            )
            if enhanced_image:
                _add_to_history(enhanced_image, "Enhanced Image") # Add to history with specific caption
                st.success("Image enhanced successfully! Check the 'Processed Image' above.", icon="✅")
                st.rerun() # Force rerun to update all displays with new image
        elif st.session_state['uploaded_files_list']:
            # Batch enhancement
            st.subheader("Batch Enhancement Results")
            processed_files = []
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)
            for i, uploaded_file in enumerate(st.session_state['uploaded_files_list']):
                my_bar.progress((i + 1) / len(st.session_state['uploaded_files_list']), text=f"Enhancing {uploaded_file.name}...")
                image_to_process = Image.open(uploaded_file).convert('RGB')
                enhanced_img = enhance_single_image_helper(image_to_process, enhancement_strength, uploaded_file.name)
                if enhanced_img:
                    base_name = os.path.splitext(uploaded_file.name)[0]
                    processed_files.append((enhanced_img, get_image_bytes(enhanced_img, "image/png"), f"enhanced_{base_name}.png"))
            my_bar.empty()
            st.success("Batch enhancement complete!", icon="✅")

            if processed_files:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for img_pil, img_bytes, filename in processed_files:
                        zf.writestr(filename, img_bytes)
                
                st.download_button(
                    label="Download All Enhanced Images (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="enhanced_images.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            else:
                st.error("No images were successfully enhanced during batch processing.", icon="🚫")
        else:
            st.error("Please upload an image first.", icon="❗")

# --- Tab 2: Image Compression ---
with tab2:
    st.header("Image Compression")
    st.write("Reduce file size using JPEG compression. Note that high compression can lead to quality loss.")
    
    compression_quality = st.slider("JPEG Compression Quality (0=Worst, 100=Best)", 
                                    min_value=10, max_value=95, value=75, 
                                    help="Higher quality values mean larger files but less degradation. Aim for 70-85 for good balance.")
    
    if st.button("🗜️ Compress Image(s)", use_container_width=True, key="compress_quality_button"):
        if st.session_state['original_uploaded_image_pil'] is not None:
            image_to_compress = st.session_state['current_edited_image_pil'] if st.session_state['current_edited_image_pil'] is not None else st.session_state['original_uploaded_image_pil']
            compressed_image = compress_image_by_quality_func(
                image_to_compress, 
                compression_quality
            )
            if compressed_image:
                _add_to_history(compressed_image, "Compressed Image") # Add to history with specific caption
                st.success("Image compressed successfully! Check the 'Processed Image' above.", icon="✅")
                st.rerun()
        elif st.session_state['uploaded_files_list']:
            st.subheader("Batch Compression Results")
            processed_files = []
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)
            for i, uploaded_file in enumerate(st.session_state['uploaded_files_list']):
                my_bar.progress((i + 1) / len(st.session_state['uploaded_files_list']), text=f"Compressing {uploaded_file.name}...")
                image_to_process = Image.open(uploaded_file).convert('RGB')
                compressed_img = compress_image_by_quality_func(image_to_process, compression_quality)
                if compressed_img:
                    base_name = os.path.splitext(uploaded_file.name)[0]
                    processed_files.append((compressed_img, get_image_bytes(compressed_img, "image/jpeg", quality=compression_quality), f"compressed_q{compression_quality}_{base_name}.jpg"))
            my_bar.empty()
            st.success("Batch compression complete!", icon="✅")

            if processed_files:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for img_pil, img_bytes, filename in processed_files:
                        zf.writestr(filename, img_bytes)
                st.download_button(
                    label="Download All Compressed Images (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="compressed_images.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            else:
                st.error("No images were successfully compressed during batch processing.", icon="🚫")
        else:
            st.error("Please upload an image first.", icon="❗")


# --- Tab 3: Image Rotation and Resizing ---
with tab3: # Tab name changed to "Rotate & Resize"
    st.header("Rotate & Resize Image")
    if st.session_state['original_uploaded_image_pil'] is not None:
        st.info("Rotation and Resizing are only available for single image processing.", icon="ℹ️")
        
        # --- Image Rotation Section ---
        st.subheader("Image Rotation")
        st.write("Rotate the current image by a specified angle.")
        
        rotation_angle = st.slider("Rotation Angle (degrees)", -180, 180, 0, 5, help="Rotate the image clockwise (positive) or counter-clockwise (negative).", key="rotation_angle_slider")

        if st.button("🔄 Apply Rotation", use_container_width=True, key="apply_rotation_button"):
            image_to_rotate = st.session_state['current_edited_image_pil'] if st.session_state['current_edited_image_pil'] is not None else st.session_state['original_uploaded_image_pil']
            
            # Determine fill color for expanded canvas based on image mode
            if image_to_rotate.mode == 'RGBA':
                fill_color = (0, 0, 0, 0) # Transparent
            else:
                fill_color = (0, 0, 0) # Black for RGB/Grayscale

            try:
                # Use expand=True to ensure the whole image fits after rotation
                rotated_image = image_to_rotate.rotate(rotation_angle, expand=True, fillcolor=fill_color)
                _add_to_history(rotated_image, "Adjusted Image") # Use "Adjusted Image" for Rotate/Resize
                st.success("Image rotated successfully! Check the 'Processed Image' above.", icon="✅")
                st.rerun()
            except Exception as e:
                st.error(f"Error during rotation: {e}", icon="❗")

        st.markdown("---")
        st.subheader("Image Resizing by Percentage")
        st.write("Resize the current image (original or edited) by a percentage scale.")
        
        image_to_resize_source_percent = st.session_state['current_edited_image_pil'] if st.session_state['current_edited_image_pil'] is not None else st.session_state['original_uploaded_image_pil']
        img_width_percent, img_height_percent = image_to_resize_source_percent.size
        st.write(f"Current Dimensions: {img_width_percent}x{img_height_percent} pixels")

        scale_percentage = st.slider("Scale Percentage", 10, 200, 100, 5, help="Set the desired scale percentage (e.g., 50% for half size, 200% for double size).", key="scale_percentage_slider")
        
        if st.button("📏 Apply Percentage Resize", use_container_width=True, key="apply_percentage_resize_button"):
            new_width = int(img_width_percent * (scale_percentage / 100))
            new_height = int(img_height_percent * (scale_percentage / 100))
            
            try:
                resized_image = image_to_resize_source_percent.resize((new_width, new_height), Image.LANCZOS)
                _add_to_history(resized_image, "Adjusted Image") # Use "Adjusted Image" for Rotate/Resize
                st.success(f"Image resized to {new_width}x{new_height} successfully! Check the 'Processed Image' above.", icon="✅")
                st.rerun() 
            except Exception as e:
                st.error(f"Error during resizing: {e}", icon="❗")
        
        st.markdown("---")
        st.subheader("Custom Dimension Resizing")
        st.write("Resize the current image to specific pixel dimensions.")

        image_to_resize_custom = st.session_state['current_edited_image_pil'] if st.session_state['current_edited_image_pil'] is not None else st.session_state['original_uploaded_image_pil']
        current_width_custom, current_height_custom = image_to_resize_custom.size

        # Define callbacks for aspect ratio locking
        def update_custom_height_from_width():
            if st.session_state.get('lock_aspect_ratio_custom_checkbox', False): 
                if current_height_custom != 0: 
                    current_ratio = current_width_custom / current_height_custom
                    st.session_state.custom_resize_new_height = int(st.session_state.custom_resize_new_width / current_ratio)
            
        def update_custom_width_from_height():
            if st.session_state.get('lock_aspect_ratio_custom_checkbox', False): 
                if current_width_custom != 0: 
                    current_ratio = current_width_custom / current_height_custom
                    st.session_state.custom_resize_new_width = int(st.session_state.custom_resize_new_height * current_ratio)

        col_custom_resize1, col_custom_resize2 = st.columns(2)
        with col_custom_resize1:
            new_width_custom = st.number_input(
                "New Width (pixels)", 
                min_value=1, 
                value=st.session_state.custom_resize_new_width, 
                key="custom_resize_new_width", 
                on_change=update_custom_height_from_width
            )
        with col_custom_resize2:
            new_height_custom = st.number_input(
                "New Height (pixels)", 
                min_value=1, 
                value=st.session_state.custom_resize_new_height, 
                key="custom_resize_new_height", 
                on_change=update_custom_width_from_height
            )
        
        lock_aspect_ratio_custom = st.checkbox(
            "Lock Aspect Ratio", 
            value=st.session_state.get('lock_aspect_ratio_custom_checkbox', True), 
            key="lock_aspect_ratio_custom_checkbox", 
            help="If checked, changing one dimension will automatically update the other to maintain proportions."
        )

        if st.button("📏 Apply Custom Resize", use_container_width=True, key="apply_custom_resize_button"):
            try:
                final_width = st.session_state.custom_resize_new_width
                final_height = st.session_state.custom_resize_new_height

                resized_image_custom = image_to_resize_custom.resize((final_width, final_height), Image.LANCZOS)
                _add_to_history(resized_image_custom, "Adjusted Image") # Use "Adjusted Image" for Rotate/Resize
                st.success(f"Image resized to {final_width}x{final_height} successfully! Check the 'Processed Image' above.", icon="✅")
                st.rerun()
            except Exception as e:
                st.error(f"Error during custom resizing: {e}", icon="❗")

    else:
        st.info("Please upload a single image to enable rotation and resizing.", icon="ℹ️")


# --- Tab 4: Image Adjustments and Filters ---
with tab4:
    st.header("Image Adjustments & Filters")
    if st.session_state['original_uploaded_image_pil'] is not None:
        st.info("Adjustments are applied to the currently displayed image.", icon="ℹ️")
        
        image_to_adjust = st.session_state['current_edited_image_pil'] if st.session_state['current_edited_image_pil'] is not None else st.session_state['original_uploaded_image_pil']

        st.markdown("### Basic Adjustments")
        col_adj1, col_adj2 = st.columns(2)
        with col_adj1:
            brightness = st.slider("Brightness", 0.0, 2.0, 1.0, 0.05, help="Adjusts the overall lightness/darkness.", key="adj_brightness")
            contrast = st.slider("Contrast", 0.0, 2.0, 1.0, 0.05, help="Adjusts the difference between light and dark areas.", key="adj_contrast")
        with col_adj2:
            saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.05, help="Adjusts the intensity of colors.", key="adj_saturation")
            
        # New Color Balance Sliders
        st.markdown("#### Color Balance (RGB Gain)")
        col_rgb1, col_rgb2, col_rgb3 = st.columns(3)
        with col_rgb1:
            red_gain = st.slider("Red Channel", 0.0, 2.0, 1.0, 0.05, help="Adjusts the intensity of the Red channel.", key="adj_red_gain")
        with col_rgb2:
            green_gain = st.slider("Green Channel", 0.0, 2.0, 1.0, 0.05, help="Adjusts the intensity of the Green channel.", key="adj_green_gain")
        with col_rgb3:
            blue_gain = st.slider("Blue Channel", 0.0, 2.0, 1.0, 0.05, help="Adjusts the intensity of the Blue channel.", key="adj_blue_gain")

        st.markdown("---")
        st.markdown("### Advanced Adjustments (Requires OpenCV)")
        hue_shift = st.slider("Hue Shift", -180, 180, 0, 10, help="Shifts the hue of the colors in the image. Range is -180 to 180, corresponding to -90 to 90 degrees in OpenCV's 0-179 hue scale.", key="adj_hue")

        # Apply adjustments (these are applied on every rerun to show live preview)
        adjusted_image = image_to_adjust
        adjusted_image = ImageEnhance.Brightness(adjusted_image).enhance(brightness)
        adjusted_image = ImageEnhance.Contrast(adjusted_image).enhance(contrast)
        adjusted_image = ImageEnhance.Color(adjusted_image).enhance(saturation)

        # Apply RGB Gain
        if red_gain != 1.0 or green_gain != 1.0 or blue_gain != 1.0:
            img_np_rgb = np.array(adjusted_image)
            img_np_rgb = img_np_rgb.astype(np.float32) # Convert to float for accurate multiplication
            
            img_np_rgb[:,:,0] = np.clip(img_np_rgb[:,:,0] * red_gain, 0, 255)
            img_np_rgb[:,:,1] = np.clip(img_np_rgb[:,:,1] * green_gain, 0, 255)
            img_np_rgb[:,:,2] = np.clip(img_np_rgb[:,:,2] * blue_gain, 0, 255)
            
            adjusted_image = Image.fromarray(img_np_rgb.astype(np.uint8))


        if hue_shift != 0:
            img_np_rgb = np.array(adjusted_image)
            img_np_hsv = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2HSV)
            
            # Fix for OverflowError: Handle hue values to stay within [0, 179]
            # Convert to int for arithmetic, add 180 to ensure positive before modulo
            # This ensures proper wrapping for negative shifts.
            # Use int16 to prevent overflow during intermediate calculations before modulo
            shifted_hue = (img_np_hsv[:,:,0].astype(np.int16) + hue_shift // 2 + 180) % 180
            img_np_hsv[:,:,0] = shifted_hue.astype(np.uint8) # Cast back to uint8
            
            adjusted_image = Image.fromarray(cv2.cvtColor(img_np_hsv, cv2.COLOR_HSV2RGB))
        
        st.markdown("---")
        st.markdown("### Pre-defined Filters")
        filter_choice = st.selectbox(
            "Select a Filter",
            ["None", "Blur", "Contour", "Detail", "Edge Enhance", "Emboss", "Sharpen", "Smooth"],
            key="filter_choice"
        )

        # Apply filter (applied on every rerun to show live preview)
        filtered_image = adjusted_image
        if filter_choice == "Blur":
            filtered_image = filtered_image.filter(ImageFilter.BLUR)
        elif filter_choice == "Contour":
            filtered_image = filtered_image.filter(ImageFilter.CONTOUR)
        elif filter_choice == "Detail":
            filtered_image = filtered_image.filter(ImageFilter.DETAIL)
        elif filter_choice == "Edge Enhance":
            filtered_image = filtered_image.filter(ImageFilter.EDGE_ENHANCE)
        elif filter_choice == "Emboss":
            filtered_image = filtered_image.filter(ImageFilter.EMBOSS)
        elif filter_choice == "Sharpen":
            filtered_image = filtered_image.filter(ImageFilter.SHARPEN)
        elif filter_choice == "Smooth":
            filtered_image = filtered_image.filter(ImageFilter.SMOOTH)

        if st.button("💾 Apply Adjustments & Filters", use_container_width=True, key="apply_adjusted_button"):
            _add_to_history(filtered_image, "Adjusted Image") # Add to history with specific caption
            st.success("Adjustments applied successfully! Check the 'Processed Image' above.", icon="✅")
            st.rerun() 
    else:
        st.info("Please upload a single image to apply adjustments and filters.", icon="ℹ️")


# --- Tab 5: Noise Reduction/Denoising ---
with tab5:
    st.header("Image Denoising")
    st.write("Reduce noise in your images using non-local means denoising. This can be computationally intensive.")

    if st.session_state['original_uploaded_image_pil'] is not None:
        st.info("Denoising is only available for single image processing.", icon="ℹ️")
        
        image_to_denoise = st.session_state['current_edited_image_pil'] if st.session_state['current_edited_image_pil'] is not None else st.session_state['original_uploaded_image_pil']

        h_denoise = st.slider("Denoising Strength (h)", 0, 50, 10, help="Parameter controlling filter strength. Higher value removes more noise but may blur details.")
        h_color_denoise = st.slider("Color Denoising Strength (hColor)", 0, 50, 10, help="Parameter controlling filter strength for color components. Higher value removes more color noise.")
        template_window_size = st.slider("Template Window Size", 1, 21, 7, 2, help="Size of the pixel neighborhood used to calculate weighted average. Must be odd.")
        search_window_size = st.slider("Search Window Size", 1, 21, 21, 2, help="Size of the window that is used to compute weighted average for given pixel. Must be odd.")

        if st.button("🧼 Apply Denoising", use_container_width=True, key="denoise_button"):
            with st.spinner("Applying denoising... This can take a while for large images."):
                try:
                    img_np_rgb = np.array(image_to_denoise)
                    # Convert to BGR as OpenCV's fastNlMeansDenoisingColored expects BGR
                    img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)

                    denoised_img_bgr = cv2.fastNlMeansDenoisingColored(
                        img_np_bgr, 
                        None, 
                        h=h_denoise, 
                        hColor=h_color_denoise, 
                        templateWindowSize=template_window_size, 
                        searchWindowSize=search_window_size
                    )
                    # Convert back to RGB for PIL Image
                    denoised_pil_image = Image.fromarray(cv2.cvtColor(denoised_img_bgr, cv2.COLOR_BGR2RGB))

                    _add_to_history(denoised_pil_image, "Adjusted Image") # Use "Adjusted Image" for Denoising
                    st.success("Denoising applied successfully! Check the 'Processed Image' above.", icon="✅")
                    st.rerun() 
                except Exception as e:
                    st.error(f"Error during denoising: {e}", icon="❗")
    else:
        st.info("Please upload a single image to apply denoising.", icon="ℹ️")


# --- Tab 6: Background Removal ---
with tab6:
    st.header("Background Removal")
    st.write("Remove the background from your image, focusing on human subjects. Requires `rembg` library.")
    st.info("If you encounter errors, ensure `rembg` is installed: `pip install rembg` and `pip install onnxruntime`", icon="💡")

    if st.session_state['original_uploaded_image_pil'] is not None:
        st.info("Background removal is only available for single image processing.", icon="ℹ️")

        image_to_process_bg = st.session_state['current_edited_image_pil'] if st.session_state['current_edited_image_pil'] is not None else st.session_state['original_uploaded_image_pil']
        
        # rembg works best with RGBA for output transparency
        if image_to_process_bg.mode != 'RGBA':
            image_to_process_bg = image_to_process_bg.convert('RGBA')

        if st.button("✂️ Remove Background", use_container_width=True, key="remove_bg_button"):
            with st.spinner("Removing background... This may take a moment."):
                removed_bg_image = remove_background_helper(image_to_process_bg)
                if removed_bg_image:
                    _add_to_history(removed_bg_image, "Removed Background") # Specific caption
                    st.success("Background removed successfully! Check the 'Processed Image' above.", icon="✅")
                    st.rerun()
    else:
        st.info("Please upload a single image to remove its background.", icon="ℹ️")

# --- Tab 7: Basic Image Forensics (ELA) ---
with tab7:
    st.header("Basic Image Forensics: Error Level Analysis (ELA)")
    st.write("ELA highlights areas in a JPEG image that have been compressed at different rates, which can indicate tampering. Modified regions often have a higher ELA signal.")
    st.warning("ELA is most effective on JPEG images. If your uploaded image is not JPEG, it will be converted to JPEG (quality 95) for analysis. The result is grayscale where brighter areas indicate higher error levels/potential manipulation.", icon="💡")

    if st.session_state['original_uploaded_image_pil'] is not None:
        st.info("ELA is only available for single image processing.", icon="ℹ️")

        image_for_ela = st.session_state['current_edited_image_pil'] if st.session_state['current_edited_image_pil'] is not None else st.session_state['original_uploaded_image_pil']

        if st.button("🕵️ Generate ELA", use_container_width=True, key="generate_ela_button"):
            with st.spinner("Generating ELA image..."):
                try:
                    # Convert to RGB (if not already) and ensure it's not RGBA for JPEG save
                    if image_for_ela.mode == 'RGBA':
                        ela_base_image = image_for_ela.convert('RGB')
                    else:
                        ela_base_image = image_for_ela.copy()

                    # Save at two different qualities
                    buffer_q95 = io.BytesIO()
                    ela_base_image.save(buffer_q95, format="JPEG", quality=95)
                    buffer_q95.seek(0)
                    img_q95 = Image.open(buffer_q95)

                    buffer_q75 = io.BytesIO()
                    ela_base_image.save(buffer_q75, format="JPEG", quality=75)
                    buffer_q75.seek(0)
                    img_q75 = Image.open(buffer_q75)

                    # Convert to numpy arrays
                    np_q95 = np.array(img_q95).astype(np.int16) # Use int16 to allow for negative differences
                    np_q75 = np.array(img_q75).astype(np.int16)

                    # Calculate absolute difference
                    diff = np.abs(np_q95 - np_q75)

                    # Normalize and convert to grayscale for display
                    # Max possible difference for 8-bit is 255. Normalize to 0-255 range.
                    # A multiplier helps make subtle differences more visible.
                    max_diff = np.max(diff)
                    if max_diff > 0:
                        ela_output_np = (diff / max_diff * 255).astype(np.uint8)
                    else:
                        ela_output_np = np.zeros_like(diff, dtype=np.uint8) # No difference, pure black

                    # Convert to grayscale PIL image
                    ela_pil_image = Image.fromarray(ela_output_np)
                    if ela_pil_image.mode != 'L': # Ensure it's grayscale
                        ela_pil_image = ela_pil_image.convert('L')
                    
                    # Update session state to display ELA in processed image area
                    st.session_state['display_ela_in_processed_area'] = True
                    st.session_state['ela_result_image'] = ela_pil_image
                    
                    # Do NOT update current_edited_image_pil or processed_image_for_download
                    # as ELA is an analysis result, not a modification to be downloaded via main button
                    st.success("ELA image generated successfully! Review the 'Processed Image' area above.", icon="✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"An error occurred during ELA generation: {e}", icon="❗")
                    st.warning("ELA works best on JPEG images. If your image is PNG, try converting it to JPEG first using the Compression tab.", icon="ℹ️")
    else:
        st.info("Please upload a single image to perform ELA.", icon="ℹ️")