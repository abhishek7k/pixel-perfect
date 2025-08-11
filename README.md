# ✨ PixelPerfect: Image Resolution Optimizer & Editor 📸

## Enhance, Optimize, and Analyze Your Digital Images with AI & Simplicity!

PixelPerfect is a comprehensive, web-based image processing application designed to empower users with advanced tools for enhancing, optimizing, and even forensically analyzing their digital images. Built with Python and Streamlit, it provides a user-friendly interface to leverage powerful AI models and traditional image manipulation techniques for various needs, from content creation to basic digital forensics.

---

## 🚀 Key Features

* **AI-Powered Image Enhancement (Super-Resolution):**
    * Upscales images by 4x using the state-of-the-art **Real-ESRGAN** AI model (PyTorch-based).
    * Includes adjustable post-enhancement sharpening for crystal-clear results.
* **Efficient Image Compression:**
    * Reduce file sizes with controllable JPEG compression quality.
* **Flexible Image Resizing:**
    * Resize images by percentage scale or to custom pixel dimensions with aspect ratio locking.
* **Comprehensive Adjustments & Filters:**
    * Fine-tune images with sliders for brightness, contrast, saturation, and hue.
    * Apply a range of pre-defined filters (Blur, Sharpen, Emboss, Contour, etc.).
* **Intelligent Denoising:**
    * Reduce noise in images using the advanced Non-Local Means algorithm.
* **Automatic Background Removal:**
    * Effortlessly remove backgrounds from images using the `rembg` AI library.
* **🕵️ Basic Image Forensics:**
    * **Error Level Analysis (ELA):** Detect potential image tampering by analyzing compression inconsistencies in JPEG images.
    * **Image Hash Generation:** Compute MD5 and SHA-256 cryptographic hashes for integrity verification of original images.
* **Non-Destructive Workflow:**
    * Robust **Undo/Redo** functionality for all editing operations.
    * Option to **Restore to Original** at any point.
* **Batch Processing:**
    * Apply enhancements or compression to multiple images simultaneously and download results as a convenient ZIP archive.
* **Intuitive Web Interface:**
    * User-friendly design with clear sidebar controls and tabbed feature organization.

---

## 💡 Why PixelPerfect?

* **AI at Your Fingertips:** Leverage powerful deep learning models without complex setups.
* **Privacy-First:** All processing is done in-memory per session; no image data is persistently stored on any server.
* **Versatile Tool:** From enhancing old photos to optimizing web assets and even checking image authenticity.
* **Open Source:** Transparent, extensible, and built with widely used Python libraries.

---

## 📸 Demo & Screenshots

You can see a live demo (if deployed to Streamlit Cloud or similar) or static screenshots here:

* **Screenshots:**
  
https://github.com/abhishek7k/pixel-perfect/blob/main/Screenshot%202025-06-02%20150654.png
https://github.com/abhishek7k/pixel-perfect/blob/main/Screenshot%202025-06-02%20170026.png
https://github.com/abhishek7k/pixel-perfect/blob/main/Screenshot%202025-06-02%20170142.png
https://github.com/abhishek7k/pixel-perfect/blob/main/Screenshot%202025-06-02%20170255.png
https://github.com/abhishek7k/pixel-perfect/blob/main/Screenshot%202025-06-02%20170350.png
https://github.com/abhishek7k/pixel-perfect/blob/main/Screenshot%202025-06-02%20170523.png
https://github.com/abhishek7k/pixel-perfect/blob/main/Screenshot%202025-06-02%20172330.png
https://github.com/abhishek7k/pixel-perfect/blob/main/Screenshot%202025-06-02%20175921.png
--

## ⚙️ Installation & Setup

Follow these steps to get PixelPerfect running on your local machine.

### Prerequisites

* **Python 3.8+** (Python 3.10 or 3.11 recommended)
* **Git** (if cloning the repository)

### Step-by-Step Guide

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/PixelPerfect.git](https://github.com/yourusername/PixelPerfect.git) # Replace with your repo URL
    cd PixelPerfect
    ```

2.  **Create and activate a Python virtual environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install project dependencies:**
    This will install Streamlit, Pillow, OpenCV, Real-ESRGAN, rembg, and all other necessary libraries.
    ```bash
    pip install -r requirements.txt
    ```
    **Important Note for GPU (NVIDIA CUDA) Users:**
    If you have an NVIDIA GPU and want to leverage it for faster AI enhancement (highly recommended for performance), you might need to install `torch` and `torchvision` with CUDA support **manually** after the above step, as `requirements.txt` might default to CPU versions.
    * First, deactivate your `venv` (`deactivate`), then reactivate.
    * Uninstall existing `torch` and `torchvision` if any:
        ```bash
        pip uninstall torch torchvision torchaudio
        ```
    * Visit [PyTorch's official website](https://pytorch.org/get-started/locally/) and copy the `pip` command specific to your CUDA version (e.g., `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`).
    * After installing PyTorch with CUDA, you might need to reinstall `realesrgan`: `pip install realesrgan`.

4.  **Download AI Models (Automatic):**
    The necessary Real-ESRGAN (`weights/RealESRGAN_x4plus.pth`) and `rembg` models will be downloaded automatically the first time you run the application and use their respective features. Ensure you have an active internet connection for the first run.

5.  **Run the application:**
    ```bash
    streamlit run app.py
    ```
    Your default web browser should open a new tab with the PixelPerfect application (usually at `http://localhost:8501`).

---

## 🚀 Usage

1.  **Upload Image(s):** Use the "Upload one or more images" button in the sidebar.
    * For **single image** editing, upload one file.
    * For **batch processing**, upload multiple files.
2.  **Select a Feature Tab:** Navigate through the tabs in the main content area (e.g., "Image Enhancement," "Image Compression," "Basic Image Forensics").
3.  **Adjust Parameters:** Use sliders, dropdowns, and input fields to configure the desired operation.
4.  **Apply Operation:** Click the "Apply" or "Generate" button within the tab.
5.  **View Results:** The "Processed Image" panel (right side) will update to show the result. For forensic tools like ELA, it will display the analysis output.
6.  **History:** Use "Undo," "Redo," and "Restore to Original" buttons in the sidebar to manage your editing history (for single images).
7.  **Download:**
    * For **single processed images or forensic results**, select your desired format in the sidebar "Download Options" and click "Download Current Image."
    * For **batch processed images**, a dedicated "Download All [X] Images (ZIP)" button will appear within the feature tab after processing is complete.

---

## 💻 Technologies Used

* **Python:** Core programming language.
* **Streamlit:** Web application framework for interactive UI.
* **Pillow (PIL Fork):** General image processing and manipulation.
* **OpenCV (`cv2`):** Advanced image processing (Hue, Denoising).
* **PyTorch (`torch`):** Deep learning framework for AI models.
* **Real-ESRGAN (`realesrgan`):** AI model for super-resolution image enhancement.
* **`rembg`:** AI library for automatic background removal.
* **`hashlib`:** For cryptographic hash generation (MD5, SHA-256).
* **`io`:** Python's standard library for in-memory byte streams.
* **`zipfile`:** Python's standard library for creating ZIP archives.
* **`numpy`:** Numerical computing for efficient array operations.
* **ONNX Runtime:** (Used internally by `rembg`) High-performance inference engine for ONNX models.

---

## 👥 Team & Work Distribution

This project was a collaborative effort by the following team members:

* **Abhishek Kumar:** Project Lead, Core AI/ML Backend Developer, System Integrator, Digital Forensics Specialist.
* **Anisha Singh:** Frontend Development Lead, UI/UX Designer, Image Adjustments & Filters Developer.
* **Preeti:** Image Optimization & Manipulation Features Developer, Requirements Analyst, User Manual Contributor.
* **Shweta Gritlahre:** Advanced Image Processing & Forensic Tools Developer, Quality Assurance (QA) Lead, Algorithm Research & Validation.

---

## 💡 Future Enhancements

We envision several exciting future enhancements for PixelPerfect:

* **Advanced Editing:** Interactive cropping, selective adjustments, text/shape overlays, content-aware fill.
* **More AI Models:** Image inpainting, style transfer, face restoration, image colorization.
* **Improved UX:** Interactive image viewer with zoom/pan, custom preset saving, drag-and-drop uploads.
* **Integration:** Cloud storage (Google Drive, Dropbox), user accounts, API exposure.
* **Deeper Forensics:** Comprehensive metadata analysis, advanced copy-move forgery detection.
* **Performance:** Asynchronous processing, client-side rendering (WebAssembly).

---

## 📧 Contact

For any questions, feedback, or collaborations, please feel free to reach out to Abhishek Kumar at abk210820@gmail.com

---
