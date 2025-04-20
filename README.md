# Geospatial Solar Energy Estimation Portal (GSEEP)

## **Objective**
The **Geospatial Solar Energy Estimation Portal (GSEEP)** aims to leverage the power of remote sensing, geospatial intelligence, and deep learning to estimate the solar energy potential of urban and semi-urban regions. The system is designed to support individuals, communities, and policymakers in making informed decisions about solar panel installations by analyzing rooftop areas and evaluating the viable solar capacity of each site.

---

## **Phase 1: Building Footprint Extraction (Completed)**
The first phase of GSEEP focused on the automated extraction of building footprints from high-resolution satellite imagery. This step serves as the foundation for downstream solar potential calculations. We implemented semantic segmentation using deep learning architectures, including **U-Net** and **Mask R-CNN**, to detect and delineate rooftops with high precision.

### **Highlights**
- Collected high-resolution satellite imagery for the target regions.  
- Performed image preprocessing and mask generation from annotated polygon shapefiles.  
- Trained and validated deep learning models for accurate rooftop segmentation.  
- Assessed performance using standard metrics like **Intersection over Union (IoU)** and **F1 Score**.  
- Generated binary segmentation masks and geospatial overlays of detected rooftops.  

This phase successfully produced a scalable and automated pipeline for creating a rooftop inventory over large geographic areas.

---

## **Phase 2: Interactive Portal Development (Upcoming)**
The second phase involves the development of an interactive web-based portal that will act as the user interface for the GSEEP system. This interface will allow users to:

- Upload geographic coordinates or select a region of interest.  
- Visualize extracted building footprints with solar energy overlays.  
- Obtain real-time estimates for solar panel placement and potential energy generation.  
- Export comprehensive reports or share findings with stakeholders.  

This portal will bridge the AI-based backend and user-end decision-making tools, ensuring accessibility and usability for various stakeholders.

---

## **Phase 1 Implementation Details**
To realize automated rooftop detection, we developed a complete machine learning pipeline, involving data collection, preprocessing, augmentation, training, and prediction.

### **Required Libraries and Frameworks**
The following Python packages were used for pipeline development:
- **NumPy, Pandas** – For efficient handling of numerical arrays and tabular datasets.  
- **OpenCV** – For image manipulation tasks like resizing, normalization, and mask operations.  
- **Matplotlib, Seaborn** – For visualizing image samples and model performance metrics.  
- **TensorFlow/Keras or PyTorch** – For implementing deep learning models and custom training loops.  
- **Albumentations** – For high-performance image augmentation techniques.  
- **GDAL, Rasterio** – For geospatial data processing, raster file handling, and coordinate referencing.  
- **Scikit-learn** – For data splitting, standardization, and evaluation metrics.  
- **TQDM** – For monitoring training progress and batch iteration.  

These libraries collectively enabled efficient development and deployment of the segmentation pipeline.

### **Data Preprocessing**
Satellite imagery preprocessing was a critical step to ensure high-quality inputs for model training. The preprocessing pipeline included:

- Resizing input tiles to fixed dimensions (e.g., 256x256 or 512x512) to match model input layers.  
- Normalization of pixel intensities for faster convergence and numerical stability.  
- Color space adjustments based on model requirements (e.g., RGB vs grayscale).  
- Generation of binary masks from shapefiles containing rooftop annotations.  
- Spatially-aware train-validation-test splits to prevent data leakage and preserve geographical diversity.  

This phase ensured standardized, clean, and contextually rich inputs for deep learning model consumption.

### **Data Augmentation**
To reduce overfitting and improve generalization, we employed augmentation techniques using the **Albumentations** library. These included:

- Horizontal and vertical flips to simulate varying orientations.  
- Rotations and affine transformations to enhance spatial variability.  
- Brightness and contrast alterations to simulate different lighting conditions.  
- Elastic distortions and noise injection to improve robustness under real-world scenarios.  

These augmentations expanded the training dataset’s diversity and enabled better performance across heterogeneous landscapes.

### **Model Training**
We implemented and compared multiple semantic segmentation models:

- **U-Net**: A symmetric encoder-decoder architecture with skip connections, optimized for dense pixel-wise segmentation.  
- **Mask R-CNN**: Used for instance segmentation, particularly beneficial when rooftops are densely packed or overlapping.  

#### **Key Training Details**
- **Loss Functions**: Binary Cross-Entropy and Dice Loss were employed to address class imbalance and segmentation accuracy.  
- **Optimizer**: The Adam optimizer was used for its adaptive learning rate and fast convergence.  
- **Metrics**: Evaluation was based on **IoU**, **F1 Score**, and pixel-level accuracy, providing a balanced view of model performance.  
- **Callbacks**: Early stopping and model checkpointing were integrated to prevent overfitting and ensure optimal model selection.  

### **Inference and Post-Processing**
After model training, we performed predictions on unseen satellite tiles. Post-processing included:

- Conversion of predicted outputs into binary masks.  
- Optional re-projection of segmentation results onto geographic coordinates for downstream analysis.  

This concludes Phase 1, providing a solid base for the solar potential estimation to be integrated in the next phase.
