# **Hand-Gesture-Recognition**

This project implements a **Hand Gesture Recognition** system using deep learning. The goal is to classify hand gestures from image data for intuitive **human-computer interaction** and **gesture-based control systems**.

---

## 🛠️ Features  

- **Data Preprocessing:**  
  - Prepares the dataset by resizing images and normalizing pixel values.  
  - Splits the data into training and testing sets.  

- **Deep Learning Model:**  
  - Builds and trains a Convolutional Neural Network (CNN) to classify hand gestures.  

- **Inference:**  
  - Loads the trained model (`model.h5`) to classify new hand gesture images.  

---



## 🔍 Steps in the Project  

1. **Preprocessing:**  
   - Resize all input images to a standard size.  
   - Normalize pixel values for faster convergence during training.  
   - Split the dataset into training and testing subsets.

2. **Model Training:**  
   - Create a CNN using Keras for image classification.  
   - Train the model on the preprocessed dataset.  
   - Save the trained model as `model.h5`.

3. **Inference:**  
   - Load the trained model.  
   - Classify new hand gesture images provided as input.  

---

## 🔧 How to Run  

1. Clone this repository:  
   ```bash
   git clone https://github.com/niamat-sirrou/PRODIGY_ML_04.git
   cd hand-gesture-recognition
   ```

2. Install the required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Navigate to the `src` folder:  
   ```bash
   cd src
   ```

4. Run the preprocessing script:  
   ```bash
   python preprocessing.py
   ```

5. Train the model:  
   ```bash
   python model.py
   ```

6. Test the model using inference:  
   ```bash
   python inference.py
   ```

---

## 📈 Results  

- **Model Accuracy:**  
  Achieved high accuracy during training (see `Accuracy.png`).  

- **Applications:**  
  The trained model can be integrated into applications such as:  
  - Virtual keyboards  
  - Gesture-controlled devices  
  - Robotics systems  

---

## 🚀 Future Improvements  

- Extend the dataset with more gestures and varying lighting conditions.  
- Optimize the model for **real-time video recognition**.  
- Integrate the system into a web or mobile application.  

---

## 📝 License  

This project is licensed under the **MIT License**. Feel free to use, modify, and share it.
