# Food Classification and Nutrition Analysis Using AI

An AI-based web application that identifies a food item from an uploaded image and displays nutrition information associated with the predicted food category. The project combines a Convolutional Neural Network (CNN) image-classification model with a CSV nutrition dataset.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Dataset and Model Setup](#dataset-and-model-setup)
- [Using the Application](#using-the-application)
- [Nutrition CSV Format](#nutrition-csv-format)
- [Limitations](#limitations)
- [Possible Future Improvements](#possible-future-improvements)

## Project Overview

Food Classification and Nutrition Analysis is a student project that explores how deep learning can be used to classify food images and present related nutritional information. Food images are organized into categories, used to train and test a CNN model, and then processed by the application to predict a food class.

After classification, the application uses the predicted class to find a matching record in a nutrition CSV file. It can then display values such as calories, protein, fat, and carbohydrates, depending on the fields available in the CSV.

> **Important:** The nutrition values are retrieved from the prepared dataset. They are not measured directly from the image, and they may not reflect the exact ingredients, recipe, serving size, or cooking method of the food shown.

## Objectives

- Classify food images into the categories supported by the trained model.
- Apply image preprocessing before prediction.
- Connect the predicted food category to nutrition data stored in a CSV file.
- Present the classification result and available nutrition information through a simple web interface.
- Demonstrate the use of CNNs, Python, and web development in an applied AI project.

## Key Features

- **Image input:** Submit a food image through the web interface.
- **CNN-based classification:** Predict a food category using a trained model.
- **Nutrition lookup:** Match the predicted category to a record in a CSV nutrition dataset.
- **Nutrition display:** Show available nutrition fields, for example calories, protein, fat, and carbohydrates.
- **Web application:** Use a Flask backend with an HTML, CSS, and JavaScript frontend.

The categories supported by the application depend on the images and labels used to train the model.

## How It Works

1. **Image upload:** The user selects a food image.
2. **Preprocessing:** The application prepares the image in the format and size expected by the model.
3. **Classification:** The trained CNN predicts the most likely food category.
4. **Nutrition lookup:** The predicted label is matched with the corresponding entry in the CSV file.
5. **Result display:** The application displays the predicted category and the nutrition values available for that entry.

```text
User uploads food image
          |
          v
   Image preprocessing
          |
          v
   Trained CNN model
          |
          v
 Predicted food category
          |
          v
 Nutrition lookup in CSV
          |
          v
 Food name + available nutrition values
```

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Frontend | HTML, CSS, JavaScript | User interface and image submission |
| Backend | Python, Flask | Request handling and application logic |
| Deep learning | TensorFlow / CNN | Food-image classification |
| Data processing | pandas | Reading and working with tabular nutrition data |
| Nutrition data | CSV file | Stores food labels and associated nutrition values |

## Project Structure

The exact filenames can differ in your project. A typical structure is shown below; update this section to match your repository.

```text
food-classification-nutrition-analysis/
├── app.py                    # Flask application entry point (if named app.py)
├── templates/                # HTML templates
├── static/                   # CSS, JavaScript, and other static assets
├── model/                    # Trained CNN model file
├── dataset/                  # Food images used for training/testing (optional)
├── nutrition.csv             # Nutrition lookup data (example filename)
├── requirements.txt          # Python dependencies
└── README.md
```

Do not upload large datasets, model files, virtual environments, or private configuration files to a public repository unless you intend to share them and have permission to do so.

## Getting Started

### Prerequisites

- Python installed on your computer.
- A TensorFlow version compatible with your Python version and trained model.
- The project source code, trained model file, and nutrition CSV file.

### 1. Download or clone the project

If the project is hosted on GitHub, clone it with:

```bash
git clone <your-repository-url>
cd <your-project-folder>
```

Alternatively, download the project and open its folder in a terminal.

### 2. Create and activate a virtual environment

**Windows (Command Prompt):**

```bat
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

If a `requirements.txt` file is present, run:

```bash
pip install -r requirements.txt
```

If the project does not yet have one, the main packages may include:

```bash
pip install Flask pandas Pillow tensorflow
```

Use package versions compatible with your Python version and the model you trained. If image-processing or other libraries are imported by your code, install those dependencies too.

### 4. Check file paths and configuration

Before running the app, verify that the paths in your Python code point to the actual locations of:

- The trained CNN model.
- The nutrition CSV file.
- Any required image upload or static folders.

For example, the README uses `nutrition.csv` and `model/` as illustrative names. Replace them with the filenames and folders used by your code.

### 5. Run the application

If your Flask entry-point file is named `app.py`, run:

```bash
python app.py
```

Open the local URL printed in the terminal (commonly `http://127.0.0.1:5000/`). If your entry-point file has a different name or your project uses another startup command, use the command configured for your project.

## Dataset and Model Setup

The model is only able to predict classes that it was trained to recognize. For reliable results:

- Organize training and testing images by their intended class labels.
- Keep label names consistent between model training and the nutrition CSV.
- Apply the same preprocessing at prediction time that was used during training, including image size and pixel scaling.
- Keep a record of the class-index-to-label mapping used by the trained model.
- Test the model with images it did not use during training.

The dataset and trained model must be prepared separately if they are not already included in the project. Training scripts, dataset sources, image counts, and evaluation metrics should be documented here once confirmed from the actual project files; this README does not assume a particular accuracy score.

## Using the Application

1. Start the Flask application.
2. Open the application in a browser.
3. Upload or select an image of a food item supported by the model.
4. Submit the image for prediction.
5. Review the predicted food category and the nutrition information returned from the CSV file.

Exact button names and screens may differ depending on the current interface.

## Nutrition CSV Format

The CSV should contain a food-label column that can be matched to the model's predicted class, plus any nutrition fields the app is designed to display. An example format is:

```csv
food_name,calories,protein_g,fat_g,carbohydrates_g
apple,52,0.3,0.2,14
rice,130,2.7,0.3,28
```

These rows are examples of the file format only. Replace them with verified nutrition data appropriate to your project. Column names must match what the Python code expects, and food labels should use a consistent format so that the lookup succeeds.

## Limitations

- Predictions may be incorrect for blurry, poorly lit, obstructed, or unfamiliar food images.
- Visually similar foods can be confused by the model.
- Foods outside the training categories may not be classified correctly.
- Nutrition information depends on the quality and completeness of the CSV data.
- A single predicted food category may not represent a mixed dish with several ingredients.
- The application does not estimate food weight or serving size unless that capability is explicitly implemented.
- Nutrition values can vary based on ingredients, portion size, brand, and cooking method.

## Possible Future Improvements

- Add more diverse training images and food categories.
- Evaluate the model using documented metrics such as accuracy, precision, recall, and a confusion matrix.
- Improve handling of low-confidence predictions and unsupported foods.
- Add clearer nutrition-data sources and serving-size information.
- Improve accessibility, mobile responsiveness, and error messages.
- Add automated tests for image preprocessing, prediction, and CSV lookup.

---

## Project Status

This is an educational AI project demonstrating food-image classification and nutrition-data lookup. Update this README as the implementation evolves, especially the repository structure, setup commands, dataset sources, model filename, and evaluation results.
