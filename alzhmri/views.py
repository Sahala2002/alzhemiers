
from django.shortcuts import render, redirect
from django.conf import settings

from .models import Prediction

import os
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "model",
    "alzheimers_final_model.keras"
)


# ============================================================
# CLASS NAMES
# IMPORTANT:
# Must be in the SAME ORDER as model training
# ============================================================

CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented"
]


# ============================================================
# LOAD MODEL
# ============================================================

model = None

try:

    model = load_model(MODEL_PATH)

    print("======================================")
    print("Alzheimer Model Loaded Successfully")
    print("Model:", MODEL_PATH)
    print("======================================")

except Exception as e:

    print("======================================")
    print("WARNING: Model could not be loaded")
    print("Model path:", MODEL_PATH)
    print("Error:", e)
    print("======================================")


# ============================================================
# HOME / PREDICTION PAGE
# ============================================================

def home(request):

    # --------------------------------------------------------
    # GET REQUEST
    # --------------------------------------------------------

    if request.method == "GET":

        return render(
            request,
            "home.html"
        )

    # --------------------------------------------------------
    # POST REQUEST
    # --------------------------------------------------------

    if request.method == "POST":

        # ----------------------------------------------------
        # CHECK IMAGE UPLOAD
        # ----------------------------------------------------

        if "image" not in request.FILES:

            return render(
                request,
                "home.html",
                {
                    "error": "Please upload an MRI image."
                }
            )

        uploaded_image = request.FILES["image"]

        # ----------------------------------------------------
        # CHECK MODEL
        # ----------------------------------------------------

        if model is None:

            return render(
                request,
                "home.html",
                {
                    "error":
                    "Machine learning model could not be loaded. "
                    "Please check the model file."
                }
            )

        # ----------------------------------------------------
        # CHECK FILE TYPE
        # ----------------------------------------------------

        allowed_extensions = [
            ".jpg",
            ".jpeg",
            ".png"
        ]

        file_extension = os.path.splitext(
            uploaded_image.name
        )[1].lower()

        if file_extension not in allowed_extensions:

            return render(
                request,
                "home.html",
                {
                    "error":
                    "Please upload a JPG, JPEG or PNG image."
                }
            )

        # ----------------------------------------------------
        # CHECK FILE SIZE
        # Maximum 10 MB
        # ----------------------------------------------------

        max_size = 10 * 1024 * 1024

        if uploaded_image.size > max_size:

            return render(
                request,
                "home.html",
                {
                    "error":
                    "Image size must be less than 10 MB."
                }
            )

        # ----------------------------------------------------
        # SAVE IMAGE TO DATABASE
        # ----------------------------------------------------

        try:

            prediction_record = Prediction.objects.create(
                image=uploaded_image,
                prediction="Processing",
                confidence=0
            )

        except Exception as e:

            return render(
                request,
                "home.html",
                {
                    "error":
                    f"Could not save image: {str(e)}"
                }
            )

        # ----------------------------------------------------
        # GET SAVED IMAGE PATH
        # ----------------------------------------------------

        try:

            image_path = prediction_record.image.path

        except Exception as e:

            prediction_record.delete()

            return render(
                request,
                "home.html",
                {
                    "error":
                    f"Could not access uploaded image: {str(e)}"
                }
            )

        # ----------------------------------------------------
        # OPEN AND PREPROCESS IMAGE
        # ----------------------------------------------------

        try:

            image = Image.open(image_path)

            # Convert image to RGB
            image = image.convert("RGB")

            # Resize to model input size
            image = image.resize((224, 224))

            # Convert image to NumPy array
            image_array = np.array(image)

            # Normalize pixel values
            image_array = image_array / 255.0

            # Add batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

        except Exception as e:

            prediction_record.delete()

            return render(
                request,
                "home.html",
                {
                    "error":
                    f"Invalid MRI image: {str(e)}"
                }
            )

        # ----------------------------------------------------
        # MAKE PREDICTION
        # ----------------------------------------------------

        try:

            predictions = model.predict(
                image_array,
                verbose=0
            )

            # Get predicted class index
            predicted_index = np.argmax(
                predictions[0]
            )

            # Make sure index is valid
            if predicted_index >= len(CLASS_NAMES):

                raise ValueError(
                    "Model returned an invalid class index."
                )

            # Get class name
            prediction = CLASS_NAMES[
                predicted_index
            ]

            # Get confidence
            confidence = (
                float(
                    predictions[0][predicted_index]
                ) * 100
            )

            confidence = round(
                confidence,
                2
            )

        except Exception as e:

            prediction_record.delete()

            return render(
                request,
                "home.html",
                {
                    "error":
                    f"Prediction failed: {str(e)}"
                }
            )

        # ----------------------------------------------------
        # UPDATE DATABASE
        # ----------------------------------------------------

        prediction_record.prediction = prediction
        prediction_record.confidence = confidence

        prediction_record.save()

        # ----------------------------------------------------
        # GET IMAGE URL
        # ----------------------------------------------------

        image_url = prediction_record.image.url

        # ----------------------------------------------------
        # SAVE RESULT IN SESSION
        # ----------------------------------------------------

        request.session["prediction"] = prediction
        request.session["confidence"] = confidence
        request.session["image_url"] = image_url

        # ----------------------------------------------------
        # GO TO RESULT PAGE
        # ----------------------------------------------------

        return redirect("result")


# ============================================================
# RESULT PAGE
# ============================================================

def result(request):

    # Get prediction information from session
    prediction = request.session.get(
        "prediction"
    )

    confidence = request.session.get(
        "confidence"
    )

    image_url = request.session.get(
        "image_url"
    )

    # --------------------------------------------------------
    # CHECK IF RESULT EXISTS
    # --------------------------------------------------------

    if not prediction:

        return render(
            request,
            "result.html",
            {
                "error":
                "No prediction result available. "
                "Please upload an MRI image first."
            }
        )

    # --------------------------------------------------------
    # SHOW RESULT
    # --------------------------------------------------------

    return render(
        request,
        "result.html",
        {
            "prediction": prediction,
            "confidence": confidence,
            "image_url": image_url,
        }
    )


# ============================================================
# ABOUT PAGE
# ============================================================

def about(request):

    return render(
        request,
        "about.html"
    )


# ============================================================
# HISTORY PAGE
# ============================================================

def history(request):

    predictions = Prediction.objects.all().order_by(
        "-created_at"
    )

    return render(
        request,
        "history.html",
        {
            "predictions": predictions
        }
    )

