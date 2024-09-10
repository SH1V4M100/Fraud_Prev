from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import asyncio
from deepface import DeepFace
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, LeakyReLU
from deepface import DeepFace
import tempfile
from PIL import Image

# Path to save uploaded files temporarily
UPLOAD_DIR = './uploads'

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set the DeepFace home directory for model weights
#os.environ['DEEPFACE_HOME'] = '/root'

@app.get("/")
async def serve_html():
    return FileResponse("static/index.html")
def load_models():
    """Load and return the pre-trained face detection model"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

def variance_of_laplacian(image):
    """Compute the Laplacian of the image and return the variance"""
    return cv2.Laplacian(image, cv2.CV_64F).var()

def predict_age_gender_race(face_image):
    """Predict the age, gender, and race of the face using DeepFace"""
    try:
        result = DeepFace.analyze(face_image, actions=['age', 'gender', 'race'], enforce_detection=False)

        if isinstance(result, list):
            result = result[0]  # Extract the result from the list if it's a list

        age = result.get('age')
        gender = max(result.get('gender', {}), key=result.get('gender', {}).get, default='NA')
        race = max(result.get('race', {}), key=result.get('race', {}).get, default='NA')

        return age, gender, race
    except Exception as e:
        print(f"Error in predicting age, gender, or race: {e}")
        return None, None, None

def age_to_bucket(age):
    """Convert age to an age bucket"""
    if age is None:
        return 'NA'
    if age >= 13 and age <= 17:
        return '13-17 years'
    elif age > 17 and age <= 24:
        return '18-24 years'
    elif age > 24 and age <= 34:
        return '25-34 years'
    elif age > 34 and age <= 44:
        return '35-44 years'
    elif age > 44 and age <= 54:
        return '45-54 years'
    elif age > 54 and age <= 64:
        return '55-64 years'
    elif age > 64:
        return 'above 65 years'
    else:
        return 'NA'
    
def image_to_frame(image_path: str):
    # Open the image using PIL
    pil_image = Image.open(image_path)

    # Convert the image to RGB (if not already in that format)
    pil_image = pil_image.convert('RGB')

    # Convert the PIL image to a NumPy array (OpenCV format)
    opencv_frame = np.array(pil_image)

    # Convert RGB to BGR (since OpenCV uses BGR by default)
    opencv_frame = cv2.cvtColor(opencv_frame, cv2.COLOR_RGB2BGR)

    return opencv_frame

def is_too_bright(frame, brightness_threshold=200):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray_frame)
    return avg_brightness > brightness_threshold

def is_too_dark(frame, darkness_threshold=50):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray_frame)
    return avg_brightness < darkness_threshold

def is_distorted(frame, blur_threshold=100):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    return laplacian_var < blur_threshold

@app.post("/analyze_video/")
async def analyze_video_endpoint(video_file:UploadFile=File(...)):
    try:

        contents=await video_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(contents)
            video_path=temp_file.name

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video")

        frame_count = 0
        too_bright_frames = 0
        too_dark_frames = 0
        distorted_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if is_too_bright(frame):
                too_bright_frames += 1

            if is_too_dark(frame):
                too_dark_frames += 1

            if is_distorted(frame):
                distorted_frames += 1

        cap.release()

        results = {
            "total_frames_analyzed": frame_count,
            "too_bright_frames": too_bright_frames,
            "too_bright_percentage": (too_bright_frames/frame_count) * 100,
            "too_dark_frames": too_dark_frames,
            "too_dark_percentage": (too_dark_frames/frame_count) * 100,
            "distorted_frames": distorted_frames,
            "distorted_percentage": (distorted_frames/frame_count) * 100,
        }
        os.remove(video_path)
        return JSONResponse(content=results)

    except HTTPException as http_ex:
        return JSONResponse(content={"detail": http_ex.detail}, status_code=http_ex.status_code)
    
    except Exception as e:
        return JSONResponse(content={"detail": "Internal Server Error. Please check the logs for more details."}, status_code=500)
    
    

    
@app.post("/analyze_image/")
async def analyze_image_endpoint(video_file:UploadFile=File(...)):
    try:

        # Read the image file
        contents = await video_file.read()
        
        # Save the image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image_file:
            temp_image_file.write(contents)
            image_path = temp_image_file.name

        # Open the image using PIL
        img = Image.open(image_path)

        frame_count = 0
        too_bright_frames = 0
        too_dark_frames = 0
        distorted_frames = 0


        frame=image_to_frame(image_path)
        frame_count += 1

        
        if is_too_bright(frame):
            too_bright_frames += 1
        
        if is_too_dark(frame):
            too_dark_frames += 1

        if is_distorted(frame):
            distorted_frames += 1

        #print (frame_count)
        results = {
            "total_frames_analyzed": frame_count,
            "too_bright_frames": too_bright_frames,
            "too_bright_percentage": (too_bright_frames/frame_count) * 100,
            "too_dark_frames": too_dark_frames,
            "too_dark_percentage": (too_dark_frames/frame_count) * 100,
            "distorted_frames": distorted_frames,
            "distorted_percentage": (distorted_frames/frame_count) * 100,
        }

        
        return JSONResponse(content=results)

    except HTTPException as http_ex:
        return JSONResponse(content={"detail": http_ex.detail}, status_code=http_ex.status_code)
    
    except Exception as e:
        return JSONResponse(content={"detail": "Internal Server Error. Please check the logs for more details."}, status_code=500)
    
    
async def analyze_face_process_frame(frame, face_cascade):
    master_frame = None
    max_faces = 0
    max_sharpness = 0
    best_prediction = {}


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    sharpness = variance_of_laplacian(gray)

    if len(faces) > max_faces or (len(faces) == max_faces and sharpness > max_sharpness):
        max_faces = len(faces)
        max_sharpness = sharpness
        master_frame = frame
        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]
            age, gender, race = predict_age_gender_race(face_image)
            if age is not None and gender is not None and race is not None:
                age_bucket = age_to_bucket(age)
                best_prediction = {'age_bucket': age_bucket, 'gender': gender, 'race': race}

    if master_frame is not None:
        return master_frame, best_prediction
    else:
        print("No faces detected in the frame.")
        return None, {}

async def detect_faces_and_predict_best_frame(video_path, face_cascade):
    """Detect faces and get the best frame with age, gender, and race prediction"""
    loop = asyncio.get_event_loop()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None, {}

    master_frame = None
    max_faces = 0
    max_sharpness = 0
    best_prediction = {}

    def process_frame(frame):
        nonlocal master_frame, max_faces, max_sharpness, best_prediction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        sharpness = variance_of_laplacian(gray)

        if len(faces) > max_faces or (len(faces) == max_faces and sharpness > max_sharpness):
            max_faces = len(faces)
            max_sharpness = sharpness
            master_frame = frame
            for (x, y, w, h) in faces:
                face_image = frame[y:y+h, x:x+w]
                age, gender, race = predict_age_gender_race(face_image)
                if age is not None and gender is not None and race is not None:
                    age_bucket = age_to_bucket(age)
                    best_prediction = {'age_bucket': age_bucket, 'gender': gender, 'race': race}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame asynchronously
        await loop.run_in_executor(None, process_frame, frame)

        if max_faces > 0:
            break

    cap.release()

    if master_frame is not None:
        return master_frame, best_prediction
    else:
        print("No faces detected in the video.")
        return None, {}
    
PROJECT_ID = "luminous-smithy-433212-t7"
vertexai.init(project=PROJECT_ID, location="us-central1")

model = GenerativeModel("gemini-1.5-pro-001")

def analyze_background_video_with_vertex_ai(video_uri: str):
    prompt = """
    Tell me if the video background is a real or fake background. Answer in JSON format.
    """
    
    video_file = Part.from_uri(
        uri=video_uri,
        mime_type="video/mp4",
    )
    
    contents = [video_file, prompt]
    
    response = model.generate_content(contents)
    return response.text

@app.post("/check_background/")
async def check_background(video_file:UploadFile=File(...)):
    try:
        contents=await video_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(contents)
            video_path=temp_file.name

        # Upload the video to Google Cloud Storage
        gcs_uri = f"gs://fraud_prev/{os.path.basename(video_path)}"
        os.system(f"gsutil cp {video_path} {gcs_uri}")
        
        # Analyze the video using Vertex AI
        background_info = analyze_background_video_with_vertex_ai(gcs_uri)
        os.remove(video_path)
        return JSONResponse(content={"background_info": background_info})
    
    except Exception as e:
        return JSONResponse(content={"detail": str(e)}, status_code=500)

def analyze_postbox_video_with_vertex_ai(video_uri: str):
    prompt = """
    Tell me if the video shows a black postbox outdoors in JSON format.
    """
    
    video_file = Part.from_uri(
        uri=video_uri,
        mime_type="video/mp4",
    )
    
    contents = [video_file, prompt]
    
    response = model.generate_content(contents)
    return response.text

@app.post("/check_postbox/")
async def check_postbox(video_file:UploadFile=File(...)):
    try:
        contents=await video_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(contents)
            video_path=temp_file.name

        # Upload the video to Google Cloud Storage
        gcs_uri = f"gs://fraud_prev/{os.path.basename(video_path)}"
        os.system(f"gsutil cp {video_path} {gcs_uri}")
        
        # Analyze the video using Vertex AI
        postbox_info = analyze_postbox_video_with_vertex_ai(gcs_uri)
        os.remove(video_path)
        return JSONResponse(content={"postbox_info": postbox_info})
    
    except Exception as e:
        return JSONResponse(content={"detail": str(e)}, status_code=500)

def analyze_video_with_vertex_ai(video_uri: str):
    prompt = """
    Tell me the OTP being said in the video in JSON format.
    """
    
    video_file = Part.from_uri(
        uri=video_uri,
        mime_type="video/mp4",
    )
    
    contents = [video_file, prompt]
    
    response = model.generate_content(contents)
    return response.text

@app.post("/extract_otp/")
async def extract_otp(video_file:UploadFile=File(...)):
    try:

        contents=await video_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(contents)
            video_path=temp_file.name

        gcs_uri = f"gs://fraud_prev/{os.path.basename(video_path)}"
        os.system(f"gsutil cp {video_path} {gcs_uri}")
        
        otp_info = analyze_video_with_vertex_ai(gcs_uri)
        os.remove(video_path)
        return JSONResponse(content={"otp": otp_info})
    
    except Exception as e:
        return JSONResponse(content={"detail": str(e)}, status_code=500)


@app.post("/analyze_faces/")
async def analyze_faces(video_file:UploadFile=File(...)):
    try:

        contents=await video_file.read()
        file_type = video_file.content_type
        if file_type.startswith("video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(contents)
                video_path=temp_file.name


            face_cascade = load_models()
            master_frame, best_prediction = await detect_faces_and_predict_best_frame(video_path, face_cascade)
        
            os.remove(video_path)
            if master_frame is None:
                return JSONResponse(content={"detail": "No faces detected"}, status_code=404)
        
            return JSONResponse(content=best_prediction)
        
        elif file_type.startswith("image"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image_file:
                temp_image_file.write(contents)
                image_path = temp_image_file.name

            # Open the image using PIL
            img = Image.open(image_path)
            frame=image_to_frame(image_path)
            face_cascade = load_models()
            master_frame, best_prediction = await analyze_face_process_frame(frame, face_cascade)

            if master_frame is None:
                return JSONResponse(content={"detail": "No faces detected"}, status_code=404)
        
            return JSONResponse(content=best_prediction)
    
    except Exception as e:
        return JSONResponse(content={"detail": str(e)}, status_code=500)

@app.post("/check_package_with_bill/")
async def check_package_with_bill(video_file:UploadFile=File(...)):
    try:

        contents=await video_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(contents)
            video_path=temp_file.name

        # Upload the video to Google Cloud Storage
        gcs_uri = f"gs://fraud_prev/{os.path.basename(video_path)}"
        os.system(f"gsutil cp {video_path} {gcs_uri}")

        # Analyze the video using Vertex AI
        prompt = """
        Analyze the video and determine if it is a person returning a package at the store with a bill.Ans in JSON format.
        """
        video_file = Part.from_uri(
            uri=gcs_uri,
            mime_type="video/mp4",
        )
        
        contents = [video_file, prompt]
        response = model.generate_content(contents)
        os.remove(video_path)
        return JSONResponse(content={"package_info": response.text})
    
    except Exception as e:
        return JSONResponse(content={"detail": str(e)}, status_code=500)

image_dimensions = {'height': 256, 'width': 256, 'channels': 3}

# Define thresholds
ANTI_DF_THRESHOLD = 0.4  # Threshold for anti-deepfake score
ANTI_SPOOF_THRESHOLD = 0.8  # Threshold for anti-spoof score

def crop_faces(image):
    try:
        faces = DeepFace.extract_faces(
            img_path=image,
            detector_backend="mtcnn",
            enforce_detection=False,  # Set to False to bypass detection check
            align=True,
            grayscale=False,
            anti_spoofing=True
        )
    except ValueError as e:
        print(f"Error in face detection: {e}")
        return None, None

    if not faces:
        print("No faces detected.")
        return None, None

    cropped_faces = []
    for face in faces:
        x, y, w, h = face["facial_area"]["x"], face["facial_area"]["y"], face["facial_area"]["w"], face["facial_area"]["h"]
        x, y = abs(x), abs(y)
        x -= 25
        y -= 25
        xl = x + w + 35
        yl = y + h + 35
        if xl > image.shape[1]:
            xl = image.shape[1]
        if yl > image.shape[0]:
            yl = image.shape[0]
        x = max(x, 0)
        y = max(y, 0)
        face_crop = image[y:yl, x:xl]
        cropped_faces.append(face_crop)

    return faces[0]['antispoof_score'], cropped_faces[0]

class Classifier:
    def __init__(self):
        self.model = None

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def load(self, path):
        self.model.load_weights(path)

class Meso4(Classifier):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    def init_model(self):
        x = Input(shape=(image_dimensions['height'], image_dimensions['width'], image_dimensions['channels']))

        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return Model(inputs=x, outputs=y)

# Initialize and load the Meso4 model
meso = Meso4()
meso.load('Meso4_DF.h5') #"D:\test_demo\demo\Meso4_DF.h5"

def pred(image):
    is_real, cropped_image = crop_faces(image)
    if cropped_image is None:
        return None, None
    cropped_image = cv2.resize(cropped_image, (256, 256))
    cropped_image = cropped_image.astype('float32') / 255.0
    cropped_image = np.expand_dims(cropped_image, axis=0)
    not_df = meso.predict(cropped_image)[0]
    return is_real, not_df

def pred_vid(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None, None

    success = True
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [0, total_frames // 2, total_frames - 1]
    anti_df_score, anti_spoof_score = 0, 0
    i = 0

    while success and i < 3:
        success, frame = cap.read()
        if not success:
            break
        if i not in frame_indices:
            continue
        if success:
            c, d = pred(frame)
            if c is None or d is None:
                continue
            anti_spoof_score += c
            anti_df_score += d
            i += 1

    cap.release()
    if i > 0:
        anti_spoof_score /= i
        anti_df_score /= i
    return anti_df_score, anti_spoof_score

def classify_scores(anti_df_score, anti_spoof_score):
    if anti_df_score <= ANTI_DF_THRESHOLD:
        df_status = "Deepfake detected"
    else:
        df_status = "No deepfake detected"

    if anti_spoof_score <= ANTI_SPOOF_THRESHOLD:
        spoof_status = "Spoofing detected"
    else:
        spoof_status = "No spoofing detected"

    return df_status, spoof_status

@app.post("/deepfake_detection/")
async def deepfake_detection(video_file:UploadFile=File(...)):
    
    try:
        contents=await video_file.read()
        file_type = video_file.content_type
        if file_type.startswith("video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(contents)
                video_path=temp_file.name

            anti_df_score, anti_spoof_score = pred_vid(video_path)
        
            if anti_df_score is None or anti_spoof_score is None:
                raise HTTPException(status_code=400, detail="Error processing video")

            df_status, spoof_status = classify_scores(anti_df_score, anti_spoof_score)

            result = {
                "anti_df_score": float(anti_df_score),
                "anti_spoof_score": float(anti_spoof_score),
                "deepfake_status": df_status,
                "spoof_status": spoof_status
            }

            explanation = (
                "Anti-Deepfake Score: This score indicates the likelihood that the video is a deepfake. "
                "A higher score suggests that the video is more likely to be a deepfake.\n\n"
                "Anti-Spoof Score: This score measures how well the video resists spoofing attacks. "
                "A higher score indicates a stronger resistance to spoofing.\n\n"
                "Classification:\n"
                f"- Deepfake Status: {df_status}\n"
                f"- Spoof Status: {spoof_status}"
            )
            os.remove(video_path)
            return JSONResponse(content={"result": result, "explanation": explanation})
        
        elif file_type.startswith("image"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image_file:
                temp_image_file.write(contents)
                image_path = temp_image_file.name

            # Open the image using PIL
            img = Image.open(image_path)
            frame=image_to_frame(image_path)

            anti_df_score, anti_spoof_score = pred(frame)
            if anti_df_score is None or anti_spoof_score is None:
                raise HTTPException(status_code=400, detail="Error processing video")
            df_status, spoof_status = classify_scores(anti_df_score, anti_spoof_score)
            result = {
                "anti_df_score": float(anti_df_score),
                "anti_spoof_score": float(anti_spoof_score),
                "deepfake_status": df_status,
                "spoof_status": spoof_status
            }
            explanation = (
                "Anti-Deepfake Score: This score indicates the likelihood that the video is a deepfake. "
                "A higher score suggests that the video is more likely to be a deepfake.\n\n"
                "Anti-Spoof Score: This score measures how well the video resists spoofing attacks. "
                "A higher score indicates a stronger resistance to spoofing.\n\n"
                "Classification:\n"
                f"- Deepfake Status: {df_status}\n"
                f"- Spoof Status: {spoof_status}"
            )
            return JSONResponse(content={"result": result, "explanation": explanation})

    except HTTPException as http_ex:
        return JSONResponse(content={"detail": http_ex.detail}, status_code=http_ex.status_code)
    
    except Exception as e:
        return JSONResponse(content={"detail": "Internal Server Error. Please check the logs for more details."}, status_code=500)
    
def compare_faces(face_image, reference_image_path):
    """Compare the detected face with a reference image and return a match score"""
    try:
        result = DeepFace.verify(face_image, reference_image_path, enforce_detection=False)
        if result["verified"]:
            return result["distance"], True
        else:
            return result["distance"], False
    except Exception as e:
        print(f"Error in comparing faces: {e}")
        return None, False
    
@app.post("/compare_faces/")
async def compare_faces_endpoint(image_file:UploadFile=File(...), add_file:UploadFile=File(...)):
    try:
        contents1 = await image_file.read()
        contents2 = await add_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image_file1:
            temp_image_file1.write(contents1)
            image_path1 = temp_image_file1.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image_file2:
            temp_image_file2.write(contents2)
            image_path2 = temp_image_file2.name

        #img = Image.open(image_path)
        distance, verification = compare_faces(image_path1, image_path2)
        match_perc = (1 - distance)*100
        if match_perc>100:
            match_perc=100
        elif match_perc<0:
            match_perc=0
        
        is_same = match_perc>=50
        result = {
            "match_perc": float(match_perc),
            "is_same": is_same
        }
        
        return JSONResponse(content=result)

    except HTTPException as http_ex:
        return JSONResponse(content={"detail": http_ex.detail}, status_code=http_ex.status_code)
    
    except Exception as e:
        return JSONResponse(content={"detail": "Internal Server Error. Please check the logs for more details."}, status_code=500)