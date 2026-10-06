import time
import face_recognition
import numpy as np
import os
import cv2  # OpenCV for image processing
from models import Person, db

# --- Photo to Sketch Conversion ---
def convert_to_sketch(image_path, output_path):
    """
    Converts an input image to a pencil sketch using OpenCV and saves it.
    """
    try:
        # Read the image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return False

        # 1. Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Invert the grayscale image to get a negative
        inverted_gray_img = 255 - gray_img

        # 3. Apply a strong Gaussian blur to the inverted image to soften it
        #    A larger kernel size (e.g., 111) makes the lines darker and thicker.
        blurred_img = cv2.GaussianBlur(inverted_gray_img, (111, 111), 0)

        # 4. Invert the blurred image
        inverted_blurred_img = 255 - blurred_img

        # 5. Create the pencil sketch by dividing the grayscale image by the inverted blurred image
        sketch_img = cv2.divide(gray_img, inverted_blurred_img, scale=256.0)

        # Save the final sketch image to the specified output path
        cv2.imwrite(output_path, sketch_img)
        return True
    except Exception as e:
        print(f"An error occurred during sketch conversion: {e}")
        return False


def convert_to_sketch_in_memory(image_bytes, ksize=121, intensity=256.0, contrast=1.0):
    """
    Converts image bytes to sketch bytes in memory using OpenCV dodge-divide pipeline.
    Preserves exact existing algorithm with optional parameter adjustments:
    - ksize: Gaussian blur kernel size (odd integer >= 3, default 121)
    - intensity: Scale factor in cv2.divide (default 256.0)
    - contrast: Linear contrast multiplier (default 1.0)
    """
    try:
        # Decode the image from bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return None

        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Invert the grayscale image
        inverted_gray_image = 255 - gray_image

        # Ensure ksize is odd and valid
        try:
            k = int(ksize)
            if k % 2 == 0:
                k += 1
            k = max(3, min(201, k))
        except (ValueError, TypeError):
            k = 121

        # Apply Gaussian blur for pencil sketch lines
        blurred_image = cv2.GaussianBlur(inverted_gray_image, (k, k), 0)

        # Invert the blurred image
        inverted_blurred_image = 255 - blurred_image

        # Scale factor
        try:
            scale_val = float(intensity)
        except (ValueError, TypeError):
            scale_val = 256.0

        # Create pencil sketch by dividing grayscale by inverted blurred image
        pencil_sketch = cv2.divide(gray_image, inverted_blurred_image, scale=scale_val)

        # Optional contrast adjustment
        try:
            c = float(contrast)
            if c != 1.0 and 0.5 <= c <= 2.0:
                pencil_sketch = cv2.convertScaleAbs(pencil_sketch, alpha=c, beta=0)
        except (ValueError, TypeError):
            pass

        # Encode the resulting sketch back to JPEG bytes
        is_success, buffer = cv2.imencode(".jpg", pencil_sketch, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if is_success:
            return buffer.tobytes()
        else:
            return None
    except Exception as e:
        print(f"Error converting image to sketch in memory: {e}")
        return None


def process_photo_to_sketch(image_bytes, ksize=121, intensity=256.0, contrast=1.0):
    """
    High-level processor that validates input bytes, measures execution latency,
    and returns dimensions and output sketch bytes.
    """
    t0 = time.perf_counter()
    if not image_bytes:
        return {'success': False, 'error': 'No image selected.'}

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return {'success': False, 'error': 'Unable to process this image. File may be corrupt or invalid.'}

        h, w = image.shape[:2]
        sketch_bytes = convert_to_sketch_in_memory(image_bytes, ksize=ksize, intensity=intensity, contrast=contrast)
        if not sketch_bytes:
            return {'success': False, 'error': 'Sketch conversion failed.'}

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            'success': True,
            'sketch_bytes': sketch_bytes,
            'input_dims': (w, h),
            'output_dims': (w, h),
            'processing_time_ms': round(elapsed_ms, 1),
            'error': None
        }
    except Exception as e:
        return {'success': False, 'error': f'Sketch processing failed: {str(e)}'}


# --- Face Recognition Core Functions ---

def get_face_encoding(image_path):
    """
    Loads an image file and returns the face encoding for the first face found.
    Returns None if no face is found or an error occurs.
    """
    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            return encodings[0]  # Return the encoding of the first face
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
    return None

def serialize_encoding(encoding):
    """
    Converts a numpy array (face encoding) into a comma-separated string for database storage.
    """
    return ','.join(map(str, encoding))

def deserialize_encoding(encoding_str):
    """
    Converts a comma-separated string from the database back into a numpy array.
    """
    return np.array(encoding_str.split(','), dtype=float)

def find_matches(sketch_encoding, top_n=5):
    """
    Compares a given sketch encoding against all known persons in the database.
    Returns a list of the top N matches based on similarity.
    """
    persons = Person.query.all()
    if not persons:
        return []

    # Get all known encodings from the database
    known_encodings = [deserialize_encoding(p.face_encoding) for p in persons]

    # Calculate the distance between the sketch and all known faces.
    # A smaller distance means a better match.
    face_distances = face_recognition.face_distance(known_encodings, sketch_encoding)

    results = []
    for i, person in enumerate(persons):
        distance = face_distances[i]
        # Convert the distance metric (0.0 to 1.0+) to a more intuitive similarity percentage.
        # A distance of 0.0 is a 100% match. We use a simple linear conversion.
        similarity = max(0, (1 - distance) * 100)

        # Only consider matches above a certain similarity threshold (e.g., 50%)
        if similarity > 50:
            results.append({
                'person': person,
                'similarity': round(similarity, 2)
            })

    # Sort the results to show the most similar faces first
    results.sort(key=lambda x: x['similarity'], reverse=True)

    return results[:top_n]
