import os
import base64
from io import BytesIO
import re
import time
import cv2
import numpy as np
from openai import OpenAI
from PIL import Image

# Initialize OpenAI client (using LiteLLM proxy)
client = OpenAI(base_url="https://litellm.deriv.ai/v1", api_key=os.getenv('LITELLM_API_KEY'))

def pil_to_cv2(pil_image):
    """Convert PIL image to OpenCV format (BGR)"""
    # Convert PIL image to RGB numpy array
    rgb_image = np.array(pil_image)
    # Convert RGB to BGR (OpenCV format)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return bgr_image

def cv2_to_pil(cv2_image):
    """Convert OpenCV image (BGR) to PIL format"""
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # Convert to PIL image
    pil_image = Image.fromarray(rgb_image)
    return pil_image

def encode_image_to_base64(cv2_image):
    """Convert an OpenCV image to base64 string."""
    # Convert to PIL image first
    pil_image = cv2_to_pil(cv2_image)
    # Save to BytesIO object
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    # Get base64 string
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def decode_base64_to_image(base64_string):
    """Convert a base64 string to an OpenCV image."""
    # Decode base64 to image data
    image_data = base64.b64decode(base64_string)
    # Convert to PIL image
    pil_image = Image.open(BytesIO(image_data))
    # Convert to OpenCV format
    cv2_image = pil_to_cv2(pil_image)
    return cv2_image

def segment_colored_objects_with_gemini(cv2_image):
    """
    Send a frame to Gemini to detect and segment red, blue, and yellow objects.
    Returns the processed image with segmented objects and bounding boxes.
    """
    # Convert the OpenCV image to base64
    base64_image = encode_image_to_base64(cv2_image)
    
    # Create the message with the image
    try:
        response = client.chat.completions.create(
            model="gemini-2.0-flash-001",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Identify all red, blue, and yellow objects in this image. Return a JSON with the following format: {\"objects\": [{\"color\": \"red/blue/yellow\", \"bbox\": [x1, y1, x2, y2]}]}. Where bbox coordinates are normalized between 0 and 1."
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        # Extract the response content
        result_text = response.choices[0].message.content
        
        # Try to parse JSON from the response
        try:
            # Find JSON in the response (it might be surrounded by markdown or other text)
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                import json
                result_json = json.loads(json_str)
                
                # Draw bounding boxes on the image
                return draw_colored_bounding_boxes(cv2_image, result_json)
            else:
                print("No valid JSON found in response")
                return cv2_image
        except Exception as e:
            print(f"Error processing response: {e}")
            print(f"Response was: {result_text}")
            return cv2_image
            
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return cv2_image

def draw_colored_bounding_boxes(image, result_json):
    """Draw colored bounding boxes for detected objects."""
    # Create a copy of the image to draw on
    output_image = image.copy()
    height, width = output_image.shape[:2]
    
    # Define colors for different object types (BGR format)
    color_map = {
        "red": (0, 0, 255),    # Red in BGR
        "blue": (255, 0, 0),   # Blue in BGR
        "yellow": (0, 255, 255)  # Yellow in BGR
    }
    
    # Check if the expected structure exists in the JSON
    if "objects" not in result_json:
        print("No 'objects' key in JSON response")
        return output_image
    
    # Draw each detected object
    for obj in result_json["objects"]:
        if "bbox" not in obj or "color" not in obj:
            continue
            
        # Get bounding box coordinates (normalized to pixel values)
        bbox = obj["bbox"]
        if len(bbox) != 4:
            continue
            
        x1, y1, x2, y2 = bbox
        x1, y1 = int(x1 * width), int(y1 * height)
        x2, y2 = int(x2 * width), int(y2 * height)
        
        # Get object color
        color_name = obj["color"].lower()
        color = color_map.get(color_name, (0, 255, 0))  # Default to green if color not found
        
        # Draw bounding box
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        text = color_name
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(output_image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(output_image, text, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return output_image

def process_video(input_path, output_path, target_fps=15, max_frames=100):
    """
    Process a video by reducing frame rate and segmenting colored objects using Gemini.
    
    Args:
        input_path: Path to the input video
        output_path: Path to save the output video
        target_fps: Target frames per second (default: 15)
        max_frames: Maximum number of frames to process (default: 100)
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Original video: {width}x{height}, {original_fps} FPS, {total_frames} frames")
    print(f"Processing only the first {max_frames} frames (approximately {max_frames/original_fps:.2f} seconds)")
    
    # Calculate frame sampling rate to achieve target FPS
    frame_sampling_rate = max(1, round(original_fps / target_fps))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    
    frame_count = 0
    processed_count = 0
    
    print("Processing video frames...")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process only every nth frame to achieve target FPS
        if frame_count % frame_sampling_rate == 0:
            print(f"Processing frame {frame_count}/{max_frames}...")
            
            # Process frame with Gemini
            processed_frame = segment_colored_objects_with_gemini(frame)
            
            # Write the frame to output video
            out.write(processed_frame)
            processed_count += 1
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
        
        frame_count += 1
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Video processing complete. Output saved to: {output_path}")
    print(f"Processed frames: {processed_count} out of {frame_count} frames")
    print(f"Target FPS: {target_fps}, Actual FPS: {original_fps/frame_sampling_rate:.2f}")

if __name__ == "__main__":
    input_video = "video_2.mp4"
    output_video = "output/segmented_video_cv2.mp4"
    
    # Process only the first 100 frames (approximately 10 seconds at original FPS)
    process_video(input_video, output_video, target_fps=15, max_frames=600) 