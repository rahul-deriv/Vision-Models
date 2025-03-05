import os
import base64
from io import BytesIO
from pathlib import Path
import datetime
from openai import OpenAI
from PIL import Image
import time
import av  # PyAV for video processing
import tempfile
import subprocess

# Initialize OpenAI client (using LiteLLM proxy as in your image2md.py)
client = OpenAI(base_url="https://litellm.deriv.ai/v1", api_key=os.getenv('LITELLM_API_KEY'))

def encode_image_to_base64(pil_image):
    """Convert a PIL image to base64 string."""
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def decode_base64_to_image(base64_string):
    """Convert a base64 string to a PIL image."""
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))

def segment_colored_objects_with_gemini(pil_image):
    """
    Send a frame to Gemini to detect and segment red, blue, and yellow objects.
    Returns the processed image with segmented objects.
    """
    # Convert the PIL image to base64
    base64_image = encode_image_to_base64(pil_image)
    
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
                            "text": "Identify and segment all red, blue, and yellow objects in this image. Return an image where these colored objects are clearly separated from the background. Make the segmentation very clear with distinct boundaries."
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
        
        # Check if there's an image in the response
        image_pattern = r"data:image\/[^;]+;base64,([^\"]+)"
        import re
        image_match = re.search(image_pattern, result_text)
        
        if image_match:
            # Extract and decode the base64 image
            base64_result = image_match.group(1)
            result_image = decode_base64_to_image(base64_result)
            return result_image
        else:
            print("No image found in Gemini response")
            return pil_image
            
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return pil_image

def process_video(input_path, output_path, target_fps=15, max_frames=30):
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
    input_container = av.open(input_path)
    input_stream = input_container.streams.video[0]
    
    # Get video properties
    original_fps = float(input_stream.average_rate)
    width = input_stream.width
    height = input_stream.height
    total_frames = input_stream.frames
    
    print(f"Original video: {width}x{height}, {original_fps} FPS, {total_frames} frames")
    print(f"Processing only the first {max_frames} frames (approximately {max_frames/original_fps:.2f} seconds)")
    
    # Calculate frame sampling rate to achieve target FPS
    frame_sampling_rate = max(1, round(original_fps / target_fps))
    
    # Create a temporary directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_files = []
        frame_count = 0
        processed_count = 0
        
        print("Processing video frames...")
        
        # Process frames
        for frame in input_container.decode(input_stream):
            # Stop after processing max_frames
            if frame_count >= max_frames:
                break
                
            # Process only every nth frame to achieve target FPS
            if frame_count % frame_sampling_rate == 0:
                print(f"Processing frame {frame_count}/{max_frames}...")
                
                # Convert frame to PIL Image
                pil_frame = frame.to_image()
                
                # Process frame with Gemini
                processed_pil_frame = segment_colored_objects_with_gemini(pil_frame)
                
                # Save processed frame to temp directory
                frame_path = os.path.join(temp_dir, f"frame_{processed_count:06d}.jpg")
                processed_pil_frame.save(frame_path)
                frame_files.append(frame_path)
                
                processed_count += 1
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
            
            frame_count += 1
        
        # Use ffmpeg to combine frames into video
        if frame_files:
            # Create a file with frame filenames
            frames_list_path = os.path.join(temp_dir, "frames_list.txt")
            with open(frames_list_path, 'w') as f:
                for frame_file in frame_files:
                    f.write(f"file '{frame_file}'\n")
            
            # Use ffmpeg to create video from frames
            cmd = [
                'ffmpeg', '-y', '-r', str(target_fps), 
                '-f', 'concat', '-safe', '0', '-i', frames_list_path,
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', 
                output_path
            ]
            
            subprocess.run(cmd, check=True)
    
    print(f"Video processing complete. Output saved to: {output_path}")
    print(f"Processed frames: {processed_count} out of {frame_count} frames")
    print(f"Target FPS: {target_fps}, Actual FPS: {original_fps/frame_sampling_rate:.2f}")

if __name__ == "__main__":
    input_video = "video_2.mp4"
    output_video = "output/segmented_video.mp4"
    
    # Process only the first 100 frames (approximately 10 seconds at original FPS)
    process_video(input_video, output_video, target_fps=15, max_frames=100)
