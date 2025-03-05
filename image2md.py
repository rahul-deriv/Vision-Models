import os
from openai import OpenAI
import PIL.Image
import base64
from io import BytesIO
from pathlib import Path
import datetime

# Initialize OpenAI client
client = OpenAI(base_url="https://litellm.deriv.ai/v1", api_key=os.getenv('LITELLM_API_KEY'))

# Create md_results directory if it doesn't exist
RESULTS_DIR = Path("md_results")
RESULTS_DIR.mkdir(exist_ok=True)

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def save_markdown(content, original_image_path):
    # Get the original image filename without extension
    image_name = Path(original_image_path).stem
    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create markdown filename
    markdown_filename = f"{image_name}_{timestamp}.md"
    # Full path for the markdown file
    markdown_path = RESULTS_DIR / markdown_filename
    
    # Save the markdown content
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    return markdown_path

def image_to_markdown(image_path):
    try:
        # Convert the image to base64
        base64_image = encode_image_to_base64(image_path)

        # Create the message with the image
        response = client.chat.completions.create(
            model="gemini-2.0-flash-001",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please analyze this image and convert its contents into well-formatted markdown. Include all relevant details and maintain a clear structure."
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
        
        markdown_content = response.choices[0].message.content
        # Save the markdown content and get the file path
        saved_path = save_markdown(markdown_content, image_path)
        return f"Markdown saved to: {saved_path}\n\nContent:\n{markdown_content}"
        
    except Exception as e:
        return f"Error processing image: {str(e)}\nType: {type(e)}"

# Example usage
if __name__ == "__main__":
    image_path = "images/dd_all_sources.png"  # Replace with your image path
    result = image_to_markdown(image_path)
    print(result)
