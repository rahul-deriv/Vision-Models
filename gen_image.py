import os
from openai import OpenAI
import base64
from io import BytesIO
from pathlib import Path
import datetime
from PIL import Image

# Initialize OpenAI client
client = OpenAI(base_url="https://litellm.deriv.ai/v1", api_key=os.getenv('LITELLM_API_KEY'))

# Create images directory if it doesn't exist
IMAGES_DIR = Path("generated_images")
IMAGES_DIR.mkdir(exist_ok=True)

def generate_image(prompt, size="1024x1024", model="imagen-3.0-fast-generate-001"):
    """
    Generate an image based on a text prompt using an AI model.
    
    Args:
        prompt (str): The text description of the image to generate
        size (str): Image size (default: "1024x1024")
        model (str): The model to use (default: "dall-e-3")
        
    Returns:
        Path: Path to the saved image file
    """
    try:
        # Call the API to generate the image
        response = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            n=1,
            response_format="b64_json"
        )
        
        # Get the base64 encoded image data
        image_data = response.data[0].b64_json
        
        # Decode the base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Create a filename based on the prompt and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a safe filename from the prompt (first 30 chars)
        safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:30]
        filename = f"{safe_prompt}_{timestamp}.png"
        
        # Save the image
        image_path = IMAGES_DIR / filename
        image.save(image_path)
        
        return image_path
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    # prompt = input("Enter a prompt for image generation: ")
    prompt = "Coder in a room with 5 screens coding AI apps."
    image_path = generate_image(prompt)
    
    if image_path:
        print(f"Image generated and saved to: {image_path}")
        
        # # Optional: Convert the generated image to markdown
        # try:
        #     from image2md import image_to_markdown
        #     markdown_result = image_to_markdown(str(image_path))
        #     print("\nConverted to markdown:")
        #     print(markdown_result)
        # except ImportError:
        #     print("\nTo convert this image to markdown, use the image2md.py script.")
