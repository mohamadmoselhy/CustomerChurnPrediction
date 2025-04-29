from PIL import Image, ImageDraw, ImageFont
import os

def create_logo():
    # Create a new image with a white background
    width, height = 200, 200
    image = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    
    # Draw a circle
    circle_bbox = [(20, 20), (180, 180)]
    draw.ellipse(circle_bbox, fill=(41, 128, 185, 255))  # Blue color
    
    # Draw a chart-like shape inside the circle
    points = [(100, 50), (150, 100), (100, 150), (50, 100)]
    draw.polygon(points, fill=(255, 255, 255, 255))
    
    # Save the image
    logo_path = os.path.join(os.path.dirname(__file__), 'logo.png')
    image.save(logo_path)
    print(f"Logo created and saved at: {logo_path}")

if __name__ == "__main__":
    create_logo() 