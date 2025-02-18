from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
from flask_cors import CORS
from .NeuralNetWork import NeuralNetwork,run
app = Flask(__name__)
nn=NeuralNetwork()
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def index():
    return "Hello World!"

@app.route('/process-image', methods=['POST'])
def process_image():
    print("accessing python code...")
    data = request.json.get('imageData')
    # data is expected to be something like: "data:image/png;base64,iVBORw0KGgo..."
    if data and data.startswith('data:image/png;base64,'):
        base64_str = data.replace('data:image/png;base64,', '')
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes))

        # Convert the PIL image back to base64 to send it to the client
        output_buffer = BytesIO()
        image.save(output_buffer, format='PNG')
        output_buffer.seek(0)
        returned_base64 = base64.b64encode(output_buffer.read()).decode('utf-8')
        returned_data_url = f"data:image/png;base64,{returned_base64}"
        nn.test_method()
        NeuralNetwork.test_static()
        # run()
        # Include the processed image in the JSON response
        return jsonify({
            'status': 'success',
            'message': 'Image processed successfully',
            'imageData': returned_data_url
        })
    else:
        return jsonify({'status': 'error', 'message': 'Invalid image data'}), 400


# @app.route('/debugbtn', methods=['POST'])
# def debugbtn():
#     print("Debugging Python code...")
#     data = request.json.get('imageData')
    
#     if data and data.startswith('data:image/png;base64,'):
#         base64_str = data.replace('data:image/png;base64,', '')
#         image_bytes = base64.b64decode(base64_str)
#         image = Image.open(BytesIO(image_bytes))

#         # Convert the image to grayscale (L mode for black-and-white images)
#         image = image.convert("L")

#         # Get pixel values
#         pixels = list(image.getdata())  # Returns a flat list of pixel intensity values (0-255)

#         # Optionally print each pixel value (for demonstration)
#         print("Pixel values:")
#         for i, pixel in enumerate(pixels):
#             print(f"Pixel {i}: {pixel}")
        
#         # Optionally reshape the flat list into a 2D list for easier visualization (32x32 in this case)
#         width, height = image.size
#         pixel_matrix = [pixels[i * width:(i + 1) * width] for i in range(height)]
#         print("Pixel matrix:")
#         for row in pixel_matrix:
#             print(row)

#         return jsonify({'status': 'success', 'message': 'Pixel values printed in server log'}), 200
#     else:
#         return jsonify({'status': 'error', 'message': 'Invalid image data'}), 400


def print_pixel_matrix(image):
    """
    Print the pixel values of a PIL image in a square format.
    Each row is printed on a new line, with pixel values formatted within square brackets.
    """
    # Ensure the image is in grayscale for simplicity
    image = image.convert("L")
    pixels = list(image.getdata())
    width, height = image.size
    for i in range(height):
        # Extract one row of pixel values
        row = pixels[i * width:(i + 1) * width]
        # Print the row in a bracketed format with each value padded for alignment
        print("[" + " ".join(f"{val:3}" for val in row) + "]")



@app.route('/debugbtn', methods=['POST'])
def debugbtn():
    print("Debugging Python code...")
    data = request.json.get('imageData')
    
    if data and data.startswith('data:image/png;base64,'):
        base64_str = data.replace('data:image/png;base64,', '')
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes)).convert("L")
        
        # Augment the image using the neural network's method
        augmented_images = nn.augment_image(image)
        
        # Print each augmented image's pixel matrix
        for idx, aug_img in enumerate(augmented_images):
            print(f"Augmented Image {idx} Pixel Matrix:")
            print_pixel_matrix(aug_img)
            print("\n" + "="*40 + "\n")
        
        return jsonify({'status': 'success', 'message': 'Augmented images pixel matrices printed in server log'}), 200
    else:
        return jsonify({'status': 'error', 'message': 'Invalid image data'}), 400



if __name__ == '__main__':
    app.run(debug=True)
