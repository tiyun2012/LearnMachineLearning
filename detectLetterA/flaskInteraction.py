from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
from flask_cors import CORS
from .NeuralNetWork import NeuralNetwork
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

        # Include the processed image in the JSON response
        return jsonify({
            'status': 'success',
            'message': 'Image processed successfully',
            'imageData': returned_data_url
        })
    else:
        return jsonify({'status': 'error', 'message': 'Invalid image data'}), 400

@app.route('/debug', methods=['POST'])
def debug():
    print("Debugging python code...")

    # For demonstration, let's assume we have a sample image or a generated image.
    # You could reuse the image data from the last request, or load a static image from disk.
    # Let's say we have a static image "debug_image.png" in the same folder.
    with open("debug_image.png", "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode('utf-8')
    returned_data_url = f"data:image/png;base64,{encoded}"

    return "----------test debug--------------", 200

if __name__ == '__main__':
    app.run(debug=True)
