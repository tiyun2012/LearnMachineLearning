from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
from flask_cors import CORS
from .NeuralNetWork import NeuralNetwork

app = Flask(__name__)
nn = NeuralNetwork()
CORS(app, resources={r"/*": {"origins": "*"}})

def convert_to_data_url(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return "data:image/png;base64," + base64.b64encode(buffer.read()).decode('utf-8')

@app.route('/')
def index():
    return "Hello World!"

@app.route('/process-image', methods=['POST'])
def process_image():
    print("accessing python code...")
    data = request.json.get('imageData')
    if data and data.startswith('data:image/png;base64,'):
        base64_str = data.replace('data:image/png;base64,', '')
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes))
        output_buffer = BytesIO()
        image.save(output_buffer, format='PNG')
        output_buffer.seek(0)
        returned_base64 = base64.b64encode(output_buffer.read()).decode('utf-8')
        returned_data_url = f"data:image/png;base64,{returned_base64}"
        nn.test_method()
        NeuralNetwork.test_static()
        return jsonify({
            'status': 'success',
            'message': 'Image processed successfully',
            'imageData': returned_data_url
        })
    else:
        return jsonify({'status': 'error', 'message': 'Invalid image data'}), 400

@app.route('/debugbtn', methods=['POST'])
def debugbtn():
    print("Debugging Python code...")
    data = request.json.get('imageData')
    if data and data.startswith('data:image/png;base64,'):
        base64_str = data.replace('data:image/png;base64,', '')
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes)).convert("L")
        
        # Generate 100 augmented images using the combined transformation method
        augmented_images = nn.augment_image_combined(
            image,
            num_images=100,
            rotation_range=(-30, 30),
            translation_ratio=1/3,
            scale_ratio=1/3
        )
        print("Augmented images generated:", len(augmented_images))
        
        # Convert augmented images to data URLs
        augmented_urls = [convert_to_data_url(img) for img in augmented_images]
        
        return jsonify({
            'status': 'success',
            'message': '100 augmented images returned',
            'augmented': augmented_urls
        }), 200
    else:
        return jsonify({'status': 'error', 'message': 'Invalid image data'}), 400

if __name__ == '__main__':
    app.run(debug=True)
