from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
from flask_cors import CORS
app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:8000"}})
CORS(app, resources={r"/*": {"origins": "*"}}) 
 
@app.route('/')
def index():
    return "Hello World!"

@app.route('/process-image', methods=['POST'])
def process_image():
    data = request.json.get('imageData')
    # data is expected to be something like: "data:image/png;base64,iVBORw0KGgo..."
    if data and data.startswith('data:image/png;base64,'):
        base64_str = data.replace('data:image/png;base64,', '')
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes))

        # Run your Python function here, e.g.:
        # result = your_python_function(image)

        # For demonstration, let's say we just return a success message:
        return jsonify({'status': 'success', 'message': 'Image processed successfully'})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid image data'}), 400

if __name__ == '__main__':
    app.run(debug=True)
