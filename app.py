from flask import Flask, request, jsonify
from flask_cors import CORS
from bot import get_response

app = Flask(__name__)

# Enable CORS to allow requests from your Blogspot frontend
CORS(app)

# Add a simple root route for testing and to avoid "Not Found" errors
@app.route('/')
def home():
    return "Chatbot API is running!"

# Define route for chatbot interaction
@app.route('/predict', methods=['POST'])
def pick_reply():
    msg = request.get_json().get('message')  # Get the message from the frontend
    response = get_response(msg)  # Call the function to get a chatbot response
    message = {'answer': response}  # Prepare the response data
    return jsonify(message)  # Send response as JSON

if __name__ == "__main__":
    app.run(debug=True)


