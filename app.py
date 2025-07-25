# app.py — Project: ScientificAnimals
# This is the main Flask application that exposes an HTTP endpoint
# where users can ask for the scientific name of an animal.
# It uses a RAG (Retrieval-Augmented Generation) pipeline defined in rag_chain.py

from flask import Flask, request, jsonify

# Import the main function from rag_chain.py responsible for handling RAG logic
# This function receives a natural language question and returns an answer using retrieval and generation techniques
from rag_chain import get_scientific_name

app = Flask(__name__)  # Initialize Flask app

# Endpoint that listens to POST requests at /scientific-name
@app.route("/scientific-name", methods=["POST"])
def scientific_name():
    # Get JSON payload from the request
    data = request.get_json()

    # Validate if 'animal' key is present
    if not data or "animal" not in data:
        return jsonify({"error": "Missing 'animal' parameter."}), 400

    # Build the question for the RAG model
    # This question is formatted in natural language so that it can be interpreted properly by the language model (LLM)
    question = f"What is the scientific name of the animal {data['animal']}?"

    # Call the retrieval-augmented function to get an answer
    answer = get_scientific_name(question)

    # Return the result as JSON
    return jsonify({"scientific_name": answer})

# Run the app on port 5000 and allow external access
# Setting host="0.0.0.0" makes the app accessible externally,
# which is necessary when deploying in a Docker container or cloud VM.
# Without it, the app would only be accessible from localhost.
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
