from flask import Flask, request, render_template
import os
import constants
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

app = Flask(__name__)

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = constants.APIKEY

# Initialize the loader and index
loader = DirectoryLoader('data/', glob='*.txt')
index = VectorstoreIndexCreator().from_loaders([loader])

# Create a function to interact with the model
def generate_response(query):
    response = index.query(query, llm=ChatOpenAI())
    return response

# Define a route to display the input form
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle user input and display the model's response
@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        user_input = request.form['user_input']
        response = generate_response(user_input)
        return render_template('result.html', user_input=user_input, response=response)

if __name__ == '__main__':
    app.run(debug=True)
