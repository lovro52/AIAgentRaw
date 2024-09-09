from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token
from dotenv import load_dotenv
import os
import gridfs
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import DuplicateKeyError
from io import BytesIO
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from pdf import get_index, PDFReader
import logging
import datetime
import PyPDF2  # Added for PDF parsing

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()

# MongoDB Atlas Configuration
uri = "mongodb+srv://test:" + os.getenv('MONGO_PASSWORD') + "@clustertest.kimvepl.mongodb.net/?retryWrites=true&w=majority&appName=ClusterTest"
client = MongoClient(uri, server_api=ServerApi('1'))

# Connect to your database
db = client['AI']
fs = gridfs.GridFS(db)
pdf_collection = db['PDFs']
users_collection = db['users']

# Initialize Flask app
app = Flask(__name__)

# CORS Configuration
CORS(app, supports_credentials=True, origins=["http://localhost:8080"], allow_headers=["Content-Type", "Authorization"])

# Initialize Bcrypt for password hashing
bcrypt = Bcrypt(app)

# Set up JWT
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')  # Load from .env
jwt = JWTManager(app)

# Ping the database to confirm connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Function to extract text from a PDF using PyPDF2
def extract_text_from_pdf(pdf_bytes):
    try:
        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}")
        return ""

# Load PDFs from GridFS and create query engines
def load_pdfs_from_gridfs(pdf_ids):
    engines = {}
    for pdf_id in pdf_ids:
        result = pdf_collection.find_one({"_id": pdf_id})
        if result is None:
            logging.error(f"No document found with _id {pdf_id}")
            continue

        file_name = result['filename']
        pdf_name = os.path.splitext(file_name)[0]

        # Check if the file exists in GridFS
        if not fs.exists(pdf_id):
            logging.error(f"File with _id {pdf_id} does not exist in GridFS")
            continue

        try:
            pdf_bytes = fs.get(pdf_id).read()
            pdf_text = extract_text_from_pdf(pdf_bytes)
            index = get_index([pdf_text], pdf_name)  # Assuming the get_index function builds an index from the content
            engines[pdf_name] = index.as_query_engine()
        except Exception as e:
            logging.error(f"An error occurred while processing PDF with _id {pdf_id}: {str(e)}")
    return engines

# Initialize data and query engines with existing files
pdf_files = list(pdf_collection.find({"type": "pdf"}))
pdf_ids = [file['_id'] for file in pdf_files]

pdf_engines = load_pdfs_from_gridfs(pdf_ids)

tools = [
    note_engine,
    *[
        QueryEngineTool(
            query_engine=engine,
            metadata=ToolMetadata(
                name=f"{name}_data",
                description=f"This gives detailed information about {name} the document",
            ),
        )
        for name, engine in pdf_engines.items()
    ],
]

# Initialize LLM and agent
llm = OpenAI(model="gpt-3.5-turbo")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

# Flask routes

# Register route
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    username = data.get('username')
    password = data.get('password')

    if not email or not username or not password:
        return jsonify({'error': 'Missing fields'}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

    try:
        users_collection.insert_one({
            'email': email,
            'username': username,
            'password': hashed_password
        })
        return jsonify({'message': 'User registered successfully'}), 201
    except DuplicateKeyError:
        return jsonify({'error': 'User already exists'}), 400
    except Exception as e:
        logging.error(f"Error registering user: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# Login route
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'error': 'Missing fields'}), 400

    user = users_collection.find_one({'username': username})

    if user and bcrypt.check_password_hash(user['password'], password):
        access_token = create_access_token(identity={'username': username, 'email': user['email']}, 
                                           expires_delta=datetime.timedelta(hours=1))
        return jsonify({'access_token': access_token}), 200
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

# Ask AI route
@app.route('/api/ask_ai', methods=['POST'])
def ask_ai():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    try:
        result = agent.query(question)
        return jsonify({'answer': str(result)})  # Ensure result is converted to string
    except Exception as e:
        logging.error(f"Error processing the request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# Upload files route
@app.route('/api/upload_files', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400

    files = request.files.getlist('files')
    for file in files:
        if file:
            file_id = fs.put(file, filename=file.filename)
            # Store file metadata in MongoDB
            file_metadata = {
                "filename": file.filename,
                "file_id": file_id,
                "type": "pdf"  # Assuming we are handling only PDFs for this task
            }
            pdf_collection.insert_one(file_metadata)

            # Update engines with the new file content
            pdf_engines.update(load_pdfs_from_gridfs([file_id]))

    return jsonify({'message': 'Files uploaded successfully'}), 200

if __name__ == '__main__':
    app.run(debug=True)