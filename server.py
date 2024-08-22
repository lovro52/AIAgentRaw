from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import pandas as pd
import logging
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from pdf import load_pdfs, pdf_engines

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
bcrypt = Bcrypt(app)

# Secret key for JWT
app.config['JWT_SECRET_KEY'] = os.getenv("JWT_SECRET_KEY", "super-secret-key")
jwt = JWTManager(app)

# Mock database
users = {}

# Setup logging
logging.basicConfig(level=logging.DEBUG)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_csvs(csv_paths):
    engines = {}
    for csv_path in csv_paths:
        csv_name = os.path.basename(csv_path).split('.')[0]
        df = pd.read_csv(csv_path)
        query_engine = PandasQueryEngine(df=df, verbose=True, instruction_str=instruction_str)
        query_engine.update_prompts({"pandas_prompt": new_prompt})
        engines[csv_name] = query_engine
    return engines

# List of CSV files
csv_paths = [os.path.join("data", "population.csv"), os.path.join("data", "movies.csv")]
csv_engines = load_csvs(csv_paths)

tools = [
    note_engine,
    *[
        QueryEngineTool(
            query_engine=engine,
            metadata=ToolMetadata(
                name=f"{name}_data",
                description=f"This gives information about {name} data",
            ),
        )
        for name, engine in csv_engines.items()
    ],
    *[
        QueryEngineTool(
            query_engine=engine.as_query_engine(),
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

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    username = data.get('username')
    password = data.get('password')

    if not email or not username or not password:
        return jsonify({'error': 'Email, username, and password are required'}), 400

    if username in users:
        return jsonify({'error': 'Username already exists'}), 400

    # Hash the password before storing it
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

    # Save the user in the mock database
    users[username] = {
        'email': email,
        'password': hashed_password
    }

    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400

    user = users.get(username)
    if not user or not bcrypt.check_password_hash(user['password'], password):
        return jsonify({'error': 'Invalid credentials'}), 401

    # Create a JWT token
    access_token = create_access_token(identity=username)
    return jsonify({'access_token': access_token}), 200

@app.route('/api/ask_ai', methods=['POST'])
@jwt_required()
def ask_ai():
    current_user = get_jwt_identity()
    app.logger.debug(f"User {current_user} is making a query")
    
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    try:
        result = agent.query(question)
        return jsonify({'answer': str(result)})  # Ensure result is converted to string
    except Exception as e:
        app.logger.error(f"Error processing the request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload_files', methods=['POST'])
@jwt_required()
def upload_files():
    current_user = get_jwt_identity()
    app.logger.debug(f"User {current_user} is uploading files")
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400

    files = request.files.getlist('files')
    for file in files:
        if file:
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            
            # Update engines if the file is a CSV or PDF
            if filename.endswith('.csv'):
                csv_engines.update(load_csvs([file_path]))
            elif filename.endswith('.pdf'):
                pdf_engines.update(load_pdfs([file_path]))

    return jsonify({'message': 'Files uploaded successfully'}), 200

if __name__ == '__main__':
    app.run(debug=True)



""" from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from pdf import canada_engine
import logging

# Load environment variables
load_dotenv()

# Initialize data and query engines
population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)

population_query_engine = PandasQueryEngine(
    df=population_df, verbose=True, instruction_str=instruction_str
)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})

# Define tools
tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_data",
            description="This gives information about the world population and demographics",
        ),
    ),
    QueryEngineTool(
        query_engine=canada_engine,
        metadata=ToolMetadata(
            name="canada_data",
            description="This gives detailed information about Canada the country",
        ),
    ),
]

# Initialize LLM and agent
llm = OpenAI(model="gpt-3.5-turbo")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup logging
logging.basicConfig(level=logging.DEBUG)

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
        app.logger.error(f"Error processing the request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
 """