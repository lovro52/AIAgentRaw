from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from pdf import pdf_engines

load_dotenv()

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

llm = OpenAI(model="gpt-3.5-turbo")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)


