'''
-Query Refinement 
-Policy Retriever
-Decision Maker 
-Json output 
'''

### What fixes are to be done ?? -> Prompts(refiner does more, the query parser is not uniform and clauses need to be properly mentioned. document name not to be mentioned.)

from store import DocumentVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
from langgraph.types import Command
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Literal, Optional, Annotated, Sequence
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

MODEL = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
VECTORSTORE = DocumentVectorStore()
PATH = r"C:\Users\TGshi\Desktop\LangChain\AppLogic\docs"

class Output(BaseModel):
    Decision: Literal["Approved", "Rejected"] = Field(description="Approved or Rejected")
    Amount: Optional[float] = Field(description="Any charges if applicable")
    Justification: str = Field(description="Explaination on the decision taken with appropriate clause stated.")

class AgentState(TypedDict):
    query : str
    messages : Annotated[Sequence[BaseMessage], add_messages]
    output : Output

@tool
def ClauseRetriever(query : str):
    """
    Searches the query provided against the documents to fetch the relevant/related information.
    """
    try:
        results = VECTORSTORE.search_documents(query, k=5)
        return results
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"

tools = [ClauseRetriever]

def RefineQuery(state : AgentState):
    """
    Refines the user query to be passed to decision maker to produce better results.
    """
    query = state["query"]
    
    refine_prompt = f"""
    You are a query refinement assistant for an insurance policy decision system.
    
    Original query: {query}
    
    Your task is to refine this query to make it more specific and suitable for searching policy documents.
    Focus on:
    1. Identifying key insurance terms and concepts
    2. Making the query more specific and remove any ambiguities with educated assumptions.
    3. Don't explicitly add new content and change the meaning of original prompt. Keep it simple and original.
    
    Provide only the refined query, nothing else.
    """
    
    response = MODEL.invoke([HumanMessage(content=refine_prompt)])
    refined_query = response.content
    
    
    return Command(
        goto="initialise-model",
        update={
            "query": refined_query
        }
    )

def InitialiseModel(state : AgentState):
    """
    Here we will make so that system instructions will be appended to messages as SystemMessage and
    refined query will be then appended as HumanMessage for invoking the DecisionMaker model in the 
    next node and also to work with MemorySaver nicely.
    """
    system_prompt = """You are an insurance policy decision assistant. Your role is to:
    1. Analyze insurance queries against policy documents
    2. Use the ClauseRetriever tool to search for relevant policy clauses
    3. Make decisions based on the retrieved information
    4. Provide clear justification for your decisions"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["query"])
    ]
    
    return Command(
        goto="decision-maker",
        update={"messages": messages}
    )

def DecisionMaker(state : AgentState):
    """
    Inputs the query and the fetches information as many times as needed with different self produced 
    queries using RefineQuery tool and outputs the decision with proper explaination mentioning the 
    clauses used to arrive to the decision.
    """
    model_with_tools = MODEL.bind_tools(tools)
    response = model_with_tools.invoke(state["messages"])
    
    if response.tool_calls:
        return Command(
            goto="tools",
            update={"messages": [response]}
        )
    else:
        return Command(
            goto="output-structurer",
            update={"messages": [response]}
        )

def OutputStructurer(state : AgentState):
    """
    Take the last decision made by the Decision Maker model and returns the output in a structure schema
    which will the update output in state. This should not append to messages.
    """
    last_message = state["messages"][-1]
    
    structure_prompt = f"""
    Based on the following decision analysis, create a structured output:
    
    {last_message.content}
    
    Extract and format the information into:
    - Decision: "Approved" or "Rejected"
    - Amount: Any charges if applicable (as a number, or null if none)
    - Justification: Clear explanation with policy clauses mentioned
    """
    
    structured_model = MODEL.with_structured_output(Output)
    output = structured_model.invoke([HumanMessage(content=structure_prompt)])
    
    return {"output": output}

memory = MemorySaver()

tool_node = ToolNode(tools=tools)
graph = StateGraph(AgentState)

graph.add_node("query-refiner", RefineQuery)
graph.add_node("initialise-model", InitialiseModel)
graph.add_node("decision-maker", DecisionMaker)
graph.add_node("output-structurer", OutputStructurer)
graph.add_node("tools", tool_node)

graph.add_edge(START, "query-refiner")
graph.add_edge("tools", "decision-maker")

app = graph.compile(checkpointer=memory)

config = {"configurable" : {
    "thread_id" : 1
}}

for filename in os.listdir(PATH):
    file_path = os.path.join(PATH, filename)
    if os.path.isfile(file_path):
        try:
            VECTORSTORE.store_file(file_path, filename)
            print(f"Successfully processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

print("Document loading completed!")

prompt = input("Enter Prompt: (EX- 46M, knee surgery, Pune, 3-month policy)") #f"46M, knee surgery, Pune, 3-month policy"

for event in app.stream({"query": prompt, "messages": []}, config=config, stream_mode="updates"):
    node_name = list(event.keys())[0]
    update = event[node_name]
    
    print(f"\n🔹 NODE: {node_name}")
    
    for key, value in update.items():
        print(f"🔸 UPDATED: {key}")
        
        if key == "messages":
            for msg in value:
                role = type(msg).__name__
                print(f"🔹 [{role}]: {getattr(msg, 'content', str(msg))}")
        elif key == "output":
            print(f"🔹 Decision: {value.Decision}")
            print(f"🔸 Amount: {value.Amount}")
            print(f"🔹 Justification: {value.Justification}")
            # for API jsonify this.
        elif key == "query":
            print(f"🔹 REFINED QUERY: {value}")
        else:
            pass