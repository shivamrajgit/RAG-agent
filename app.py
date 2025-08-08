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

MODEL = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
VECTORSTORE = DocumentVectorStore()
PATH = r"C:\Users\TGshi\Desktop\LangChain\AppLogic\docs"


class Output(BaseModel):
    """
    A structured format for the final adjudication output.
    """
    Decision: Literal["Approved", "Rejected"] = Field(
        ...,
        description="Binary decision. Must default to 'Approved' if information is insufficient and checkpoints must be listed under justification."
    )
    
    Amount: Optional[float] = Field(
        default=None,
        description="The approved monetary value. Only include if the Decision is 'Approved' and an amount is specified."
    )

    Justification: str = Field(
        ...,
        description="A complete explanation for the decision, which must include a direct quotation of the supporting policy clause(s) and checkpoints if any for claim."
    )


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
    Your goal is to refine the given query so it is more precise and relevant for searching insurance policy documents.
    Refinement rules:
        -Preserve original meaning — do not add, remove, or replace key medical, financial, or policy terms.
        -Clarify, not alter — you may expand abbreviations, fix grammar, or add necessary context from the query itself, but never introduce new procedures, conditions, or assumptions that are not explicitly stated.
        -Enhance search relevance — identify important insurance-related keywords (e.g., coverage, claim eligibility, exclusions, pre-authorization) and incorporate them only if they are directly implied or stated.
        -Remove vague or extraneous language to make the query concise and unambiguous.

    Return only the refined query text, nothing else.
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
    system_prompt = """
    You are an expert Insurance Policy Analyst AI. Your primary function is to adjudicate insurance queries by strictly interpreting policy documents. You must operate with objectivity, precision, and adherence to the provided information.

    Your workflow is as follows:
    1.  **Deconstruct the Query:** Identify the key facts, dates, and the specific question being asked by the user.
    2.  **Retrieve Clauses:** Use the `ClauseRetriever` tool to find all relevant policy sections, including definitions, coverage grants, exclusions, and conditions related to the query.
    3.  **Analyze and Synthesize:** Scrutinize the retrieved clauses. Compare the facts of the query directly against the policy language.
    4.  **Formulate Determination:** Based *only* on the retrieved clauses, make a clear determination (Approved/Rejected).
    5.  **Generate Final Output:** Based on your analysis, you must generate a structured output that conforms to the following schema. Do not output any other text or explanation outside of this structure.
            1. *Decision:* (Required) This field must be either "Approved" or "Rejected".
            If the policy language clearly supports the user's request, set this to "Approved".
            If the policy language excludes, limits, or fails to cover the request, set this to "Rejected".
            If the query lacks the necessary information to confirm coverage, you must default to "Approved" and Specify the unavailable information as requirement or checkpoints.
            2. *Amount:* (Optional) This field is for a numeric value.
            Include the approved monetary amount if the decision is "Approved" and an amount is specified or calculable.
            If the decision is "Rejected" or if no specific amount is applicable to the decision, this field should be omitted.
            3. *Justification:* (Required) This field is a string containing your complete reasoning.
            You must provide a clear explanation for your Decision.
            This explanation must include a direct quotation of the most critical policy clause(s) used to make the determination.

    **CRITICAL CONSTRAINTS:**
    * You must base your decision **solely** on the information retrieved from the policy document. Do not make assumptions or use external knowledge.
    * You are not a lawyer or a financial advisor. **Never** provide legal advice, financial recommendations, or opinions on the fairness of a policy.
    * If the provided information is insufficient to make a decision, your determination must be "More Information Needed" and you must specify what information is required.
    """
    
    new_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["query"])
    ]
    
    return {
        "messages": new_messages
    }

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
    - Decision: Binary decision. 'Approved' or 'Rejected'.
    - Amount: The approved monetary value. Only include if the Decision is 'Approved' and an amount is specified.
    - Justification: A complete explanation for the decision, which must include a direct quotation if any, of the supporting policy clause(s) and checkpoints neccessary.
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
graph.add_edge("initialise-model", "decision-maker")
graph.add_edge("tools", "decision-maker")

app = graph.compile(checkpointer=memory)

config = {"configurable" : {
    "thread_id" : 1
}}

# Use the new process_directory method with intelligent caching
print("Starting document processing...")
VECTORSTORE.process_directory(PATH)
print("Document processing completed!")

prompt = input("Enter Prompt-> (EX- 46M, knee surgery, Pune, 3-month old policy)\n:") #f"46M, knee surgery, Pune, 3-month policy" #f"46M, knee surgery, Pune, 3-month policy"

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

# Utility functions for cache management
def clear_cache():
    """Clear the vectorstore cache - use this if you want to force rebuild"""
    VECTORSTORE.clear_vectorstore()
    print("Cache cleared. Next run will process all files fresh.")

def show_cache_info():
    """Show information about the current cache state"""
    info = VECTORSTORE.get_cache_info()
    print("\n=== Cache Information ===")
    print(f"Vectorstore loaded: {info['vectorstore_exists']}")
    print(f"Cache directory: {info['cache_directory']}")
    print(f"Cached files count: {info['cached_files_count']}")
    if info['cached_files']:
        print("Cached files:")
        for file in info['cached_files']:
            print(f"  - {file}")
    print("========================\n")

def force_rebuild():
    """Force rebuild of vectorstore from all files"""
    print("Force rebuilding vectorstore...")
    VECTORSTORE.process_directory(PATH, force_rebuild=True)
    print("Force rebuild completed!")

# Uncomment any of these lines to use the utility functions:
# show_cache_info()
# clear_cache()
# force_rebuild()