from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
import os
import gradio as gr
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Read credentials from environment
MODEL_ID = os.environ["MODEL_ID"]
ENDPOINT_BASE_URL = os.environ["ENDPOINT_BASE_URL"]
CDP_TOKEN = os.environ["CDP_TOKEN"]

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Tools
@tool
def add(a: int, b: int):
    return a + b

@tool
def subtract(a: int, b: int):
    return a - b

@tool
def multiply(a: int, b: int):
    return a * b

tools = [add, subtract, multiply]

# LLM
llm = ChatOpenAI(
    model_name=MODEL_ID,
    openai_api_base=ENDPOINT_BASE_URL,
    openai_api_key=CDP_TOKEN
)

# Core LLM call
def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI assistant, please answer my query to the best of your ability."
    )
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

# Conditional edge logic
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

# Build graph
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")
graph.add_conditional_edges("our_agent", should_continue, {"continue": "tools", "end": END})
graph.add_edge("tools", "our_agent")

# Compile graph
app = graph.compile()

# Gradio processing function
def process_input(user_input):
    input_messages = [HumanMessage(content=user_input)]
    state_input = {"messages": input_messages}
    output = app.invoke(state_input)
    final_message = output["messages"][-1]
    return final_message.content

# Launch Gradio app
with gr.Blocks() as demo:
    gr.Markdown("## LangGraph Assistant with Math Tools")
    with gr.Row():
        user_input = gr.Textbox(lines=2, label="Enter your prompt")
    with gr.Row():
        output = gr.Textbox(label="Response")

    submit_btn = gr.Button("Submit")
    submit_btn.click(fn=process_input, inputs=user_input, outputs=output)

if __name__ == "__main__":
    demo.launch(share=False,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_APP_PORT')))
