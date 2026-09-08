import os
import glob
from typing import Annotated, Sequence
from typing import TypedDict

# LangChain & LangGraph components
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# -----------------------------------------------------------------------------
# 1. Initialize Local LLM and Embedding Model
# -----------------------------------------------------------------------------
llm = ChatOllama(
    model="llama3.2", 
    temperature=0
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# -----------------------------------------------------------------------------
# 2. Build the Document Ingestion & RAG Pipeline
# -----------------------------------------------------------------------------
def initialize_knowledge_base(docs_dir: str = "./documents"):
    """Loads documents, splits into chunks, and stores into a FAISS vector index."""
    raw_docs = []
    for file_path in glob.glob(f"{docs_dir}/*.txt"):
        loader = TextLoader(file_path)
        raw_docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(raw_docs)
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 2})

retriever = initialize_knowledge_base()

# -----------------------------------------------------------------------------
# 3. Define Tools
# -----------------------------------------------------------------------------
@tool
def lookup_college_knowledge(query: str) -> str:
    """Searches official college documents, rules, grading policies, syllabi, and FAQs."""
    docs = retriever.invoke(query)
    if not docs:
        return "No specific document found matching your query."
    return "\n\n".join([doc.page_content for doc in docs])

@tool
def check_exam_schedule(department: str) -> str:
    """Retrieves upcoming mid-semester and end-semester dates by department."""
    schedules = {
        "computer science": "Midterm: CS201 Data Structures on Oct 10, OS on Oct 14.",
        "electrical": "Midterm: EE101 Circuit Theory on Oct 11, Signals on Oct 15.",
        "mechanical": "Midterm: ME202 Thermodynamics on Oct 12, Fluid Mechanics on Oct 16."
    }
    return schedules.get(
        department.strip().lower(),
        f"No timetable found for '{department}'. Please check with your department coordinator."
    )

# Bundle tools and attach them to the model
tools = [lookup_college_knowledge, check_exam_schedule]
llm_with_tools = llm.bind_tools(tools)

# -----------------------------------------------------------------------------
# 4. Define Agent State and Graph Workflow
# -----------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

SYSTEM_PROMPT = SystemMessage(
    content="You are a helpful College Support Assistant. Always use your search tool "
            "to answer policy, syllabus, and FAQ questions accurately. Keep track of the "
            "student's personal details (name, department) to personalize your answers."
)

def agent_node(state: AgentState):
    """Invokes LLM with system context and conversation history."""
    messages = [SYSTEM_PROMPT] + list(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """Decides if tools need to run or if the agent is ready to respond."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# Graph Construction
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

# Enable Short-Term / Multi-turn Session Memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# -----------------------------------------------------------------------------
# 5. Interactive Chat Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    session_config = {"configurable": {"thread_id": "student_turn_session_1"}}

    print("=" * 60)
    print("🎓 AI Student Support Assistant Ready (Type 'exit' to quit)")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nStudent: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye! Best of luck with your studies.")
                break

            response = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=session_config
            )

            assistant_reply = response["messages"][-1].content
            print(f"\nAssistant: {assistant_reply}")

        except KeyboardInterrupt:
            print("\nSession ended.")
            break