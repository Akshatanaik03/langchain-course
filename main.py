import os

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

load_dotenv()


class SearchSummary(BaseModel):
    """Structured summary of a web search."""

    answer: str = Field(description="A concise answer to the user's question")
    key_points: list[str] = Field(
        description="Important facts from the search results"
    )
    sources: list[str] = Field(
        description="Source URLs taken from the provided search results"
    )


@tool
def web_search(query: str) -> str:
    """Search the web for current information using DuckDuckGo."""
    results = DDGS().text(query, max_results=5)
    if not results:
        return "No search results were found."

    return "\n\n".join(
        f"Title: {result.get('title', '')}\n"
        f"URL: {result.get('href', '')}\n"
        f"Snippet: {result.get('body', '')}"
        for result in results
    )


def main() -> None:
    model = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "gemma3:270m"),
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0")),
    )
    structured_model = model.with_structured_output(
        SearchSummary,
        method="json_schema",
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's question using the web search results below. "
                "Do not invent facts or source URLs. If the results are "
                "insufficient, say so in the answer and use an empty "
                "key_points list when appropriate.",
            ),
            (
                "human",
                "Question: {query}\n\nWeb search results:\n{search_results}",
            ),
        ]
    )
    chain = prompt | structured_model

    query = input("What would you like to search for? ").strip()
    if not query:
        print("Please enter a search query.")
        return

    print(f"\n[Tool call] web_search(query={query!r})")
    try:
        search_results = web_search.invoke({"query": query})
    except DDGSException as error:
        print(f"\nDuckDuckGo search failed: {error}")
        print(
            "Check your internet connection or try a more specific query, "
            "such as 'current weather in Bangalore India'."
        )
        return

    print("\n[Tool results]")
    print(search_results)

    response = chain.invoke({"query": query, "search_results": search_results})

    print("\n[Final answer]")
    print(response.model_dump_json(indent=2))


# Not yet worked on: OpenAI + Tavily agent example.
# This example needs OpenAI and Tavily API keys and their LangChain packages.
# def openai_tavily_search_agent() -> None:
#     """Let the OpenAI model decide whether to search with Tavily."""
#     from langchain.agents import create_agent
#     from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
#     from langchain_openai import ChatOpenAI
#     from langchain_tavily import TavilySearch
#
#     model = ChatOpenAI(
#         model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
#         temperature=0,
#     )
#     search_tool = TavilySearch(max_results=5)
#     agent = create_agent(model=model, tools=[search_tool])
#
#     query = input("What would you like to search for? ").strip()
#     if not query:
#         print("Please enter a search query.")
#         return
#
#     result = agent.invoke({"messages": [HumanMessage(content=query)]})
#     for message in result["messages"]:
#         if isinstance(message, AIMessage):
#             for tool_call in message.tool_calls:
#                 print(f"[Tool call] {tool_call['name']}")
#                 print(f"[Tool input] {tool_call['args']}")
#         elif isinstance(message, ToolMessage):
#             print(f"[Tool result: {message.name}]")
#             print(message.content)
#
#     print("\n[Final answer]")
#     print(result["messages"][-1].content)


if __name__ == "__main__":
    main()