import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()


def main() -> None:
    model = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "gemma3:270m"),
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a concise, factual assistant. Summarize the well-known "
                "person named by the user. Include their background, notable "
                "work, and why they are influential. If a detail may be outdated "
                "or uncertain, say so.",
            ),
            ("human", "Summarize the key information about {person}."),
        ]
    )
    chain = prompt | model

    person = input("Whose information would you like summarized? ").strip()
    if not person:
        person = "Elon Musk"

    response = chain.invoke({"person": person})
    print(response.content)


if __name__ == "__main__":
    main()
