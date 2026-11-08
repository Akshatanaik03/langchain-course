from dotenv import load_dotenv

load_dotenv()
import os


def main():
    print("Hello, World!")
    print("Your OpenAI API Key is:", os.getenv("OPEN_AI_API_KEY"))


if __name__ == "__main__":
    main()
