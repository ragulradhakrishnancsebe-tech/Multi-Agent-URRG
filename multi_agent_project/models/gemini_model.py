from langchain_google_genai import ChatGoogleGenerativeAI


def get_gemini_llm():
    # Using Gemini 2.5 Flash
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        top_p=0.9,
        max_output_tokens=4096
    )