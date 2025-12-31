try:
    import openai
    print("OpenAI: OK")
except ImportError:
    print("OpenAI: MISSING")

try:
    import groq
    print("Groq: OK")
except ImportError:
    print("Groq: MISSING")
