import sys
from full_flow import process_pdf
from pymongo import MongoClient
import requests
import json
from groq import Groq  # Import the Groq library

# 🔑 Groq API setup
# IMPORTANT: Replace "gsk_Your_Key_Here" with your actual Groq API key
GROQ_API_KEY = "fill your groq api key here" 

def ask_llama(user_query, pdf_context=""):
    try:
        # Initialize the Groq client
        client = Groq(api_key=GROQ_API_KEY)

        # The prompt should be structured as a list of message objects
        messages = []
        
        # Add a system prompt if PDF context is provided
        if pdf_context.strip():
            system_prompt = f"""
            You are a helpful assistant.
            Use the following PDF context if relevant, otherwise use your general knowledge.

            PDF Context:
            {pdf_context}
            """
            messages.append({"role": "system", "content": system_prompt})
        
        # Add the user's question
        messages.append({"role": "user", "content": user_query})

        # Make the API call using the client.chat.completions.create method
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",  # Groq's model name for Llama 3 8B
            temperature=0.5,
            max_tokens=400
        )
        
        # 🔍 Add debug prints here
        print("\n--- DEBUG INFO ---")
        print("🔍 Model:", "llama3-8b-8192")
        print("🔍 Messages:", json.dumps(messages, indent=2))
        print("🔍 Raw Response (Groq object):", chat_completion)
        print("--- END DEBUG ---\n")

        # The generated content is in chat_completion.choices[0].message.content
        return chat_completion.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ Error from Groq API: {e}"

# ---------------- Chatbot ----------------
def chatbot():
    print("🤖 Welcome! Upload a PDF and I'll process it for you.")

    # 1. User se PDF path lena
    pdf_path = input("📂 Enter PDF path: ").strip()

    # 2. Process the PDF using full_flow_edit.py
    try:
        db, headings, collection = process_pdf(pdf_path)
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        sys.exit(1)

    # 3. Unique headings list banana
    unique_headings = list(set(headings))
    if not unique_headings:
        print("⚠️ No headings found in the document.")
        return

    # ---- OUTER LOOP (Domains loop) ----
    while True:
        print("\n📌 Available Domains/Headings:")
        for i, heading in enumerate(unique_headings, 1):
            print(f"{i}. {heading}")

        choice = input("\n👉 Enter heading number OR type 'custom' to ask your own question OR 'exit' to quit: ").strip()

        if choice.lower() == "exit":
            print("👋 Exiting chatbot. Goodbye!")
            break

        if choice.lower() == "custom":
            user_query = input("\n❓ Enter your custom question: ")

            # Step 1 → Try DB match first
            db_answer = collection.find_one(
                {"question": {"$regex": user_query, "$options": "i"}},
                {"_id": 0, "question": 1, "answer": 1}
            )

            if db_answer and db_answer.get("answer"):
                print(f"\n📄 PDF Answer: {db_answer['answer']}")
            else:
                # Step 2 → No PDF answer → ask Llama (general knowledge)
                answer = ask_llama(user_query)
                print(f"\n💡 Llama Answer: {answer}")

            continue

        # ---- If domain is chosen ----
        try:
            choice = int(choice)
            selected_heading = unique_headings[choice - 1]
        except (ValueError, IndexError):
            print("❌ Invalid choice. Try again.")
            continue

        print(f"\n✅ You selected: {selected_heading}\n")

        # ---- INNER LOOP (Questions for that domain) ----
        while True:
            questions = list(collection.find(
                {"heading": selected_heading},
                {"_id": 0, "question": 1, "answer": 1}
            ))

            if not questions:
                print(f"⚠️ No questions found for heading: {selected_heading}")
                break

            print(f"💡 Questions under {selected_heading}:")
            for i, q in enumerate(questions, 1):
                print(f"{i}. {q['question']}")

            q_choice = input(
                "\n👉 Enter question number for answer OR type 'back' to choose another domain / 'exit' to quit / 'custom' to ask your own: "
            ).strip()

            if q_choice.lower() == "exit":
                print("👋 Exiting chatbot. Goodbye!")
                sys.exit(0)

            if q_choice.lower() == "back":
                print("↩️ Going back to domain selection...")
                break

            if q_choice.lower() == "custom":
                user_query = input("\n❓ Enter your custom question: ")

                db_answer = collection.find_one(
                    {"question": {"$regex": user_query, "$options": "i"}},
                    {"_id": 0, "question": 1, "answer": 1}
                )

                if db_answer and db_answer.get("answer"):
                    print(f"\n📄 PDF Answer: {db_answer['answer']}")
                else:
                    # fallback → let Llama answer
                    answer = ask_llama(user_query)
                    print(f"\n💡 Llama Answer: {answer}")

                input("\n🔁 Press Enter to continue...")
                continue

            try:
                q_choice = int(q_choice)
                selected_q = questions[q_choice - 1]
            except (ValueError, IndexError):
                print("❌ Invalid choice. Try again.")
                continue

            print(f"\n❓ Question: {selected_q['question']}")
            print(f"📄 Answer: {selected_q.get('answer', '⚠️ No answer found in DB')}")

            input("\n🔁 Press Enter to continue...")  # Wait before showing list again


if __name__ == "__main__":
    print(ask_llama("Who is the prime minister of India?"))
    chatbot()