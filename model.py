import streamlit as st
import speech_recognition as sr
from google_trans_new import google_translator
import pyttsx3
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

chat_history = []

def translate_text(text, target_language):
    translator = google_translator()
    translated_text = translator.translate(text, lang_tgt=target_language)
    return translated_text

def text_to_speech(text, language):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150) 
    engine.say(text)
    engine.runAndWait()

def recognize_and_translate(target_language, input_language):
    r = sr.Recognizer()

    with sr.Microphone() as source:
        st.write("Speak now...")
        audio = r.listen(source)

    try:
        speech_text = r.recognize_google(audio, language=input_language)
        st.write(f"Recognized Speech: {speech_text}")

        translated_text = translate_text(speech_text, target_language) 
        st.write(f"Translated to {target_language}: {translated_text}")

        st.write(f"Speaking in {target_language}...")
        text_to_speech(translated_text, target_language)
        return translated_text

    except sr.UnknownValueError:
        st.write('Speech recognition could not understand audio.')
        return None
    except sr.RequestError:
        st.write('Could not request results from Google Speech Recognition.')
        return None

def get_pdf_text():
    text = ""
    pdf_reader = PdfReader("data.pdf")
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(output_language, input_language, user_question=None):
    if user_question is None:
        user_question = st.text_input("Enter your query here:")

    if user_question:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()

        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        ans = response["output_text"]

        translated_response = translate_text(ans, output_language)
        st.write(translated_response)
        text_to_speech(translated_response, output_language)
        
        chat_history.append(("User", user_question))
        chat_history.append(("Bot", translated_response))


def main():
    st.header("NITW Query Bot")
    input_languages = {
        "English": "en",
        "Telugu": "te",
        "Hindi": "hi",
        "Bengali": "bn",
        "Tamil": "ta",
        "Malayalam": "ml",
        "Urdu": "ur",
        "Arabic": "ar",
        "Kannada": "kn",
        "Bihari": "bh",
        "Spanish": "es",
        "French": "fr"
    }
    input_language = st.selectbox("Select Input Language for ", list(input_languages.keys()))
    user_question = st.text_input("Enter your query here:")

    if st.button("Speak"):
        speak_text = recognize_and_translate("en", input_language)  
        if speak_text:
            user_question = speak_text

    output_languages = {
        "English": "en",
        "Telugu": "te",
        "Hindi": "hi",
        "Bengali": "bn",
        "Tamil": "ta",
        "Malayalam": "ml",
        "Urdu": "ur",
        "Arabic": "ar",
        "Kannada": "kn",
        "Bihari": "bh",
        "Spanish": "es",
        "French": "fr"
    }

    output_language = st.selectbox("Select Output Language", list(output_languages.keys()))

    if user_question:
        user_input(output_languages[output_language], input_language, user_question)

    raw_text = get_pdf_text()
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)

    st.subheader("Chat History")
    for role, text in chat_history:
        st.write(f"{role}: {text}")

if __name__ == "__main__":
    main()
