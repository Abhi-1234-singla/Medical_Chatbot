import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MediBot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* ---------- Main Background ---------- */

    .stApp {
        background: linear-gradient(
            135deg,
            #f8fbff 0%,
            #eef7ff 50%,
            #f7fbff 100%
        );
    }


    /* ---------- Header ---------- */

    .main-header {
        background: linear-gradient(
            135deg,
            #0f766e,
            #0891b2
        );

        padding: 25px 30px;
        border-radius: 18px;

        color: white;

        box-shadow:
            0 8px 25px rgba(15, 118, 110, 0.20);

        margin-bottom: 25px;
    }

    .main-header h1 {
        margin: 0;
        font-size: 34px;
        font-weight: 700;
    }

    .main-header p {
        margin-top: 7px;
        margin-bottom: 0;

        font-size: 15px;
        opacity: 0.9;
    }


    /* ---------- Chat Messages ---------- */

    .user-message {
        background: #dff6ff;

        padding: 14px 18px;

        border-radius: 18px 18px 4px 18px;

        margin: 8px 0 8px auto;

        max-width: 75%;

        color: #164e63;

        box-shadow:
            0 3px 10px rgba(0, 0, 0, 0.05);
    }


    .assistant-message {
        background: white;

        padding: 16px 20px;

        border-radius: 18px 18px 18px 4px;

        margin: 8px auto 8px 0;

        max-width: 80%;

        color: #1f2937;

        border: 1px solid #e2e8f0;

        box-shadow:
            0 4px 15px rgba(0, 0, 0, 0.05);
    }


    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: #ffffff;

        border-right: 1px solid #e5e7eb;
    }

    .sidebar-title {
        font-size: 23px;
        font-weight: 700;

        color: #0f766e;

        margin-bottom: 5px;
    }

    .sidebar-text {
        color: #64748b;
        font-size: 14px;
        line-height: 1.6;
    }


    /* ---------- Buttons ---------- */

    .stButton > button {
        width: 100%;

        border-radius: 10px;

        border: 1px solid #cbd5e1;

        background: white;

        color: #334155;

        font-weight: 600;

        padding: 10px;
    }

    .stButton > button:hover {
        border-color: #0f766e;

        color: #0f766e;
    }


    /* ---------- Chat Input ---------- */

    .stChatInput {
        border-radius: 15px;
    }


    /* ---------- Footer ---------- */

    .footer {
        text-align: center;

        color: #94a3b8;

        font-size: 12px;

        margin-top: 30px;

        padding-bottom: 15px;
    }


    /* ---------- Info Cards ---------- */

    .info-card {
        background: white;

        border: 1px solid #e2e8f0;

        padding: 15px;

        border-radius: 14px;

        margin-top: 10px;

        box-shadow:
            0 3px 12px rgba(0,0,0,0.04);
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# VECTOR DATABASE
# =========================================================

DB_FAISS_PATH = "vectorstore/db_faiss"


@st.cache_resource
def get_vectorstore():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db


# =========================================================
# PROMPT
# =========================================================

def set_custom_prompt(custom_prompt_template):

    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

    return prompt


# =========================================================
# MAIN APP
# =========================================================

def main():

    # -----------------------------------------------------
    # SESSION STATE
    # -----------------------------------------------------

    if "messages" not in st.session_state:
        st.session_state.messages = []


    # -----------------------------------------------------
    # SIDEBAR
    # -----------------------------------------------------

    with st.sidebar:

        st.markdown(
            '<div class="sidebar-title">🩺 MediBot</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="sidebar-text">
                Your AI-powered medical information assistant.
                <br><br>
                MediBot uses Retrieval-Augmented Generation (RAG)
                to answer questions from its medical knowledge base.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        st.markdown("### 💡 What can I ask?")

        st.markdown("""
        - 💊 Medicine information
        - 🩺 Diseases & symptoms
        - 🧬 Medical concepts
        - ⚕️ Treatments
        - 🧪 Drug interactions
        - 📚 General medical knowledge
        """)

        st.divider()

        st.markdown("### ⚠️ Disclaimer")

        st.info(
            "MediBot provides informational answers only. "
            "It is not a substitute for professional medical advice."
        )

        st.divider()

        if st.button("🗑️ Clear Conversation"):

            st.session_state.messages = []

            st.rerun()


    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="main-header">

            <h1>🩺 MediBot</h1>

            <p>
                AI-powered medical assistant • RAG based •
                Ask questions about your medical knowledge base
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # WELCOME MESSAGE
    # -----------------------------------------------------

    if len(st.session_state.messages) == 0:

        st.markdown(
            """
            <div class="info-card">

                <h3>👋 Welcome to MediBot!</h3>

                <p>
                    Ask me a medical question and I'll search my
                    knowledge base to provide a relevant answer.
                </p>

                <p>
                    <b>Example:</b>
                    What is the difference between caffeine and
                    calcium channel blockers?
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # DISPLAY CHAT HISTORY
    # -----------------------------------------------------

    for message in st.session_state.messages:

        if message["role"] == "user":

            with st.chat_message(
                "user",
                avatar="👤"
            ):

                st.markdown(message["content"])

        else:

            with st.chat_message(
                "assistant",
                avatar="🩺"
            ):

                st.markdown(message["content"])


    # -----------------------------------------------------
    # USER INPUT
    # -----------------------------------------------------

    prompt = st.chat_input(
        "💬 Ask MediBot a medical question..."
    )


    if prompt:

        # -----------------------------------------------
        # DISPLAY USER MESSAGE
        # -----------------------------------------------

        with st.chat_message(
            "user",
            avatar="👤"
        ):

            st.markdown(prompt)


        # Save user message

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )


        # -----------------------------------------------
        # CUSTOM PROMPT
        # -----------------------------------------------

        CUSTOM_PROMPT_TEMPLATE = """

        Use the pieces of information provided in the context
        to answer the user's question.

        If you don't know the answer, just say that you don't know.
        Do not make up an answer.

        Only use information provided in the context.

        Context:
        {context}

        Question:
        {question}

        Start the answer directly.
        No small talk please.

        """


        # -----------------------------------------------
        # GET ANSWER
        # -----------------------------------------------

        try:

            with st.spinner("🔎 Searching medical knowledge..."):

                vectorstore = get_vectorstore()


                if vectorstore is None:

                    st.error(
                        "Failed to load the vector store."
                    )

                    return


                # ---------------------------------------
                # GROQ LLM
                # ---------------------------------------

                llm = ChatGroq(

                    model_name="openai/gpt-oss-20b",

                    temperature=0.1,

                    groq_api_key=os.environ[
                        "GROQ_API_KEY"
                    ]

                )


                # ---------------------------------------
                # RETRIEVAL QA
                # ---------------------------------------

                qa_chain = RetrievalQA.from_chain_type(

                    llm=llm,

                    chain_type="stuff",

                    retriever=vectorstore.as_retriever(
                        search_kwargs={
                            "k": 6
                        }
                    ),

                    return_source_documents=True,

                    chain_type_kwargs={
                        "prompt": set_custom_prompt(
                            CUSTOM_PROMPT_TEMPLATE
                        )
                    }

                )


                # ---------------------------------------
                # INVOKE
                # ---------------------------------------

                response = qa_chain.invoke(
                    {
                        "query": prompt
                    }
                )


                result = response["result"]


            # -------------------------------------------
            # DISPLAY ASSISTANT RESPONSE
            # -------------------------------------------

            with st.chat_message(
                "assistant",
                avatar="🩺"
            ):

                st.markdown(result)


            # Save assistant response

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result
                }
            )


        except Exception as e:

            st.error(
                f"❌ Something went wrong: {str(e)}"
            )


    # -----------------------------------------------------
    # FOOTER
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="footer">

            🩺 MediBot • RAG-powered Medical Assistant
            <br>
            For educational purposes only

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()

