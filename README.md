## **Learning GenAI with LangChain**

### model using by
- langchain_llms or specific by langchain_ollama
- ollama for free use by locally and no need to download weights everytime using API inference

---

1. Prompting and applying Prompt template different 

- prompts by langchain_core.prompts package all imports below class
    1. PromptTemplate()
    2. ChatPromptTemplate()
    3. MessageInterface

- also SystemMessage(), HumanMessage(), AIMessage()
- create chatbot.py as also with precious chat context simple and specific we can create it by this method

---

2. Chaining and using 3 different technique
    - sequential chaining
    - parallel chaining
    - conditional chaining

---

3. Runnables used in chaining
- runnables by langchain_core.runnables
- RunnableSequence and '|' pipe operater are same cause RunnableSequence() is used in every other Runnable's too so for easy code writing use pipe(|) operater as sequence chain

    - RunnableParallel()
    - RunnableBranch()(used for conditional chaining)
    - RunnableLambda()

---
### for structured output use pydantic method of python as get structured output from LLM or Chatllm

    
## RAG (retrieved Augumented generation )
- so RAG is combination of information and languange generation which is retrived relevent chunks from knowledge and feed user prompt and retrived chunk to LLM and get output from it. by this accurate and Grounded response we get.
- it is better for privacy.
- up to date information with LLMs.

1. **Document Loader:**
- 
Everything in LangChain's RAG pipeline uses this standard format:
Document
├── page_content  → the actual text string
└── metadata      → info about the source
                    {"source": "file.pdf", "page": 3}

- TextLoader()
- PDFLoader()
- DirectoryLoader(path='',glob='*.pdf// data/*.csv// ',loader_cls=PyPDFLoader(as needed if txt files then TextLoader))
    - Glob is a file pattern filter. It tells DirectoryLoader which files to pick up:
    - "*.txt"    → all .txt files in folder
    - "*.pdf"    → all PDF files in folder
    - "**/*.pdf" → all PDFs including subfolders (** means recursive)
    - "*.csv"    → all CSV files

- CSVLoader()
- WEBBASELoader(url)


load method and lazy load function

