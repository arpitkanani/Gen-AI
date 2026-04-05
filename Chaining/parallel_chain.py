from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

model = ChatOllama(model="llama3.2:3b")

# ✅ ChatPromptTemplate instead of PromptTemplate
prompt1 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful study assistant."),
    ("human", "Generate short and simple notes from this text:\n{text}")
])

prompt2 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful study assistant."),
    ("human", "Generate 5 short question answers from this text:\n{text}")
])

prompt3 = ChatPromptTemplate.from_messages([
    ("system", "You are a document formatter."),
    ("human", "Merge these notes and quiz into one clean document.\nNotes: {notes}\nQuiz: {quiz}")
])

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | model | parser,
    "quiz":  prompt2 | model | parser
})

merge_chain = prompt3 | model | parser
chain = parallel_chain | merge_chain


text="""

**1. Introduction**

ISRO is established in 1969 as part of India's Ministry of Defence to develop and operate India's space programme, aiming to achieve self-sufficiency in satellite technology.

**2. History**

ISRO was formed through the merger of INCAE and NPRL on November 29, 1969, with M. Sarajit as its first director.

**3. Mission**

ISRO's mission is to promote space exploration, develop and operate satellites, conduct space missions, and ensure national security through space-based technologies.

**4. Organisation Structure**

ISRO is divided into three main departments: Headquarters, SDC (Satellite Development Centre), and ISRL (Indian Space Research Laboratory) in Bengaluru, Karnataka.

**5. Satellite Missions**

ISRO has launched numerous satellites for various purposes, including communication, navigation, remote sensing, weather forecasting, and Earth observation.

**6. Spacecraft**

ISRO has developed several spacecraft, such as Indra, RISAT, Cartosat-2 series, GeoSat, and Mangalyaan, which have contributed to India's space programme. 

**7. Robotic Missions**

ISRO has sent robotic missions to explore the Moon and Mars using Gaganyaan mission and Mangalyaan spacecraft.

**8. Notable Achievements**

ISRO has achieved notable milestones, including being the first human in space (Rakesh Sharma), launching India's first communication satellite (SSB-1) and navigation satellite (GLONASS).

**9. Challenges and Future Plans**

ISRO faces challenges such as budget constraints, technological advancements, and international cooperation to overcome, while planning to increase funding and develop new technologies.

**10. Conclusion**

ISRO is a critical component of India's national space programme, aiming to achieve self-sufficiency in satellite technology through continued development and exploration of space.
"""

result=chain.invoke({"text":text})
print(result)
