# !pip install -q streamlit
# !pip install -q langchain langchain-community langchain-core transformers==4.52.4 pypdf

# %%writefile app.py

import streamlit as st
import torch
import re
import json
import tempfile
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader



st.set_page_config(page_title="AI Career Coach",layout="wide")
st.title("AI Career Coach")
st.write("Upload your CV and get a career analysis based on your target role.")

@st.cache_resource
def loadModel():
    modelName = "mistralai/Mistral-Nemo-Instruct-2407"
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    model = AutoModelForCausalLM.from_pretrained(modelName,torch_dtype=torch.float16,device_map="auto")

    return tokenizer, model


with st.spinner("Loading AI model..."):
    tokenizer, model = loadModel()



def generateText(prompt, max_length=3000, num_return_sequences=1):

    inputDevice = model.device

    if inputDevice.type == "meta":
        inputDevice = torch.device("cpu")

    inputs = tokenizer(prompt,return_tensors="pt").to(inputDevice)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        top_k=30,
        top_p=0.90,
        temperature=0.7,
    )

    return [
        tokenizer.decode(output,skip_special_tokens=True)
        for output in outputs
    ]


st.header("Candidate Information")

upCV = st.file_uploader(
    "Upload your CV",
    type=["pdf"]
)

target = st.text_input(
    "Target Role",
    placeholder="Example: Machine Learning Engineer"
)


descriptions = """
Software Developer:
Requires programming, OOP, Data Structures,
Algorithms, Git, Databases, and projects.

Frontend Developer:
Requires HTML, CSS, JavaScript,
Git, APIs, and frontend projects.

Backend Developer:
Requires programming, OOP, SQL,
Databases, APIs, Git, and backend projects.

Full Stack Developer:
Requires HTML, CSS, JavaScript,
Programming, SQL, Databases, APIs,
Git, and full stack projects.

Data Analyst:
Requires Excel, SQL, Statistics,
Python, Pandas, Data Visualization,
and data analysis projects.

Data Scientist:
Requires Python, Statistics, SQL,
NumPy, Pandas, Machine Learning,
and data science projects.

Machine Learning Engineer:
Requires Python, Statistics, NumPy,
Pandas, Machine Learning, Scikit-learn,
and ML projects.

AI Engineer:
Requires Python, Mathematics,
Machine Learning, Deep Learning,
and AI projects.

Cybersecurity Analyst:
Requires Networking, Linux,
Cybersecurity Fundamentals,
Security Tools, and security labs.

DevOps Engineer:
Requires Linux, Git, Networking,
Docker, CI/CD, and deployment projects.
"""


roadmap = """
Software Developer:
Programming → OOP → Data Structures →
Algorithms → Git → Databases → Projects

Frontend Developer:
HTML → CSS → JavaScript →
Git → APIs → Frontend Framework → Projects

Backend Developer:
Programming → OOP → SQL →
Databases → APIs → Backend Framework →
Git → Projects

Full Stack Developer:
HTML → CSS → JavaScript →
Programming → SQL → Databases →
APIs → Backend Framework → Git → Projects

Data Analyst:
Excel → SQL → Statistics →
Python → Pandas → Data Visualization → Projects

Data Scientist:
Python → NumPy → Pandas → Statistics → SQL → Machine Learning → Projects

Machine Learning Engineer:
Python → NumPy → Pandas → 
Statistics → Machine Learning → Scikit-learn → ML Projects

AI Engineer:
Python → Mathematics →
Machine Learning → Deep Learning → AI Projects

Cybersecurity Analyst:
Networking → Linux → Cybersecurity Fundamentals → Security Tools → Security Labs

DevOps Engineer:
Linux → Git → Networking →
Docker → CI/CD → Deployment Projects
"""


currentSkillsSchema = ResponseSchema(
    name="current_skills",
    description="A list of strings containing the candidate's current skills."
)

missingSkillsSchema = ResponseSchema(
    name="missing_skills",
    description="A list of strings containing skills required for the target role but missing from the candidate."
)

roadmapSchema = ResponseSchema(
    name="recommended_roadmap",
    description="A list of strings containing the recommended learning roadmap."
)

jobsSchema = ResponseSchema(
    name="suggested_jobs",
    description="A list of strings containing jobs suitable for the candidate."
)

improvementSchema = ResponseSchema(
    name="improvement_plan",
    description="A list of strings containing the candidate's improvement plan."
)


responseSchemas = [currentSkillsSchema, missingSkillsSchema,
                   roadmapSchema, jobsSchema, improvementSchema]

outputParser = StructuredOutputParser.from_response_schemas(responseSchemas)
format_instructions = outputParser.get_format_instructions()


if st.button("Analyze Career"):
    if upCV is None:
        st.error("Please upload your CV.")
        st.stop()

    if not target.strip():
        st.error("Please enter your target role.")
        st.stop()




    with st.spinner("Reading CV..."):

        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as temp_file:
            temp_file.write(upCV.getvalue())
            pdfPath = temp_file.name


        loader = PyPDFLoader(pdfPath)
        pages = loader.load()
        cvText = "\n".join(page.page_content for page in pages)

        os.remove(pdfPath)


    prompt = PromptTemplate(
        template="""
You are an AI Career Coach.

Analyze the candidate's CV for the target role.

CV:
{cv_text}

Target Role:
{target_role}

Use the following information as the career knowledge:

Job Descriptions:
{job_descriptions}

Learning Roadmaps:
{learning_roadmaps}

Tasks:

1. Extract the candidate's current skills from the CV.
2. Extract and consider the candidate's experience from the CV.
3. Compare the candidate's skills and experience with the target role requirements.
4. Identify the missing skills.
5. Create a recommended learning roadmap.
6. Suggest suitable jobs from the available jobs.
7. Create an improvement plan.

Do not invent information about the candidate.

Use only information available in the CV.

If information is missing, use an empty string or empty list.

Return ONLY valid JSON.

{format_instructions}
""",
        input_variables=[
            "cv_text",
            "target_role",
            "job_descriptions",
            "learning_roadmaps",
            "format_instructions"
        ]

    ).format(
        cv_text=cvText,
        target_role=target,
        job_descriptions=descriptions,
        learning_roadmaps=roadmap,
        format_instructions=format_instructions
    )



    with st.spinner("Analyzing your career..."):
        response = generateText(prompt,max_length=3000)[0]



    def extractJSON(text):

        pattern = r'```json\s*(.*?)\s*```'

        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            return matches[-1]

        return text

    text = extractJSON(response)

    try:
        data = json.loads(text)

    except json.JSONDecodeError:
        st.error("The model did not return valid JSON.")
        st.text(response)
        st.stop()

    st.success("Process completed!")
    st.header("Career Analysis")

    st.subheader("Current Skills")
    st.write(data.get("current_skills",""))

    st.subheader("Missing Skills")
    st.write(data.get("missing_skills",""))

    st.subheader("Recommended Roadmap")
    st.write(data.get("recommended_roadmap",""))


    st.subheader("Suggested Jobs")
    st.write(data.get("suggested_jobs",""))


    st.subheader("Improvement Plan")
    st.write(data.get("improvement_plan",""))
