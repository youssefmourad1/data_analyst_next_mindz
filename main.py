import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from groq import Groq
import sqlite3
import pandasql as psql
from dotenv import load_dotenv

# LangChain imports
from langchain.agents import AgentExecutor, Tool, AgentType, initialize_agent
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from langchain import PromptTemplate

# Load environment variables
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(page_title="Advanced Data Analyst Expert", layout="wide")

# Title of the app
st.title("📊 Advanced Data Analyst Expert")

# Sidebar for uploading data
st.sidebar.header("1. Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Sidebar for API keys
st.sidebar.header("2. API Keys Configuration")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY"))

# Initialize Groq client
if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
else:
    groq_client = None
    st.sidebar.warning("Please enter your Groq API key.")

# Function to perform EDA
def perform_eda(df):
    st.subheader("🧮 Exploratory Data Analysis (EDA)")

    st.write("### 📈 Data Overview")
    st.dataframe(df.head())

    st.write("### 📊 Summary Statistics")
    st.write(df.describe())

    st.write("### ❓ Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        st.write(missing)
    else:
        st.write("No missing values detected.")

    st.write("### 📊 Data Types")
    st.write(df.dtypes)

# Function to perform Groq API call
def perform_groq_analysis(prompt):
    if groq_client:
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a highly skilled senior data analyst assistant."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192"
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            st.error(f"Groq API call failed: {e}")
            return None
    else:
        st.warning("Groq client is not configured.")
        return None

# Function to execute SQL queries
def execute_sql_query(df, query):
    try:
        # Connect to an in-memory SQLite database
        conn = sqlite3.connect(':memory:')
        df.to_sql('data', conn, index=False, if_exists='replace')
        result = psql.sqldf(query, conn)
        conn.close()
        return result
    except Exception as e:
        st.error(f"SQL Query failed: {e}")
        return None

# Function to create correlation matrix
def plot_correlation_matrix(df, selected_columns):
    st.subheader("📈 Correlation Matrix")
    try:
        numeric_df = df[selected_columns].select_dtypes(include=[np.number])
        if numeric_df.empty:
            st.warning("No numerical columns available for correlation matrix.")
            return
        corr = numeric_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Failed to plot correlation matrix: {e}")

# Function to create t-SNE plot (Removed as per user request)
# Function to create interactive correlation matrix
def interactive_correlation_matrix(df):
    st.subheader("📈 Interactive Correlation Matrix")
    try:
        columns = df.columns.tolist()
        selected_cols = st.multiselect("Select columns for correlation matrix:", options=columns, default=columns[:5])
        if selected_cols:
            plot_correlation_matrix(df, selected_cols)
        else:
            st.info("Please select at least two columns to display the correlation matrix.")
    except Exception as e:
        st.error(f"Failed to create interactive correlation matrix: {e}")

# Function to encode non-numerical columns
def encode_non_numerical_columns(df):
    try:
        non_numeric_cols = df.select_dtypes(include=['object', 'category']).columns
        le = LabelEncoder()
        for col in non_numeric_cols:
            df[col] = le.fit_transform(df[col].astype(str))
        return df
    except Exception as e:
        st.error(f"Error encoding non-numerical columns: {e}")
        return df

# Function to suggest plots using Groq
def suggest_plots(df, user_context=""):
    st.subheader("🔍 Suggested Visualizations")

    if groq_client:
        # Encode non-numerical data to avoid errors in plots
        df_encoded = encode_non_numerical_columns(df.copy())

        prompt = f"""
        You are an experienced senior data analyst. Based on the following data summary and user context, suggest three types of plots that would be most insightful for Exploratory Data Analysis.

        Data Summary:
        {df_encoded.describe().to_string()}

        User Context:
        {user_context}

        Provide your suggestions in a numbered list.
        """
        suggestions = perform_groq_analysis(prompt)
        if suggestions:
            st.write(suggestions)
            return suggestions
    else:
        st.warning("Groq client is not configured properly.")
    return ""

# Function to suggest next steps using Groq
def suggest_next_steps(df, user_context=""):
    st.subheader("🔄 Suggested Next Steps")
    if groq_client:
        # Encode non-numerical data
        df_encoded = encode_non_numerical_columns(df.copy())

        prompt = f"""
        You are an intelligent senior data analysis assistant. Based on the following data summary and user context, suggest the next three steps for a comprehensive analysis.

        Data Summary:
        {df_encoded.describe().to_string()}

        User Context:
        {user_context}

        Provide your suggestions in a numbered list.
        """
        suggestions = perform_groq_analysis(prompt)
        if suggestions:
            st.write(suggestions)
    else:
        st.warning("Groq client is not configured properly.")

# Function to generate plots based on suggestions from Groq
def generate_plots(df, suggestions):
    st.subheader("📊 Generated Visualizations")
    suggestions_list = [s.strip() for s in suggestions.split('\n') if s.strip()]
    for suggestion in suggestions_list:
        if suggestion.startswith(tuple(['1.', '2.', '3.'])):
            plot_desc = suggestion.split('.', 1)[1].strip().lower()
            if 'bar plot' in plot_desc or 'bar chart' in plot_desc:
                st.write("#### Bar Chart")
                if df.select_dtypes(include=[np.number]).shape[1] >= 1:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    plt.figure(figsize=(10,6))
                    sns.countplot(data=df, x=numeric_cols[0])
                    plt.xticks(rotation=45)
                    st.pyplot(plt)
            elif 'histogram' in plot_desc:
                st.write("#### Histogram")
                if df.select_dtypes(include=[np.number]).shape[1] >= 1:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    plt.figure(figsize=(10,6))
                    sns.histplot(data=df, x=numeric_cols[0], bins=30, kde=True)
                    st.pyplot(plt)
            elif 'scatter plot' in plot_desc:
                st.write("#### Scatter Plot")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    plt.figure(figsize=(10,6))
                    sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1])
                    st.pyplot(plt)
            elif 'box plot' in plot_desc:
                st.write("#### Box Plot")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 1:
                    plt.figure(figsize=(10,6))
                    sns.boxplot(data=df, y=numeric_cols[0])
                    st.pyplot(plt)
            elif 'heatmap' in plot_desc:
                st.write("#### Heatmap")
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    corr = numeric_df.corr()
                    plt.figure(figsize=(10,8))
                    sns.heatmap(corr, annot=True, cmap='viridis')
                    st.pyplot(plt)
            elif 'pair plot' in plot_desc:
                st.write("#### Pair Plot")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    sns.pairplot(df[numeric_cols])
                    st.pyplot(plt)
            # Add more plot types as needed

# Custom LangChain LLM Wrapper for Groq
class GroqLLM(LLM):
    def __init__(self, groq_client):
        self.groq_client = groq_client

    @property
    def _llm_type(self):
        return "groq"

    def _call(self, prompt, stop=None):
        response = perform_groq_analysis(prompt)
        if response:
            return response
        else:
            return ""

    @property
    def _identifying_params(self):
        return {}

# Define Tool functions
def execute_sql_query_tool(query):
    if 'data' not in st.session_state:
        return "No data uploaded."
    df = st.session_state.data
    result = execute_sql_query(df, query)
    if result is not None:
        return result.to_string()
    else:
        return "SQL Query failed."

def generate_plots_tool(suggestions):
    if 'data' not in st.session_state:
        return "No data uploaded."
    df = st.session_state.data
    generate_plots(df, suggestions)
    return "Plots generated successfully."

def clean_data_tool(instruction):
    if 'data' not in st.session_state:
        return "No data uploaded."
    df = st.session_state.data.copy()
    try:
        if "handle missing values" in instruction.lower():
            df = df.fillna(method='ffill').fillna(method='bfill')
        if "remove duplicates" in instruction.lower():
            df = df.drop_duplicates()
        if "correct data types" in instruction.lower():
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except:
                        pass
        st.session_state.data = df
        return "Data cleaned successfully."
    except Exception as e:
        return f"Data cleaning failed: {e}"

def feature_engineer_tool(instruction):
    if 'data' not in st.session_state:
        return "No data uploaded."
    df = st.session_state.data.copy()
    try:
        if "create new feature" in instruction.lower():
            # Example: Create a new feature by combining two existing features
            if len(df.columns) >= 2:
                df['new_feature'] = df.iloc[:,0] * df.iloc[:,1]
        if "encode categorical" in instruction.lower():
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if not categorical_cols.empty:
                le = LabelEncoder()
                for col in categorical_cols:
                    df[col] = le.fit_transform(df[col].astype(str))
        st.session_state.data = df
        return "Feature engineering completed successfully."
    except Exception as e:
        return f"Feature engineering failed: {e}"

# Function to set up LangChain Agent with Groq
def setup_agent():
    if not groq_client:
        st.error("Groq client is not configured.")
        return None

    # Define tools
    tools = [
        Tool(
            name="SQL Query",
            func=execute_sql_query_tool,
            description="Executes SQL queries on the uploaded data. Use this to retrieve or manipulate data using SQL."
        ),
        Tool(
            name="Plot Generator",
            func=generate_plots_tool,
            description="Generates plots based on the provided suggestions. Use this to visualize data insights."
        ),
        Tool(
            name="Data Cleaner",
            func=clean_data_tool,
            description="Performs data cleaning tasks such as handling missing values, removing duplicates, and correcting data types."
        ),
        Tool(
            name="Feature Engineer",
            func=feature_engineer_tool,
            description="Performs feature engineering tasks such as creating new features, encoding categorical variables, and scaling numerical features."
        ),
        # Add more advanced tools as needed
    ]

    # Initialize memory
    memory = ConversationBufferMemory(memory_key="chat_history")

    # Initialize custom Groq LLM
    groq_llm = GroqLLM(groq_client=groq_client)

    # Define the agent
    agent = initialize_agent(
        tools,
        groq_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )

    return agent

# Function to run code with retry mechanism
def run_code_with_retry(code, retries=3):
    for attempt in range(retries):
        try:
            exec(code, globals())
            return "Code executed successfully."
        except Exception as e:
            st.error(f"Error executing code: {e}")
            if attempt < retries - 1:
                st.info(f"Retrying... ({attempt + 1}/{retries})")
            else:
                return f"Failed to execute code after {retries} attempts."

# Main section
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.data = df  # Store data in session state
        st.success("✅ Data loaded successfully!")

        # User can add context about the data
        st.sidebar.header("3. Add Data Context")
        user_context = st.sidebar.text_area("Provide additional context or information about your data (optional):")

        perform_eda(df)
        plot_suggestions = suggest_plots(df, user_context=user_context)

        # Suggest next steps
        st.markdown("---")
        suggest_next_steps(df, user_context=user_context)

        # Display suggested plots
        st.markdown("---")
        st.header("📊 Visualizations")

        # Interactive Correlation Matrix
        interactive_correlation_matrix(df)

        # Generate additional plots based on suggestions
        if plot_suggestions:
            st.markdown("---")
            st.header("🖼️ Generate Suggested Plots")
            generate_plots(df, plot_suggestions)

        # SQL Query section
        st.markdown("---")
        st.header("🗄️ Execute SQL Queries on Your Data")
        st.write("You can write SQL queries to manipulate or analyze your data.")
        sql_query = st.text_area("Enter your SQL query here (e.g., SELECT * FROM data WHERE column > value):")
        if st.button("Run SQL Query"):
            if sql_query:
                sql_result = execute_sql_query(df, sql_query)
                if sql_result is not None:
                    st.write("### SQL Query Result")
                    st.dataframe(sql_result)
            else:
                st.error("Please enter a SQL query.")

        # Groq Analysis section
        st.markdown("---")
        st.header("🤖 Groq Analysis")
        analysis_prompt = st.text_area("Enter your analysis prompt here:")
        if st.button("Run Groq Analysis"):
            if analysis_prompt:
                result = perform_groq_analysis(analysis_prompt)
                if result:
                    st.write(result)
            else:
                st.error("Please enter a prompt for analysis.")

        # Decision-Making Agent
        st.markdown("---")
        st.header("🧠 Decision-Making Assistant")
        agent_prompt = st.text_area("Ask the Decision-Making Agent for insights or recommendations:")
        if st.button("Get Recommendation"):
            if agent_prompt:
                agent = setup_agent()
                if agent:
                    with st.spinner("Processing your request..."):
                        response = agent.run(agent_prompt)
                    st.write(response)

                    # After getting the response, provide actionable suggestions
                    st.markdown("---")
                    st.subheader("🔄 What would you like to do next?")
                    suggestions = [
                        "Deep Dive into Specific Insights",
                        "Generate Additional Plots",
                        "Perform Data Cleaning",
                        "Feature Engineering",
                        "Run Another Analysis"
                    ]
                    cols = st.columns(len(suggestions))
                    for idx, suggestion in enumerate(suggestions):
                        with cols[idx]:
                            if st.button(suggestion):
                                additional_context = st.text_input(f"Provide additional context for '{suggestion}':")
                                if st.button(f"Confirm '{suggestion}'"):
                                    extended_prompt = f"{suggestion}. Additional Context: {additional_context}"
                                    response = agent.run(extended_prompt)
                                    st.write(response)
                                    # Optionally, you can also run code generated by the agent
                                    # For safety, ensure that only trusted code is executed
                                    # Here is a simplistic implementation
                                    if "```python" in response:
                                        code = response.split("```python")[1].split("```")[0]
                                        st.code(code, language='python')
                                        run_confirmation = st.button("Run Generated Code")
                                        if run_confirmation:
                                            execution_result = run_code_with_retry(code)
                                            st.write(execution_result)
            else:
                st.error("Please enter a question or prompt for the agent.")

    except Exception as e:
        st.error(f"Error loading the file: {e}")
else:
    st.info("📂 Awaiting CSV file to be uploaded.")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ by Youssef from Next Mindz")
