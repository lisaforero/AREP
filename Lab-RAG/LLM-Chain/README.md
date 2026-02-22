# LangChain LLM Chain

This project is a hands-on lab demonstrating how to build a smart AI Agent using **LangChain** and **Google Gemini**. The implementation is contained within a Jupyter Notebook, showcasing the step-by-step reasoning process of an agent.

## Architecture & components

The notebook implements a reasoning loop where the LLM decides which tools to call based on the provided context:

1.  **Engine:** `gemini-2.5-flash`.
2.  **Logic:** An Agent created with `create_agent` that manages a "Plan-and-Execute" style flow.
3.  **Tools:**
    - `get_user_location`: Extracts the city from a mock database using the `user_id`.
    - `get_weather_for_location`: Simulates weather data retrieval.
5.  **State management:** `InMemorySaver` provides short-term memory, allowing the agent to follow a conversation thread.
6.  **Output:** Pydantic-style structured output via `ResponseFormat` to ensure data consistency.

## Installation & execution

Follow these steps to get the project running locally:

### 1. Prerequisites
* Python 3.9 or higher.
* A Google AI Studio API Key (Get it at [aistudio.google.com](https://aistudio.google.com/)).

### 2. Install dependencies
Run the following command to install the required libraries:
```bash
pip install -U langchain langchain-google-genai langgraph python-dotenv
```
### 3. Environment variables
Create a file named .env in the root directory and add your API key:
```bash
GOOGLE_API_KEY=your_actual_api_key_here
```
### 4. Running the Notebook
- Open your editor (VS Code, Jupyter Lab, etc.).
- Select the .venv as your kernel.
- Run all cells sequentially.

## Examples

The notebook validates two main capabilities:

**A. Tool calling and context detection**
When asked "How's the weather?", the agent detects it doesn't know the location, calls the location tool for user_id: 1, and finds out it's in Florida.

**B. Thread isolation (Memory)**
The notebook demonstrates that User 1 (Florida) and User 2 (SF) have separate memory threads. The agent will not confuse the weather of one user with the other, even if they ask the same question.



