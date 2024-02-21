import vertexai
import streamlit as st
from vertexai.preview import generative_models
from vertexai.preview.generative_models import GenerativeModel, Tool, Part, Content, ChatSession
from services.flight_manager import search_flights, book_flights

project = "gemini-explorer-414913"
vertexai.init(project = project)

# Define Tool for Search Flights
get_search_flights = generative_models.FunctionDeclaration(
    
    name = "get_search_flights",
    description = "Tool for searching a flight with origin, destination, and departure date",
    
    parameters = {
        "type": "object",
        "properties": {
            "origin": {
                "type": "string",
                "description": "The airport of departure for the flight given in airport code such as LAX, SFO, BOS, etc."
            },
            "destination": {
                "type": "string",
                "description": "The airport of destination for the flight given in airport code such as LAX, SFO, BOS, etc."
            },
            "departure_date": {
                "type": "string",
                "format": "date",
                "description": "The date of departure for the flight in YYYY-MM-DD format"
            },
        },
        "required": [
            "origin",
            "destination",
            "departure_date"
        ]
    },
)

# Define Tool for Search Booking
get_book_flight = generative_models.FunctionDeclaration(
    
    name = "get_book_flight",
    description = "Tool for searching a booked flight with flight ID, type of seat, and number of seats",
    
    parameters = {
        "type": "object",
        "properties": {
            "flight_id": {
                "type": "integer",
                "description": "The ID of flights, in a integer; such as: 1, 2, 3, and so on"
            },
            "seat_type": {
                "type": "string",
                "description": "The type of seat flight, in a string; such as: economy, business, or first_class"
            },
            "num_seats": {
                "type": "integer",
                "description": "The number of seats, in a integer; such as: 1, 2, 3, and so on"
            },
        },
        "required": [
            "origin",
            "destination",
            "departure_date"
        ]
    },
)

# Define tool and model with tools
search_tool = generative_models.Tool(
    function_declarations = [get_search_flights, 
                             get_book_flight],
)

config = generative_models.GenerationConfig(
    temperature = 0.4,
    top_p = 0.59,
    top_k = 9,
    max_output_tokens = 2048
)

# Load model with config
model = GenerativeModel(
    "gemini-pro",
    tools = [search_tool],
    generation_config = config
)

# Helper function to unpack responses
def handle_response(response):
    
    # Check for function call with intermediate step, always return response
    if response.candidates[0].content.parts[0].function_call.args:
        
        # Function call exists, unpack and load into a function
        response_args = response.candidates[0].content.parts[0].function_call.args
        
        function_params = {}
        for key in response_args:
            value = response_args[key]
            function_params[key] = value

        function_name = response.candidates[0].content.parts[0].function_call.name
        
        if function_name == "get_search_flights":
            
            results = search_flights(**function_params)
            
            if results:
                
                intermediate_response = chat.send_message(
                    Part.from_function_response(
                        name = "get_search_flights",
                        response = results
                    )
                )
                
                return intermediate_response.candidates[0].content.parts[0].text
            
            else:
                return "Search Failed!"
            
        elif function_name == "get_book_flight":
            
            results = book_flights(**function_params)
            
            if results:
                
                intermediate_response = chat.send_message(
                    Part.from_function_response(
                        name = "get_book_flight",
                        response = results
                    )
                )
                
                return intermediate_response.candidates[0].content.parts[0].text
            
            else:
                return "Booking Failed!"

    else:
        
        # Return just text
        return response.candidates[0].content.parts[0].text

# Helper function to display and send streamlit messages
def model_answer(chat, query):
    
    response = chat.send_message(query)
    output = handle_response(response)
    
    with st.chat_message("model"):
        st.markdown(output)
    
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })
    
    st.session_state.messages.append({
        "role": "vertex-ai-model",
        "content": output
    })

st.title("Gemini Flights")

chat = model.start_chat()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display and load to chat history
for message in st.session_state.messages:
    
    content = Content(
        role = message["role"],
        parts = [Part.from_text(message["content"])]
    )
    
    chat.history.append(content)

# For Initial message startup
if len(st.session_state.messages) == 0:
    
    # Invoke initial message
    initial_prompt = "My name is Vertex, an assistant powered by Google Gemini."
    model_answer(chat, initial_prompt)

# For capture user input
query = st.chat_input("What can Vertex do for you, friend?")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    
    model_answer(chat, query)