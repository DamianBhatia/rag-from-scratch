# Practice Building a ReAct Agent Loop.
# Language Model: llama3.2:3b
import ollama

LANGUAGE_MODEL = 'llama3.2:3b'
MAX_ITERATIONS = 4


# Define Tools
def get_current_weather(location: str):
    database = {
        "london": "15°C, Rainy 🌧️",
        "tokyo": "26°C, Sunny ☀️",
        "new york": "22°C, Windy 💨"
    }

    return database.get(location.lower(), "Weather data not available for this location.")


available_tools = {
    'get_current_weather': get_current_weather
}

while True:
    user_input = input("\nUser: ")

    messages = []
    iterations = 0

    response = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'user', 'content': user_input}
        ],
        stream=True,
        tools=[{
            'type': 'function',
            'function': {
                'name': 'get_current_weather',
                'description': 'Get the current weather for a specific city location',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'location': {
                            'type': 'string',
                            'description': 'The city name, e.g. London, Tokyo',
                        },
                    },
                    'required': ['location'],
                },
            },
        }]    
    )

    messages.append({'role': 'user', 'content': user_input})


    for _ in range(MAX_ITERATIONS):
        tool_calls = []

        # Partition LLM text response and tool responses
        for chunk in response:
            message_chunk = chunk.get('message', {})

            if 'content' in message_chunk and message_chunk['content']:
                print(f"{message_chunk['content']}", end='', flush=True)
                messages.append({'role': message_chunk['role'], 'content': message_chunk['content']})
            
            if 'tool_calls' in message_chunk and message_chunk['tool_calls']:
                tool_calls = message_chunk['tool_calls']

        if not tool_calls:
            break

        # Call tools requested by LLM
        for tool_call in tool_calls:
            func_name = tool_call['function']['name']
            func_args = tool_call['function']['arguments']

            print(f"\n⚙️ [Agent Action] LLM called tool: {func_name}({func_args})")

            if func_name in available_tools:
                tool_to_call = available_tools[func_name]
                location_arg = func_args.get('location')

                observation = tool_to_call(location_arg)

                messages.append({'role': 'tool', 'content': observation, 'name': func_name})

        response = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=messages,
            stream=True,
            tools=[{
                'type': 'function',
                'function': {
                    'name': 'get_current_weather',
                    'description': 'Get the current weather for a specific city location',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'location': {
                                'type': 'string',
                                'description': 'The city name, e.g. London, Tokyo',
                            },
                        },
                        'required': ['location'],
                    },
                },
            }]    
        )

        iterations += 1

    for chunk in response:
        print(chunk['message']['content'], end='', flush=True)

    print("\n✅ [Agent Finished] Conversation completed in {} iterations.".format(iterations))