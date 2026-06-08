import json
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, MAX_TOOL_ROUNDS
from tools import lookup_plant, get_seasonal_conditions

_client = Groq(api_key=GROQ_API_KEY)

# ──────────────────────────────────────────────
# Tool definitions
#
# These are the schemas that tell the LLM what tools are available and how to
# call them. The LLM reads these descriptions and decides when (and how) to use
# each tool. They're already complete — your job is to implement the tool
# functions in tools.py and the agent loop below.
# ──────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_plant",
            "description": (
                "Look up care information for a specific houseplant by name. "
                "Returns detailed watering, light, humidity, and temperature requirements. "
                "Use this whenever the user asks about a specific plant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plant_name": {
                        "type": "string",
                        "description": "The plant name to look up. Can be a common name, scientific name, or nickname (e.g., 'pothos', 'devil's ivy', 'Monstera deliciosa').",
                    }
                },
                "required": ["plant_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_seasonal_conditions",
            "description": (
                "Get seasonal care adjustments for houseplants. "
                "Returns guidance on watering, fertilizing, light, and pests for the current or specified season. "
                "Use this when a user asks a season-specific question, or to complement plant care advice with seasonal context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "season": {
                        "type": "string",
                        "description": "The season to get care conditions for. If omitted, the current season is detected automatically.",
                        "enum": ["spring", "summer", "fall", "winter"],
                    }
                },
                "required": [],
            },
        },
    },
]

# ──────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a knowledgeable and friendly plant care advisor. "
    "Help users care for their houseplants by looking up specific plant information "
    "and current seasonal conditions using your available tools.\n\n"
    "Always use your tools to look up plant-specific information before answering — "
    "don't rely on your general knowledge alone. If a plant isn't in your database, "
    "say so clearly and offer general guidance based on what the user describes.\n\n"
    "Keep your advice practical and specific. Cite the source of your information "
    "when you have it (e.g., 'According to the care data for your monstera...')."
)

# ──────────────────────────────────────────────
# Tool dispatch
#
# This is already complete. It routes tool calls from the LLM to the actual
# Python functions in tools.py, and returns results as JSON strings (which is
# what the Groq API expects for tool results).
# ──────────────────────────────────────────────

def dispatch_tool(tool_name: str, tool_args: dict) -> str:
    """Route a tool call to the correct function and return the result as a JSON string."""
    print(f"  -> Tool call: {tool_name}({tool_args})")
    if tool_name == "lookup_plant":
        result = lookup_plant(tool_args["plant_name"])
    elif tool_name == "get_seasonal_conditions":
        result = get_seasonal_conditions(tool_args.get("season"))
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
    print(f"  <- Result: {json.dumps(result)[:120]}{'...' if len(json.dumps(result)) > 120 else ''}")
    return json.dumps(result)


# ──────────────────────────────────────────────
# Agent loop
# ──────────────────────────────────────────────

def run_agent(user_message: str, history: list) -> str:
    """
    Run the plant care agent for one user turn and return its response.

    The agent loop follows a specific pattern that is implemented here:
      1. Build a messages list: system prompt + conversation history + new user message
      2. Call the LLM with messages and TOOL_DEFINITIONS
      3. If the response contains tool_calls:
           a. Append the assistant message (with tool_calls) to messages
           b. For each tool call: execute via dispatch_tool(), append the result
           c. Call the LLM again with the updated messages
           d. Repeat until no more tool_calls (or MAX_TOOL_ROUNDS is reached)
      4. Return the final text response
    """
    print(f"\n--- Agent Turn ---")
    print(f"User message: {user_message}")

    # 1. Build messages list starting with System Prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Replay conversation history in the OpenAI messages format
    for item in history:
        if isinstance(item, dict):
            # Gradio 6.x format: list of {"role": "...", "content": "..."}
            messages.append({"role": item["role"], "content": item["content"]})
        elif hasattr(item, "role") and hasattr(item, "content"):
            # Gradio ChatMessage object
            messages.append({"role": item.role, "content": item.content})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            # Gradio 4.x/5.x format: [user_msg, assistant_msg]
            user_msg, assistant_msg = item
            messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})

    # Append current user message
    messages.append({"role": "user", "content": user_message})

    # 2. Tool call loop
    rounds = 0
    while True:
        try:
            response = _client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
            )
        except Exception as e:
            print(f"Error calling Groq LLM API: {e}")
            return (
                "I'm sorry, I encountered an issue connecting to the AI service. "
                "Please make sure your API key is correct and try again."
            )

        if not response.choices:
            return "I apologize, I didn't receive a response from the AI service. Please try again."

        assistant_message = response.choices[0].message

        # Check if the model has a final text response (no tool calls)
        if not assistant_message.tool_calls:
            if assistant_message.content:
                return assistant_message.content
            return "I'm sorry, I couldn't generate a text response. Please try asking again."

        # If the model wants to call tools, check safety limits
        if rounds >= MAX_TOOL_ROUNDS:
            print(f"Warning: reached MAX_TOOL_ROUNDS ({MAX_TOOL_ROUNDS}). Terminating loop.")
            if assistant_message.content:
                return assistant_message.content
            return (
                "I apologize, I reached my research limit for this request. "
                "Please try asking again or simplifying your question."
            )

        rounds += 1

        # We must append the assistant message to the history *before* tool results.
        messages.append(assistant_message)

        # Execute all requested tool calls
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            tool_id = tool_call.id

            try:
                tool_args = json.loads(tool_call.function.arguments)
                if not isinstance(tool_args, dict):
                    tool_args = {}
            except Exception as e:
                print(f"Error parsing tool arguments: {e}")
                tool_args = {}

            try:
                tool_result = dispatch_tool(tool_name, tool_args)
            except Exception as e:
                print(f"Error running tool {tool_name}: {e}")
                tool_result = json.dumps({"error": f"Failed to execute tool: {e}"})

            # Append the tool result to the conversation
            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": tool_result,
            })
