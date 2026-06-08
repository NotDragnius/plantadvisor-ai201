# Spec: `run_agent()`

**File:** `agent.py`
**Status:** Partially pre-filled — complete the two blank fields before implementing

---

## Purpose

Orchestrate a single conversational turn for the Plant Advisor agent. Given a user message and the conversation history, call the LLM with available tools, execute any tool calls the LLM requests, and return the final text response.

This is the core of what makes Plant Advisor an *agent* rather than a simple chatbot: the ability to decide which tools to call, use their results to inform its response, and loop until it has everything it needs.

---

## Input / Output Contract

**Inputs:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_message` | `str` | The user's current message |
| `history` | `list` | Gradio conversation history — list of `[user_msg, assistant_msg]` pairs |

**Output:** `str`

The agent's final text response for this turn. Should never be empty — if something goes wrong, return a user-readable fallback message.

---

## Design Decisions

*Read `specs/system-design.md` (especially the "How the Groq Tool Calling API Works" section) before reviewing these. Complete the two blank fields before writing any code.*

---

### Messages list structure

The messages list must start with the system prompt, then replay the conversation
history, then add the new user message. Gradio history is a list of `[user, assistant]`
pairs — convert each pair to two API-format dicts:

```python
messages = [{"role": "system", "content": SYSTEM_PROMPT}]

for user_msg, assistant_msg in history:
    messages.append({"role": "user", "content": user_msg})
    if assistant_msg:
        messages.append({"role": "assistant", "content": assistant_msg})

messages.append({"role": "user", "content": user_message})
```

---

### Initial LLM call

Pass the model, the messages list, the tool definitions, and `tool_choice="auto"`
so the LLM can decide whether to call a tool or respond directly:

```python
response = client.chat.completions.create(
    model=LLM_MODEL,
    messages=messages,
    tools=TOOL_DEFINITIONS,
    tool_choice="auto",
)
```

---

### Detecting tool calls in the response

The response object has a `choices` list. Index 0 gives the assistant message.
Check its `tool_calls` attribute — if it's truthy, the LLM wants to call tools:

```python
assistant_message = response.choices[0].message

if not assistant_message.tool_calls:
    # No tool calls — LLM has a final answer
    ...
```

---

### Appending the assistant message

When there are tool calls, append the full assistant message object to `messages`
**before** appending any tool results. The API requires this ordering — a tool
result message must immediately follow the assistant message that requested it:

```python
messages.append(assistant_message)  # must come first
```

---

### Executing and appending tool results

For each tool call, extract the name and arguments, call `dispatch_tool()`, and
append the result as a `"tool"` role message. The `tool_call_id` links this result
back to the specific tool call that requested it:

```python
for tool_call in assistant_message.tool_calls:
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)
    tool_result = dispatch_tool(tool_name, tool_args)

    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": tool_result,
    })
```

---

### Loop termination conditions

*The loop should stop when: (a) the LLM returns a response with no tool calls, OR (b) the MAX_TOOL_ROUNDS limit is reached. Describe how you will detect each condition and what you will return in each case.*

```
Condition (a) - No Tool Calls:
- Detection: Check `not assistant_message.tool_calls` after each LLM call.
- What to return: The text content of this message (`assistant_message.content`).

Condition (b) - MAX_TOOL_ROUNDS reached:
- Detection: Maintain a counter `rounds` incremented each time we process a round of tool calls. Before invoking tool execution, if `rounds >= MAX_TOOL_ROUNDS`, we break out of the loop.
- What to return: If the last message contains text content, we return `assistant_message.content`. If it is None or empty (because the LLM only returned tool_calls), we return a fallback message: "I apologize, I reached my research limit for this request. Please try asking again or simplifying your question."
```

---

### Extracting the final text response

*Once the loop exits because there are no more tool calls, how do you extract the text content from the response object? What field holds the string you should return?*

```
The assistant message is obtained via `assistant_message = response.choices[0].message`.
The text content is stored in the `content` field of this message: `assistant_message.content`.
We should strip it and check that it is a non-empty string.
```

---

## Implementation Notes

**Trace of a working agent turn (what tools were called and in what order):**

```
Query: "How often should I water my snake plant in winter?"
Round 1 tool call: lookup_plant({'plant_name': 'snake plant'})
Round 2 tool call: get_seasonal_conditions({'season': 'winter'})
Final response: Recommends watering the snake plant once a month or less in winter, citing both the plant's high drought tolerance and winter seasonal care adjustments.
```

**What happens when you ask about a plant that isn't in the database?**

```
The agent calls lookup_plant() and receives a 'not found' response listing available plants. Based on the instruction in the message, the agent politely tells the user that the plant is not in its database, offers general care tips using its general LLM knowledge (e.g. noting that string of pearls is a succulent), and asks the user for details about the plant's environment.
```

**One thing about the tool call API that surprised you:**

```
The API returns arguments as a raw JSON string rather than a parsed dictionary. Additionally, if the LLM decides to omit optional parameters, it may pass 'null' or empty arguments, which can cause JSON decoding to return `None` rather than an empty dictionary, requiring defensive code like `if not isinstance(tool_args, dict): tool_args = {}` to avoid errors.
```
