# Transcend ChatGPT (fork of AI Makerspace Beyond ChatGPT Challenge)

import datetime
from typing import Any
import logging
import re

from openai import AsyncOpenAI  # importing openai for API usage
from anthropic import AsyncAnthropic
from ollama import AsyncClient

import chainlit as cl  # importing chainlit for our app
from chainlit.input_widget import Select, Slider

import wikipedia as wiki
from duckduckgo_search import DDGS

from dotenv import load_dotenv

REACT_PROMPT = """You have access to the following tools:
{tools}

To use a tool, you must always use the following format:
Thought: Do I need to use a tool? Yes
Action: the action to take, must be one of {tool_names}
Input: the input to the action

When you have a response to say to the Human, or if you do not need to use a tool, you must always use the format:
Thought: Do I need to use a tool? No
Final Answer: Your well formulated response

You must strictly adhere to the above formats.
Once one block is finished, end your message immediately to begin the next loop."""

REACT_REGEXES = {
    "thought": re.compile(r"Thought[:]? Do I need to use a tool\? (.+)"),
    "action": re.compile(r"Action[:]? (.+)"),
    "input": re.compile(r"Input[:]?[ ]?[\"]?(.+)[\"]?"),
    "final": re.compile(r"Final Answer[:]? ([\s\S]+)")
}

ROLES = {
    "Cheery Chatbot": {
        "system": "You are a helpful assistant who always speaks in a pleasant tone!",
        "user": "{input}\nThink through your response step by step."
    },
    "Internet Investigator": {
        "system": "You are an intelligent web searcher who effectively writes search queries,"
        "receives search results, and summarizes their contents, giving sources for all information."
        f"The current date is {datetime.datetime.now().isoformat()}",
        "user": "{input}\nSearch the web for results relating to this request.",
        "tools": {
            "search": {
                "description": "Searches the web for webpage snippets given a search query.",
                "function": lambda query: str(DDGS().text(query, max_results=5)),
            }
        }
    },
    "Wikipedia Wizard": {
        "system": "You are an information tool for Wikipedia content.",
        "user": "{input}\nIf relevant, use your tools to find information on Wikipedia.",
        "tools": {
            "search": {
                "description": "Searches for page titles.",
                "function": lambda query: str(wiki.search(query))
            },
            "suggest": {
                "description": "Returns the suggested title for the query.",
                "function": lambda query: str(wiki.suggest(query))
            },
            "summary": {
                "description": "Gives a summary of a page given a title.",
                "function": lambda title: str(wiki.summary(title))
            },
            "page": {
                "description": "Retrieves the entire page from a title.",
                "function": lambda title: str(wiki.page(title).content)
            },
            "random": {
                "description": "Gets a list of random article titles, the number of pages can be specified.",
                "function": lambda pages=1: str(wiki.random(pages))
            },
            "set_lang": {
                "description": "Sets the language given a two character code.",
                "function": lambda code: str(wiki.set_lang(code))
            }
        }
    }
}

LLMS = {
    "gpt-4.5-preview": "openai",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gpt-4-turbo": "openai",
    "gpt-3.5-turbo": "openai",
    "claude-3-7-sonnet-latest": "anthropic",
    "claude-3-5-sonnet-latest": "anthropic",
    "claude-3-opus-latest": "anthropic",
    "claude-3-sonnet-20240229": "anthropic",
    "claude-3-haiku-20240307": "anthropic",
    "llama3.1": "ollama",
    "llama3.2": "ollama",
    "qwen2": "ollama",
    "qwq": "ollama",
    "granite3-dense": "ollama",
    "granite3-moe": "ollama",
    "mistral": "ollama",
    "mixtral": "ollama"
}

CLIENTS = {
    "openai": AsyncOpenAI,
    "anthropic": AsyncAnthropic,
    "ollama": AsyncClient
}

AGENTIC_ITERATION_LIMIT = 5

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
load_dotenv()

@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    settings = await cl.ChatSettings(
        [
            Select(
                id="role",
                label="Role",
                values=ROLES.keys(),
                initial_index=0
            ),
            Select(
                id="model",
                label="Model",
                values=LLMS.keys(),
                initial_index=0
            ),
            Slider(
                id="temperature",
                label="Temperature",
                initial=0.7,
                min=0,
                max=2,
                step=0.1
            ),
            Slider(
                id="max_tokens",
                label="Max Tokens",
                initial=512,
                min=64,
                max=2048,
                step=1
            ),
            Slider(
                id="top_k",
                label="Top K",
                initial=40,
                min=0,
                max=200,
                step=1
            ),
            Slider(
                id="top_p",
                label="Top P",
                initial=0.9,
                min=0,
                max=1,
                step=0.01
            )
        ]
    ).send()
    settings["max_tokens"] = int(settings["max_tokens"]) # Initially returned as a floating point value
    cl.user_session.set("settings", settings)
    cl.user_session.set("messages", [])

@cl.on_settings_update
async def settings_updated(settings):
    cl.user_session.set("settings", settings)

def create_message(role:str, content:str):
    return {
        "role": role,
        "content": content
    }

async def process_token(token, msg:cl.Message):
    if not token:
        token = ""
    await msg.stream_token(token)
    return token

@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    settings:dict[str, Any] = cl.user_session.get("settings")
    messages:list[dict[str, str]] = cl.user_session.get("messages")
    role:str = settings["role"]
    settings = {key: value for key, value in settings.items() if key != "role"}

    client = CLIENTS[LLMS[settings["model"]]]()

    system_prompt = ROLES[role]["system"]
    if "tools" in ROLES[role]:
        tools:dict = ROLES[role]["tools"]
        def create_tools_prompt():
            return "\n".join([f"{name}: {tool['description']}" 
                             for name, tool in tools.items()])
        def create_tool_names():
            return ", ".join([name for name in tools.keys()])
        react_prompt:str = REACT_PROMPT.format(tools=create_tools_prompt(),
                                               tool_names=create_tool_names())
        system_prompt = f"{react_prompt}\n{system_prompt}"
    user_prompt = ROLES[role]["user"].format(input=message.content)
    messages.extend([
        create_message("system", system_prompt),
        create_message("user", user_prompt)
    ])
    logger.log(level=logging.DEBUG, msg=f"user message: {messages[-2:]}")

    msg = cl.Message(content="", author=settings['model'])

    # Agentic loop
    loop = True
    stop_counter = AGENTIC_ITERATION_LIMIT
    while loop and stop_counter > 0:
        # Call LLM provider API
        response = ""
        try:
            match LLMS[settings["model"]]:
                case "openai":
                    if "top_k" in settings:
                        del settings["top_k"]
                    async for stream_resp in await client.chat.completions.create(
                        messages=messages, stream=True, **settings):
                        response += await process_token(stream_resp.choices[0].delta.content, msg)
                case "anthropic":
                    async with client.messages.stream(
                        system=system_prompt,
                        messages=[m for m in messages if m["role"] != "system"], 
                        **settings) as stream:
                            async for text in stream.text_stream:
                                response += await process_token(text, msg)
                case "ollama":
                    async for part in await client.chat(
                        model=settings["model"], messages=messages, stream=True, options=settings):
                        response += await process_token(part["message"]["content"], msg)
        except Exception as e:
            if e == "Task stopped by user":
                continue
            logger.error(f"error when streaming LLM response: {e}")
            msg = cl.ErrorMessage(f"An error occurred when streaming LLM response: {e}")
            await msg.send()
            return

        # Process ReAct paradigm responses if tools are included
        if "tools" in ROLES[role]:
            def final_answer():
                nonlocal response
                match = re.search(REACT_REGEXES["final"], response)
                if match:
                    nonlocal loop
                    response = match.group(1)
                    loop = False
                else:
                    return False
            action = "No"
            match = re.search(REACT_REGEXES["thought"], response)
            if match is None:
                if not final_answer():
                    messages.append(create_message("assistant", "Missing \"Thought:\", please follow the specified format."))
            else:
                action = match.group(1)
                if "yes" in action.lower():
                    match = re.search(REACT_REGEXES["action"], response)
                    if match is None:
                        messages.append(create_message("assistant", "Missing \"Action:\", please follow the specified format."))
                    else:
                        tool = match.group(1).strip()
                        if tool not in tools:
                            messages.append(create_message("assistant",
                                f"{tool} is not an available tool, please try: {create_tool_names()}"
                            ))
                        if "Input" in response:
                            match = re.search(REACT_REGEXES["input"], response)
                            if match is None:
                                messages.append(create_message("assistant", "Missing \"Input:\", please follow the specified format."))
                            else:
                                input = match.group(1).strip()
                                output = tools[tool]["function"](input)
                        else:
                            output = tools[tool]["function"]()
                        logger.log(level=logging.DEBUG, msg=f"{tool}: {input} -> {output}")
                        messages.append(create_message("assistant", f"Observation: {output}"))
                    await process_token("\n", msg)
                else:
                    if not final_answer():
                        messages.append(create_message("assistant", "Missing \"Final Answer:\", please follow the specified format."))
        # Otherwise, finish loop
        else:
            loop = False

        # Update the prompt object with the completion
        logger.log(level=logging.DEBUG, msg=f"{LLMS[settings["model"]]} response: {response}")
        messages.append({
            "role": "assistant",
            "content": response
        })
        cl.user_session.set("messages", messages)
        stop_counter -= 1

    if stop_counter <= 0:
        process_token("Stopped agent loop due to too many iterations.", msg)

    # Send and close the message stream
    await msg.send()
