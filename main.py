import os
import time
import numpy as np
import openai
from config import RESPONSE_PROMPT
from model import Message, SpeakerEnum
import json
from pathlib import Path

SAVE_LOCATION = Path.cwd() / "data"

from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise Exception("OPENAI_API_KEY not found in .env file")


def similarity(v1: list[int], v2: list[int]) -> int:
    result = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return result


def get_response_from_openai(user_prompt: str, convo_history: list[Message]):
    chat_log = ""
    for message in convo_history:
        chat_log += f"-{message.message}\n"

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=RESPONSE_PROMPT.format(prompt=user_prompt, chat_log=chat_log),
        temperature=0.2,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["USER:"],
    )
    model_response = response["choices"][0].text.strip()
    # Quick validation that it does indeed begin with AGENT:
    if not model_response.startswith("AGENT:"):
        return f"AGENT: {model_response}"
    return model_response


def get_latest_messages(number: int = 4) -> list[Message]:
    messages = []
    for file in os.listdir(SAVE_LOCATION):
        with open(SAVE_LOCATION / file, "r") as f:
            message = json.load(f)
            messages.append(message)
    messages = sorted(messages, key=lambda x: x["timestamp"])
    messages = [Message(**message) for message in messages]
    if number > len(messages):
        return messages
    return messages[-number:]


def save_message(message: Message):
    filename = f"{message.uuid}.json"
    with open(SAVE_LOCATION / filename, "w") as f:
        json.dump(message.dict(), f)


def remove_enqueued_message(curr_messages: list[Message], curr_limit: int = 4):
    if len(curr_messages) > curr_limit:
        curr_messages.pop(0)
        curr_messages.pop(0)
    return curr_messages


if __name__ == "__main__":
    if not os.path.exists(SAVE_LOCATION):
        os.mkdir(SAVE_LOCATION)

    convo_history = get_latest_messages(4)

    while True:
        user_message = input("User: ")
        userMessage = Message(
            timestamp=time.time(), message=user_message, speaker=SpeakerEnum.USER
        )
        save_message(userMessage)
        response = get_response_from_openai(user_message, convo_history)
        print(response)

        modelResponse = Message(
            timestamp=time.time(), message=response, speaker=SpeakerEnum.AGENT
        )
        save_message(modelResponse)

        convo_history.append(userMessage)
        convo_history.append(modelResponse)
        remove_enqueued_message(convo_history)
