from datetime import datetime
from typing import Generator, Optional, Union, Literal, TypedDict
from .models import (
    OpenAIChatCompletionChunkResponse,
    OpenAIChatCompletionMessage,
    OpenAIChatModels,
    OpenAIChatParams,
    OpenAIChat,
)


def _stream_chat_response(
    openai,
    chat_params: OpenAIChatParams,
) -> Generator[OpenAIChatCompletionChunkResponse, None, None]:

    resp = openai.ChatCompletion.create(
        stream=True, **chat_params.dict(exclude_none=True)
    )
    for chunk in resp:
        chunk_resp = OpenAIChatCompletionChunkResponse(**chunk.to_dict())
        yield chunk_resp


StreamTextT = Union[Literal["role"], Literal["content"]]


def stream_chat_text(
    openai,
    messages: list[OpenAIChatCompletionMessage],
    model: OpenAIChatModels = "gpt-3.5-turbo",
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop: Optional[Union[str, list[str]]] = None,
    max_tokens: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[dict[str, float]] = None,
    name: Optional[str] = None,
) -> Generator[tuple[str, StreamTextT], None, None]:

    chat_params = OpenAIChatParams(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        user=name,
    )

    for chunk in _stream_chat_response(openai, chat_params):
        if (role := chunk.choices[0].delta.role) is not None:
            yield role, "role"
        elif (content := chunk.choices[0].delta.content) is not None:
            yield content, "content"


class NewMessageT(TypedDict):
    role: str
    content: list[str]


def make_chat(
    openai,
    next_message: str,
    messages: Optional[list[OpenAIChatCompletionMessage]] = None,
    model: OpenAIChatModels = "gpt-3.5-turbo",
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop: Optional[Union[str, list[str]]] = None,
    max_tokens: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[dict[str, float]] = None,
    name: Optional[str] = None,
    display_live=True,
    display_all=False,
) -> OpenAIChat:

    if messages is None:
        messages = []
    all_messages = messages.copy()
    all_messages.append(
        OpenAIChatCompletionMessage(role="user", content=next_message, name=name)
    )

    for message in all_messages:
        if display_all:
            print(str(message))

    chat_params = OpenAIChatParams(
        model=model,
        messages=all_messages,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        user=name,
    )

    new_message: NewMessageT = {"role": "assistant", "content": []}

    created = datetime.now()
    for data, dt in stream_chat_text(openai, **chat_params.dict(exclude_none=True)):
        if dt == "role":
            new_message["role"] = data
            if display_live:
                print("> {}: ".format(data), end="", flush=True)
        elif dt == "content":
            new_message["content"].append(data)
            if display_live:
                print(data, end="", flush=True)
        else:
            if display_live:
                print("\n")

    chat_params.messages.append(
        OpenAIChatCompletionMessage(
            role=new_message["role"], content="".join(new_message["content"])
        )
    )

    return OpenAIChat(created=created, params=chat_params)


def continue_chat(
    openai,
    next_message: str,
    prev_chat: OpenAIChat,
    display_live=True,
    display_all=False,
) -> OpenAIChat:
    return make_chat(
        openai,
        next_message,
        display_live=display_live,
        display_all=display_all,
        **prev_chat.params.dict(exclude_none=True),
    )
