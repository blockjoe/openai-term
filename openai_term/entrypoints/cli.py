import os
import string
from typing import Literal, Optional, Union

import openai
from pydantic import BaseSettings
from rich.console import Console
from rich.theme import Theme
import typer

from ..chat import stream_chat_text, NewMessageT
from ..models import (
    OpenAIChatModelEnum,
    OpenAIChatCompletionMessage,
    OpenAIChat,
    OpenAIChatParams,
)
from ..history import get_oldest_filename, iter_filenames, iter_filepaths


class Config(BaseSettings):
    open_api_token: str
    data_dir: str = os.path.expanduser("~/.local/share/openai-term")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


CONF = Config()

app = typer.Typer(rich_markup_mode="rich")
openai.api_key = CONF.open_api_token

cli_theme = Theme(
    {
        "user_prompt": "bold light_slate_blue",
        "user_text": "light_slate_blue",
        "ai_prompt": "bold magenta",
        "system_prompt": "bold bright_red",
        "system_text": "bright_red",
    }
)


def terminal_chat_stream(
    console: Console,
    chat_params: OpenAIChatParams, quiet: bool = False
) -> NewMessageT:
    new_message: NewMessageT = {"role": "assistant", "content": []}

    chars = 0
    for data, dt in stream_chat_text(openai, **chat_params.dict(exclude_none=True)):
        if dt == "role":
            new_message["role"] = data
            if not quiet:
                console.rule(title="assistant", style="magenta")
        elif dt == "content":
            new_message["content"].append(data)
            if quiet:
                console.print(data, end="")
            else:
                while len(data) + chars > 88 and data not in string.punctuation:
                    words = data.split(" ")
                    for i, word in enumerate(words):
                        if len(word) + 1 + chars > 88:
                            data = " ".join(words[i:])
                            data = data.lstrip()
                            console.print(" ".join(words[:i]))
                            chars = 0
                        elif word.endswith("\n"):
                            chars = 0
                        else:
                            chars += len(word) + 1

                if "\n" in data:
                    parts = data.split("\n")
                    chars = len(parts[-1]) if len(parts) > 1 else 0
                chars += len(data)
                console.print(data, end="")
    console.print()
    if not quiet:
        console.rule(style="magenta")

    return new_message


@app.command("new-chat")
def cli_chat(
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Immediately feed the prompt. If none is given, the user will be prompted for input."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Surpress any output with the exception of the next message in the stream."),
    model: OpenAIChatModelEnum = typer.Option(OpenAIChatModelEnum.gpt_35_turbo, "--model", "-m", help="The OpenAI Chat Model to utilize"),
    temperature: Optional[float] = typer.Option(None, "--temp", "-t", min=0, max=2.0,  help="The temperature, see [link]https://platform.openai.com/docs/api-reference/parameter-details[/link]"),
    top_p: Optional[float] = typer.Option(None, "--top-p", min=0, max=2.0, help="The top_p value, see [link]https://platform.openai.com/docs/api-reference/parameter-details[/link]"),
    stop: Optional[str] = typer.Option(None, "--stop", "-s", help="A phrase to stop the chat completion when reached."),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", "-m", help="The maximum amount of tokens to limit the response, no value means no limit."),
    presence_penalty: Optional[float] = typer.Option(None, "--presence-penalty", "-pp", min=-2.0, max=2.0, help="The presence penalty, see [link]https://platform.openai.com/docs/api-reference/parameter-details[/link]"),
    frequency_penalty: Optional[float] = typer.Option(None, "--frequency-penalty", "-fp", min=-2.0, max=2.0, help="The frequency penalty, see [link]https://platform.openai.com/docs/api-reference/parameter-details[/link]"),
    user: Optional[str] = typer.Option(None, help="An optional identifier for the individual sending the quieries."),
):
    """
    Begin a new AI Chat.
    """

    console = Console(theme=cli_theme, width=88)

    if prompt is None:
        console.print()
        user_message = console.input(
            "[user_prompt]user{} > [/user_prompt]".format(
                "" if user is None else "({})".format(user)
            )
        )
        console.print()
    else:
        user_message = prompt

    messages = [
        OpenAIChatCompletionMessage(role="user", content=user_message, name=user)
    ]

    chat_params = OpenAIChatParams(
        model=model.value,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        user=user,
    )

    new_message = terminal_chat_stream(console, chat_params, quiet=quiet)

    chat_params.messages.append(
        OpenAIChatCompletionMessage(
            role=new_message["role"], content="".join(new_message["content"])
        )
    )

    chat_obj = OpenAIChat(params=chat_params)
    chat_obj.persist(CONF.data_dir)


def _load_chat(chat_name: Union[str, Literal["latest"], Literal["oldest"]]) -> OpenAIChat:
    if chat_name != "latest" and chat_name != "oldest":
        if not chat_name.endswith(".chat"):
            chat_name += ".chat"
        chat_name = os.path.join(CONF.data_dir, "chats", chat_name)

    return OpenAIChat.from_file(chat_name, data_dir=CONF.data_dir)


@app.command("continue-chat")
def cli_continue(
    chat_name: str = typer.Option("latest", "--chat", "-c", help="Either 'latest', 'oldest', or the filename of the chat relative to its storage location."),
    next_prompt: Optional[str] = typer.Option(None, "--prompt", "-p ", help="Immediately feed the next prompt. If none is given, the user will be prompted for input."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Surpress any output with the exception of the next message in the stream."),
):
    """
    Continue a previous AI Chat.
    """
    console = Console(theme=cli_theme, width=88)

    chat_obj = _load_chat(chat_name)
    if not quiet:
        console.print(chat_obj)

    if next_prompt is None:
        console.print()
        user_prompt = console.input(
            "[user_prompt]user{} > [/user_prompt]".format(
                "" if chat_obj.params.user is None else "({})".format(chat_obj.params.user)
            )
        )
        console.print()
    else:
        user_prompt = next_prompt
    user_message = OpenAIChatCompletionMessage(
        role="user", content=user_prompt, name=chat_obj.params.user
    )
    new_params = chat_obj.params.copy()
    new_params.messages.append(user_message)

    new_message = terminal_chat_stream(console, new_params, quiet=quiet)

    new_params.messages.append(
        OpenAIChatCompletionMessage(
            role=new_message["role"], content="".join(new_message["content"])
        )
    )

    new_chat = OpenAIChat(params=new_params)
    new_chat.persist(CONF.data_dir, overwrites=chat_obj)


@app.command("view-chat")
def cli_view_chat(
    chat_name: str = typer.Option("latest", "--chat", "-c", help="Either 'latest', 'oldest', or the filename of the chat relative to its storage location."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Force the view of unformatted text"),
    fzf: bool = typer.Option(False, "--fzf", help="If true, force terminal output for fzf preview"),
) -> None:
    """
    Display a previous AI Chat.
    """

    if fzf:
        console = Console(theme=cli_theme, width=88, force_terminal=True)
    else:
        console = Console(theme=cli_theme, width=88)

    if ")" in chat_name:
        chat_name = chat_name.split(")")[0].strip()
    if chat_name == "Begin New Chat":
        message = OpenAIChatCompletionMessage(role="user", content="")
        console.print("Begin a New Chat")
        console.print(str(message) if quiet else message)
    else:
        chat_obj = _load_chat(chat_name)
        console.print(str(chat_obj) if quiet else chat_obj)

@app.command("delete-chat")
def cli_delete_chat(
    chat_name: str = typer.Option("latest", "--chat", "-c", help="Either 'latest', 'oldest', or the filename of the chat relative to its storage location."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Force the view of unformatted text"),
) -> None:
    """
    Display a previous AI Chat.
    """

    console = Console(theme=cli_theme, width=88)

    if "\n" in chat_name:
        chats = chat_name.split("\n")
    else:
        chats = [chat_name]
    for c in chats:
        if ")" in c:
            chat_name = c.split(")")[0].strip()
        else:
            chat_name = c
        chat_obj = _load_chat(chat_name)
        chat_obj.clear(CONF.data_dir)
        if not quiet:
            console.print("Removed chat from {}".format(chat_obj.created))

@app.command("list-chats")
def cli_list_chats(
    absolute: bool = typer.Option(False, "--abs", "-e", help="Expand to the full absolute path"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only display the chat's raw name"),
    fzf: bool = typer.Option(False, "--fzf", help="If true, force terminal output for fzf preview"),
) -> None:
    """
    List all stored AI Chats.
    """

    if fzf:
        console = Console(theme=cli_theme, width=88, force_terminal=True)
        console.print("Begin New Chat")
    else:
        console = Console(theme=cli_theme, width=88)

    for fpath in iter_filepaths("chats", ".chat", CONF.data_dir):
        outname = fpath
        if quiet:
            if not absolute:
                outname, _ = os.path.splitext(os.path.basename(outname))
            console.print(outname)
        else:
            if not absolute:
                outname, _ = os.path.splitext(os.path.basename(outname))
            chat_obj = OpenAIChat.from_file(fpath)
            content = chat_obj.params.messages[0].content.strip()
            out_text = (content[:45].strip() + "...") if len(content) > 48 else content
            out = "{}) {}".format(outname, out_text)
            console.print(out)


def main():
    app()

if __name__ == "__main__":
    main()
