from datetime import datetime
from enum import Enum
import json
import os
from typing import Literal, Optional, Union

from rich.panel import Panel
from rich.table import Table
from rich.console import Console, ConsoleOptions, RenderResult
from rich.style import Style
from pydantic import BaseModel, Field, HttpUrl, confloat, conlist

from .history import get_latest_filename, get_oldest_filename


class Base(BaseModel):
    class Config:
        populate_by_name = True


class OpenAIModelPermission(Base):
    allow_create_engine: bool
    allow_fine_tuning: bool
    allow_logprobs: bool
    allow_sampling: bool
    allow_search_indices: bool
    allow_view: bool
    is_blocking: bool
    created: datetime
    group: Optional[str] = None
    id_: str = Field(..., alias="id")
    object_: str = Field(..., alias="object")
    organization: str


class OpenAIModel(Base):
    created: datetime
    id_: str = Field(..., alias="id")
    object_: str = Field(..., alias="object")
    owned_by: str
    parent: Optional[str] = None
    root: str
    permission: list[OpenAIModelPermission]


class OpenAIModelsResponse(Base):
    data: list[OpenAIModel]
    object_: str = Field(..., alias="object")


class OpenAICompletionChoice(Base):
    text: str
    index: int
    logprobs: Optional[list[str]] = None
    finish_reason: str


class OpenAICompletionUsage(Base):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAICompletionResponse(Base):
    created: datetime
    id_: str = Field(..., alias="id")
    object_: str = Field(..., alias="object")
    model: str
    choices: list[OpenAICompletionChoice]
    usage: OpenAICompletionUsage


OpenAIChatRole = Union[Literal["user"], Literal["system"], Literal["assistant"]]


class OpenAIChatCompletionMessage(Base):
    role: OpenAIChatRole
    content: str
    name: Optional[str] = None

    def __str__(self):
        if self.name is not None:
            return "> {} ({}): {}".format(self.role, self.name, self.content)
        return "> {}: {}".format(self.role, self.content)

    def __rich__(self) -> Panel:
        if self.role == "user":
            return Panel(
                self.content,
                title=self.role,
                subtitle="user: {}".format(self.name)
                if self.name is not None
                else self.name,
                border_style=Style(color="light_slate_blue"),
            )
        elif self.role == "assistant":
            return Panel(
                self.content, title=self.role, border_style=Style(color="magenta")
            )
        else:
            return Panel(
                self.content, title=self.role, border_style=Style(color="bright_red")
            )


class OpenAIChatCompletionChoice(Base):
    index: int
    message: OpenAIChatCompletionMessage
    finish_reason: str


class OpenAIChatCompletionResponse(Base):
    created: datetime
    id_: str = Field(..., alias="id")
    object_: str = Field(..., alias="object")
    choices: list[OpenAIChatCompletionChoice]
    usage: OpenAICompletionUsage


class OpenAIChatCompletetionMessageChunk(Base):
    role: Optional[OpenAIChatRole] = None
    content: Optional[str] = None


class OpenAIChatCompletetionMessageChunkChoice(Base):
    finish_reason: Optional[str]
    index: int
    delta: OpenAIChatCompletetionMessageChunk


class OpenAIChatCompletionChunkResponse(Base):
    created: datetime
    id_: str = Field(..., alias="id")
    object_: str = Field(..., alias="object")
    model: str
    choices: list[OpenAIChatCompletetionMessageChunkChoice]


class OpenAIEditResponse(Base):
    created: datetime
    object_: str = Field(..., alias="object")
    choices: list[OpenAIChatCompletionChoice]
    usage: OpenAICompletionUsage


class OpenAIURLImage(Base):
    url: HttpUrl


class OpenAIB64Image(Base):
    b64_json: str


OpenAIImage = Union[OpenAIB64Image, OpenAIURLImage]


class OpenAIImageResponse(Base):
    created: datetime
    data: list[OpenAIImage]


class OpenAIPromptUsage(Base):
    prompt_tokens: int
    total_tokens: int


class OpenAIEmbedding(Base):
    object_: str = Field(..., alias="object")
    embedding: list[float]
    index: int


class OpenAIEmbeddingResponse(Base):
    object_: str = Field(..., alias="object")
    data: list[OpenAIEmbedding]
    model: str
    usage: OpenAIPromptUsage


class OpenAITextResponse(Base):
    text: str


class OpenAIFile(Base):
    created: datetime
    id_: str = Field(..., alias="id")
    object_: str = Field(..., alias="object")
    bytes_: int = Field(..., alias="bytes")
    filename: str
    purpose: str


class OpenAIFileDeletedResponse(Base):
    id_: str = Field(..., alias="id")
    object_: str = Field(..., alias="object")
    deleted: bool


class OpenAIFilesResponse(Base):
    data: list[OpenAIFile]
    object_: str = Field(..., alias="object")


class OpenAIModerationResult(Base):
    categories: dict[str, bool]
    category_scores: dict[str, float]
    flagged: bool


class OpenAIModerationResponse(Base):
    id_: str = Field(..., alias="id")
    model: str
    results: list[OpenAIModerationResult]


class OpenAIEngine(Base):
    id_: str = Field(..., alias="id")
    object_: str = Field(..., alias="object")
    owner: str
    ready: bool


class OpenAIEnginesResponse(Base):
    data: list[OpenAIEngine]
    object_: str = Field(..., alias="object")


OpenAIChatModels = Union[
    Literal["gpt-4"],
    Literal["gpt-4-0314"],
    Literal["gpt-4-32k"],
    Literal["gpt-4-32k-0314"],
    Literal["gpt-3.5-turbo"],
    Literal["gpt-3.5-turbo-0301"],
]


class OpenAIChatModelEnum(str, Enum):
    gpt_4 = "gpt-4"
    gpt_4_0314 = "gpt-4-0314"
    gpt_4_32k = "gpt-4-32k"
    gpt_4_32k_0314 = "gpt-4-32k-0314"
    gpt_35_turbo = "gpt-3.5-turbo"
    gpt_35_turbo_0301 = "gpt-3.5-turbo-0301"


OpenAICompletionModels = Union[
    Literal["text-davinci-003"],
    Literal["text-davinci-002"],
    Literal["text-curie-001"],
    Literal["text-babbage-001"],
    Literal["text-ada-001"],
]

OpenAIEditModels = Union[
    Literal["text-davinci-edit-001"],
    Literal["code-davinci-edit-001"],
]

OpenAITranscriptionModels = Literal["whisper-1"]
OpenAITranslationModels = Literal["whisper-1"]

OpenAIEmbeddingModels = Union[
    Literal["text-embedding-ada-002"],
    Literal["text-search-ada-doc-001"],
]

OpenAIModerationModels = Union[
    Literal["text-moderation-stable"],
    Literal["text-moderation-latest"],
]


class OpenAIChatParams(Base):
    model: OpenAIChatModels
    messages: list[OpenAIChatCompletionMessage]
    temperature: confloat(gt=0, lt=2) = None
    top_p: confloat(gt=0, lt=1) = None
    stop: Optional[Union[str, conlist(item_type=str)]] = None
    max_tokens: Optional[int] = None
    presence_penalty: confloat(gt=-2, lt=2) = None
    frequency_penalty: confloat(gt=-2, lt=2) = None
    logit_bias: Optional[dict[str, confloat(gt=-100, lt=100)]] = None
    user: Optional[str] = None


class OpenAIChat(Base):
    created: datetime = Field(default_factory=datetime.now)
    params: OpenAIChatParams

    @classmethod
    def from_file(
        cls,
        chat_file: Union[str, Literal["latest"], Literal["oldest"]],
        data_dir: Optional[str] = None,
    ):
        if chat_file == "latest":
            if not data_dir:
                raise ValueError(
                    "A data_dir must be specified for reading the 'latest' chat file"
                )
            chat_file = get_latest_filename("chats", ".chat", data_dir=data_dir)
            chat_file = os.path.join(data_dir, chat_file)
        elif chat_file == "oldest":
            if not data_dir:
                raise ValueError(
                    "A data_dir must be specified for reading the 'oldest' chat file"
                )
            chat_file = get_oldest_filename("chats", ".chat", data_dir=data_dir)
            chat_file = os.path.join(data_dir, chat_file)

        if not os.path.exists(chat_file):
            raise FileNotFoundError("OpenAIChat File {} not found.".format(chat_file))
        with open(chat_file, "r", encoding="utf-8") as j:
            data = json.load(j)
        return cls(**data)

    def __str__(self):
        output = ["{}\n".format(str(msg)) for msg in self.params.messages]
        return "\n".join(output)[:-1] if output else ""

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        cmd_opts = Table("Model", self.params.model)
        cmd_opts.add_row("Time of Chat", self.created.strftime("%m/%d/%Y - %H:%M:%S"))
        yield cmd_opts
        for msg in self.params.messages:
            yield msg.__rich__()

    def get_file_contents(self) -> str:
        return self.json(exclude_none=True, by_alias=True)

    def get_filename(self) -> str:
        return str(int(self.created.timestamp() * 1000))

    def get_filepath(self, data_dir: str) -> str:
        fname = self.get_filename()
        chat_dir = os.path.join(data_dir, "chats")
        os.makedirs(chat_dir, exist_ok=True)
        return os.path.join(chat_dir, fname + ".chat")

    def clear(self, data_dir: str):
        os.remove(self.get_filepath(data_dir))

    def persist(
        self, data_dir: str, overwrites: Optional[Union[str, "OpenAIChat"]] = None
    ):
        file_data = self.get_file_contents()
        fpath = self.get_filepath(data_dir)
        with open(fpath, "w", encoding="utf-8") as j:
            j.write(file_data)

        if overwrites:
            if not isinstance(overwrites, str):
                overwrites = overwrites.get_filepath(data_dir)
            if os.path.exists(overwrites):
                os.remove(overwrites)
