(responses-api-evolution)=

# Responses API Evolution and Schema Comparison

This engineering note explains why NeMo Gym uses the OpenAI Responses API as its native schema and how it differs from the older Chat Completions API.

## How did OpenAI Responses API evolve and why is it necessary?
Fundamentally, LLMs accept a sequence of tokens on input and produce a sequence of tokens on output. Critically, even if the direct information provided to the LLM is very simple, the outputs can be de-tokenized into a string which can be parsed in a variety of different manners.

In late 2022, OpenAI released text-davinci-003 which used the [Completions API](https://developers.openai.com/api/docs/guides/completions/), accepting a prompt string on input and returning a text string as response.

In March 2023, OpenAI released [GPT-3.5 Turbo](https://developers.openai.com/api/docs/models/gpt-3.5-turbo) along with the Chat Completions API. This API accepted not just a plain prompt string, but rather a sequence of objects representing the conversation input to the model. This API also returned not just a plain text string, but an object representing the model response that contained more directly useful information parsed from the original plain text string. 

In other words, Chat Completions upgraded from Completions to provide a richer user experience in leveraging the response. For example, the Chat Completions response returned a list of "function calls" that were directly usable to select a particular function and call that function with model-provided arguments. This enabled the model to interact not just with the user but with its environment as well.

In March 2025, OpenAI released the [Responses API](https://openai.com/index/new-tools-for-building-agents/) in order to better facilitate building agentic systems. Specifically, the Responses API returned not only a single model response like Chat Completions, but rather a sequence of possibly interleaved reasoning, function calls, function call execution results, and chat responses. So previously, while a single Chat Completion was limited to just a single model generation, the Responses API could generate some model response including a function call, execute that function call on the OpenAI server side, and return both results as part of a single Response to the user.

Responses schema is also a superset of Chat Completions.

Currently, the community has still yet to shift from Chat Completions schema to Responses schema. Part of this issue is that the majority of open-source models are still being trained using Chat Completions format, rather than in Responses format.

Moving forward, Chat Completions will eventually be deprecated, but it will take time for the community to adopt the Responses API. OpenAI has tried to accelerate the effort, for example releasing additional guidance and acceptance criteria for how to implement an [open-source version of Responses API](https://www.openresponses.org/).


## Chat Completions vs Responses API schema.
The primary difference between Chat Completions and Responses API is that the Responses API Response object consists of a sequence of output items, while the Chat Completion only consists of a single model response.

The `output` list for a Response can contain multiple item types, such as:
- `ResponseOutputMessage` - The main user-facing message content returned by the model.
- `ResponseOutputItemReasoning` - Internal reasoning or "thinking" traces that explain the model’s thought process.
- `ResponseFunctionToolCall` - A request from the model to invoke an external function or tool.

**Example**
If a chat completion contains both thinking traces and user-facing text:
```python
ChatCompletion(
    Choices=[
        Choice(
            message=ChatCompletionMessage(
                content="<think>I'm thinking</think>Hi there!",
                tool_calls=[{...}, {...}],
                ...
            )
        )
    ],
    ...
)
```
In the Responses schema, this would be represented as:
```python
Response(
    output=[
        ResponseOutputItemReasoning(
            type="reasoning",
            summary=[
                Summary(
                    type="summary_text",
                    text="I'm thinking",
                )
            ]
        ),
        ResponseOutputMessage(
            role="assistant",
            type="message",
            content=[
                ResponseOutputText(
                    type="output_text",
                    text="Hi there!",
                )
            ]
        ),
        ResponseFunctionToolCall(
            type="function_call",
            ...

        ),
        ResponseFunctionToolCall(
            type="function_call",
            ...

        ),
        ...
    ]
)
```