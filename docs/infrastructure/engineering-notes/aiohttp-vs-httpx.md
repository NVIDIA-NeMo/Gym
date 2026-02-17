(aiohttp-vs-httpx)=

# Why use aiohttp backend instead of httpx/httpcore for async HTTP?

TL;DR: httpx is O(n^2) runtime where n is the number of queued requests (that is, for each request, we check all other queued requests). This is terribly inefficient and results in major slowdowns.

On September 17, 2025, inspired by the Deepseek R1 Nature paper, we tried launching a larger rollout batch run with up to 16 off policy steps in NeMo RL. Our setting resulted in NeMo Gym being slammed with 16,000 concurrent requests. At the time, we were using a single NeMo Gym instance with multiple data-parallel vLLM workers, and that setup hung for 40 minutes before the first request was processed. Something was wrong.

Before that time, we had also gotten reports that the rollout collection in NeMo Gym could not be used with high concurrency; that is, in some cases people had to set the concurrency to 32 requests in parallel. Putting these two data points together, we figured something was wrong with the concurrency setup in NeMo Gym.

For some context, NeMo Gym is a set of servers that end up calling a model endpoint server at some point. It is really important that we never artificially restrict the concurrency on the NeMo Gym side since technically we are always clients of that model endpoint server; the model endpoint server could handle many more requests than we are restricting the concurrency to. So we always want NeMo Gym to be as efficient as possible and not have, for example, a max parallel requests or something parameter in NeMo Gym.

Eventually, we isolated the issue to our async HTTP backend—httpx and httpcore. We originally decided to use httpx for the async HTTP backend in NeMo Gym because the OpenAI client uses it by default so we can share the same backend HTTP client. Unfortunately, the httpcore connection pool subroutine for pooling connections over requests is O(n^2) where n is the number of queued requests.

Networking mental model:
1. A request is sent by NeMo Gym to the model endpoint server.
2. This request requires a connection from our client side to the server side.
   1. This connection is a socket (identified by a port) and a socket is an open file (managed by the operating system).
   2. If we are sending 100 requests, in the worst case we could open 100 connections == 100 open files. This quickly becomes very expensive.
   3. So, async HTTP backends will pool requests across connections to a single endpoint, where multiple requests can leverage the same file if they are going to the same endpoint origin.
   4. This is called connection pooling. It is possible that all 100 requests share a single connection.
3. But this connection pooling now needs some management logic. When the client sends a new request, it needs to determine if that request can reuse an existing connection.
   1. And this is where the httpcore connection pool logic is very inefficient.

Here are the key calls in the stack trace:
1. OpenAI client at some point calls httpx client
2. httpx client calls into the [transport](https://github.com/encode/httpx/blob/4b23574cf83307ce27d3b14b4a425dc58c57d28d/httpx/_client.py#L1014)
3. Transport calls into [httpcore connection pool](https://github.com/encode/httpx/blob/4b23574cf83307ce27d3b14b4a425dc58c57d28d/httpx/_transports/default.py#L250)
4. For each request, the httpcore connection pool calls this `_assign_requests_to_connections` [subroutine](https://github.com/encode/httpcore/blob/5974b03c7df89d3ee4e23779900d5349d550753c/httpcore/_async/connection_pool.py#L228)
   1. This subroutine [loops through connections](https://github.com/encode/httpcore/blob/5974b03c7df89d3ee4e23779900d5349d550753c/httpcore/_async/connection_pool.py#L284)
   2. and [loops through queued requests](https://github.com/encode/httpcore/blob/5974b03c7df89d3ee4e23779900d5349d550753c/httpcore/_async/connection_pool.py#L303)
   3. Which results in a total of O(n^2) runtime if the number of queued requests is large. Which is always the case if we slam with some larger number of requests.

In the end, we decided to swap our HTTP backend from httpx to aiohttp since we had good prior experience working with aiohttp in production infra.

Here are some GitHub issues related to this problem. They did not help too much, but they did validate our solution to use aiohttp as an async HTTP backend instead:

- [openai-python issue 1596](https://github.com/openai/openai-python/issues/1596)
- [httpx issue 3215](https://github.com/encode/httpx/issues/3215#issuecomment-2220795088)

If you are using AsyncOpenAI client with a parallelism > 32, you may also want to check if this kind of inefficiency also affects your setup.
