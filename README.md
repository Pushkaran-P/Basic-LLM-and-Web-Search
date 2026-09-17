# Basic-LLM-and-Web-Search
I built a basic LLM chatbot with web search using just ollama function calls, using Qwen 3.5 param models of 0.8B and 2B for the following reasons.

- To test the latency and TTFT of my 6GB GPU PC. 
- Try routing prompts and LLMS just for routing
- Revising and familiarizing some concepts I learnt from my Claude Architect Certification like structured outputs, provenance, returning source documents etc...

## The simple workflow
- Once user enters the prompt, a router LLM decides whether this question requires web search or not, if it does it returns search_required as True and a proper rewritten version of the question asked which will be searched. If not search_required is False and the original prompt is passed as is.
- Searching happens using Ollama's web_search inbuilt function which we can also host as a MCP server if needed.
- Top 3 web results are received sent to the 2B model which then gives the answer to the question along with the source.
- Loop repeats until user quits chatting.

## Next Steps that can be done...
To retrieve relevant content from each website (data cleaning) these steps can be done
- Split content of each returned website into chunks and compare them with the query searched (fixed or recursive character chunking or others). or...
- Semantic Chunking. or...
- Using a LLM to directly return relevent parts
