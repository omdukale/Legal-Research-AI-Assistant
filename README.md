
# Legal Research AI Assistant

An agentic AI mini-project developed as part of my second-year project seminar course at IIT Bhubaneswar.

The project implements a multi-agent legal research workflow:

**RAG Retrieval → Contextual Answering → Web Cross-Check → Final Output**

- **Retrieval Agent:** Retrieves relevant information from provided documents using RAG.
- **Answering Agent:** Generates answers to the user's query based on the retrieved context.
- **Verification Agent:** Uses the Perplexity Ask MCP server to cross-check the generated answer against web sources for up-to-date information.

**Technologies:** Python, OpenAI Agents SDK, OpenAI Assistants API (File Search), Perplexity MCP Server.

**Documentation:** [OpenAI File Search](https://platform.openai.com/docs/assistants/tools/file-search) | [OpenAI Agents SDK](https://platform.openai.com/docs/guides/agents-sdk)
