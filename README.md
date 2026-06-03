# Regradar

AI-powered search and alerts for the U.S. Federal Register. Track new regulations as they're proposed, query them in plain English, and get notified about rules that affect you.

> 🚧 **Status:** Early development. Building in public.

## What it does

The U.S. Federal Register publishes hundreds of new regulatory documents every business day — proposed rules, final rules, and notices that carry real legal and financial weight. Most people and small businesses have no practical way to monitor them. Regradar makes the Federal Register searchable in plain English and alerts you when new regulations match the topics you care about.

## Features (planned)

- 🔍 Natural-language search across Federal Register documents with cited answers
- 📬 Email alerts when new regulations match your tracked topics
- 🏛️ Filter by agency, document type, and comment deadline
- 📝 Plain-English summaries of dense regulatory text

## Tech stack

- **Backend:** Python 3.12, FastAPI, PostgreSQL + pgvector
- **AI:** Claude API (primary), local LLMs via Ollama (fallback), hybrid retrieval with re-ranking
- **Frontend:** Next.js 15, TypeScript, Tailwind CSS
- **Data source:** [Federal Register API](https://www.federalregister.gov/developers/documentation/api/v1)

## Development

This project uses [uv](https://github.com/astral-sh/uv) for Python dependency management.

\`\`\`bash
# Install dependencies
uv sync

# Run linting and formatting
uv run ruff check .
uv run ruff format .

# Run type checking
uv run mypy .

# Run tests
uv run pytest
\`\`\`

## License

MIT