# Dubizzle Cars AI Assistant

A conversational assistant for exploring the supplied dubizzle used-car inventory.

The application supports natural-language inventory search, multi-turn vehicle references, returning-user preference memory, simulated test-drive bookings, lead qualification, and grounded vehicle recommendations.

Inventory facts always come from the supplied dataset. Missing attributes remain unknown rather than being inferred by the language model.

## Features

- Natural-language inventory search across approximately 100 supplied listings
- Structured filtering by make, model, trim, year, price, mileage, and regional specification
- Semantic ranking using Gemini embeddings
- Multi-turn references such as "the first one" and "does it have warranty?"
- Returning-user preference memory across completely new sessions
- Simulated test-drive booking
- Lead qualification with local CSV persistence
- General automotive conversation
- Guardrails for unrelated requests and competing marketplaces
- Streamlit interface with vehicle cards and listing images

## Architecture

```text
Browser
   |
   v
Streamlit
   |
   | HTTP
   v
FastAPI
   |
   +-------------------------+
   |                         |
   v                         v
Agent Service            SQLite Memory
   |                         |
   |                         +--> Users
   |                         +--> Sessions
   |                         +--> Preferences
   |                         +--> Bookings
   |
   +--> Validated Tools
          |
          +--> Inventory Search
          |       |
          |       +--> Structured Filters
          |       +--> Gemini Embeddings
          |
          +--> Car Details
          +--> Test-Drive Booking
          +--> Lead Qualification
                          |
                          +--> Local CSV
```

The Streamlit client communicates with FastAPI over HTTP and does not access inventory, SQLite, or lead files directly.

The conversational layer is isolated behind a provider interface. OpenAI function calling is used for the live conversational assistant, while Gemini `gemini-embedding-001` embeddings are used for semantic inventory retrieval. Python validates tool calls and remains responsible for inventory facts, session references, booking rules, and persistence.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

Install dependencies:

```bash
uv sync --locked
```

Copy `.env.example` to `.env` and configure:

```env
GEMINI_API_KEY=
OPENAI_API_KEY=
```

`GEMINI_API_KEY` is used for semantic inventory embeddings.

`OPENAI_API_KEY` is used for the conversational assistant.

Do not commit `.env` or real API credentials.

## Run the Backend

In the first terminal:

```bash
uv run --env-file .env uvicorn main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Run the Streamlit Interface

In a second terminal:

```bash
uv run streamlit run streamlit_app.py
```

Streamlit normally opens at:

```text
http://localhost:8501
```

Enter a stable User ID, optionally enter a display name, and select **Start conversation**.

## Implementation Choices

I chose Streamlit because the task is primarily conversational and it provides a lightweight way to build a chat interface, display vehicle cards and images, and demonstrate returning-user behavior without introducing a separate frontend framework. FastAPI remains responsible for the backend API and business logic.

Inventory retrieval uses deterministic structured filtering followed by optional semantic ranking with Gemini embeddings over listing title and description. Structured constraints are always applied first, so semantic similarity cannot override explicit requirements such as make, year, mileage, or budget. SQLite stores users, sessions, preferences, and bookings, while explicit session state stores ordered search results and the active listing so references such as "the first one" and "it" can be resolved reliably.

## Design Decisions

For approximately 100 listings, local cosine ranking with a reusable embedding cache is simpler and easier to inspect than introducing a vector database. Vehicle cards are created only from backend inventory results, not directly from generated model text. Optional metadata such as price, mileage, warranty mentions, and regional specifications is extracted conservatively, and ambiguous values remain unknown.

Short-term memory is implemented explicitly in application state rather than relying only on conversation history. Long-term automotive preferences are stored against a stable user ID in SQLite. Booking availability is also enforced deterministically in Python: Monday to Saturday, 08:00 to 20:00 Dubai time. Qualified leads are written to a local CSV.

## Memory

Within a session, the backend stores the ordered IDs of displayed search results and an active listing.

Example:

```text
User:
Show me Hondas.

User:
What's the mileage on the first one?

User:
Is there a warranty on it?
```

The assistant resolves "the first one" and "it" using explicit session state rather than requiring the user to restate the vehicle.

For long-term memory, each user has a stable `user_id`, while every conversation receives a new `session_id`. Automotive preferences are persisted in SQLite and can be recalled in a completely new session for the same user.

## Booking and Lead Qualification

Test-drive bookings are simulated locally.

Valid viewing times are:

```text
Monday to Saturday
08:00 to 20:00
Asia/Dubai timezone
```

Sunday and out-of-hours requests are rejected by Python validation.

Lead qualification gathers an AED budget and meaningful automotive needs. Qualified leads are appended locally to:

```text
data/leads.csv
```

The assistant does not fabricate contact details.

## Guardrails and Grounding

The assistant can handle greetings, general automotive questions, inventory searches, vehicle comparisons, bookings, and lead qualification.

It politely declines unrelated requests such as programming or history questions and avoids recommending or comparing competing used-car marketplaces.

Inventory-specific facts are grounded in tool results. If a listing does not specify an attribute, the assistant reports that the information is not specified rather than inventing a value.

Listing images are used only for display and are not used for visual inference.

## Testing

Run the offline test suite:

```bash
uv run pytest
```

Lint:

```bash
uv run ruff check .
```

Formatting:

```bash
uv run ruff format --check src tests main.py streamlit_app.py
```

Offline state smoke test:

```bash
uv run python -m dubizzle_cars.smoke_state
```

Automated tests use fake model providers, temporary SQLite databases, temporary CSV files, and mocked HTTP transports. They do not require real API credentials.

## Future Scope

A production version could add authenticated users, secure data-retention controls, hosted relational and vector infrastructure, multi-worker coordination, production observability, and a larger evaluation suite.

The current booking flow is intentionally simulated. Future versions could integrate real dealership availability, calendars, CRM systems, notifications, and lead-management workflows.
