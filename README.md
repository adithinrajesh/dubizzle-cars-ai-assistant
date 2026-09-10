# dubizzle Cars AI Assistant

A conversational used-car assistant built for the dubizzle Cars take-home assessment.

The application lets users search a supplied inventory of approximately 100 cars using natural language, continue multi-turn conversations about specific listings, return in a new session with remembered preferences, book simulated test drives, and save qualified buying enquiries.

Inventory facts always come from the supplied dataset. Missing attributes remain unknown rather than being inferred by the language model.

## Features

- Natural-language inventory search
- Structured filtering by make, model, trim, year, price, mileage, and regional specification
- Semantic ranking over listing title and description
- Multi-turn references such as "the first one" and "does it have warranty?"
- Returning-user preference memory across sessions
- Simulated test-drive bookings
- Lead qualification with local CSV persistence
- General automotive conversation
- Guardrails for unrelated requests and competing marketplaces
- Streamlit chat interface with vehicle cards and listing images
- FastAPI backend with clear separation between UI, agent, retrieval, and persistence
- Offline unit and integration tests

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
   +-----------------------+
   |                       |
   v                       v
Agent Service          SQLite Memory
   |                       |
   |                       +--> Users
   |                       +--> Sessions
   |                       +--> Preferences
   |                       +--> Bookings
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

The conversational and retrieval layers are intentionally separate.

The final live setup uses **OpenAI for conversational tool calling** and **Gemini `gemini-embedding-001` for semantic retrieval**. Both sit behind application abstractions, so model providers do not own business logic or persistence.

Python remains responsible for validating tool calls, inventory facts, session references, booking rules, and lead persistence.

## Project Structure

```text
.
├── main.py
├── streamlit_app.py
├── pyproject.toml
├── uv.lock
├── .env.example
├── data/
│   └── sample_cars_dataset.xlsx
├── src/
│   └── dubizzle_cars/
│       ├── api/
│       ├── agent/
│       ├── client/
│       ├── memory/
│       ├── retrieval/
│       ├── actions.py
│       ├── extraction.py
│       ├── inventory.py
│       ├── loader.py
│       ├── models.py
│       ├── smoke_agent.py
│       ├── smoke_retrieval.py
│       └── smoke_state.py
├── tests/
└── docs/
    └── screenshots/
```

`main.py` is intentionally thin and exposes the FastAPI application required by the assignment. Application logic lives under `src/dubizzle_cars/`.

## Setup

### Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

Install dependencies from the repository root:

```bash
uv sync --locked
```

Create a local `.env` file from `.env.example`.

The application uses:

```env
GEMINI_API_KEY=
OPENAI_API_KEY=
```

- `GEMINI_API_KEY` is used for semantic inventory embeddings.
- `OPENAI_API_KEY` is used for the conversational assistant.

Never commit `.env` or real API credentials.

Optional configuration includes:

```text
DUBIZZLE_CHAT_MODEL
DUBIZZLE_SEMANTIC_ENABLED
DUBIZZLE_WORKBOOK_PATH
DUBIZZLE_EMBEDDING_CACHE_PATH
DUBIZZLE_STATE_DB_PATH
DUBIZZLE_LEADS_CSV_PATH
DUBIZZLE_API_BASE_URL
```

Typical defaults are:

```text
Chat model: gpt-4o-mini
Embedding model: gemini-embedding-001
Workbook: data/sample_cars_dataset.xlsx
Embedding cache: .cache/inventory_embeddings.json
SQLite state: data/dubizzle_assistant.db
Lead CSV: data/leads.csv
API URL: http://127.0.0.1:8000
```

## Run the Application

### 1. Start the FastAPI backend

In the first terminal:

```bash
uv run --env-file .env uvicorn main:app --reload
```

The backend will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### 2. Start the Streamlit client

In a second terminal:

```bash
uv run streamlit run streamlit_app.py
```

Streamlit normally opens at:

```text
http://localhost:8501
```

Enter a stable User ID, optionally provide a display name, and select **Start conversation**.

## API Endpoints

### Health

```http
GET /health
```

Returns backend readiness and loaded inventory count.

### Inventory Details

```http
GET /inventory/{listing_id}
```

Returns one source-backed vehicle listing.

### Inventory Search

```http
POST /inventory/search
```

Supports deterministic filters with optional semantic ranking.

Example:

```json
{
  "filters": {
    "make": "mercedes-benz",
    "year_min": 2020
  },
  "semantic_query": "luxurious and well equipped",
  "limit": 5
}
```

### Chat

```http
POST /chat
```

Example:

```json
{
  "message": "Show me Mercedes-Benz cars from 2020 onwards that are luxurious.",
  "user_id": "demo-user",
  "session_id": "existing-session-id"
}
```

The response contains conversational text, structured vehicle cards, and the session identifier.

### Create Session

```http
POST /sessions
```

Example:

```json
{
  "user_id": "demo-user",
  "display_name": "Demo User"
}
```

A new `session_id` is created every time while preserving long-term memory for the same `user_id`.

### User Profile

```http
GET /users/{user_id}/profile
```

Returns remembered automotive preferences for that user.

## Inventory Retrieval

The supplied workbook contains structured fields such as make, model, trim, year, title, and description. Some attributes such as price, mileage, regional specification, and warranty information appear only inside listing text.

These optional values are extracted conservatively. Ambiguous information remains `null`.

Search uses two stages:

1. **Deterministic structured filtering**
2. **Semantic ranking over the remaining candidates**

For example:

```text
"Mercedes-Benz from 2020 onwards under AED 120,000 and luxurious"
```

can be interpreted as:

```text
make = mercedes-benz
year_min = 2020
price_max = 120000
semantic_query = luxurious
```

The hard constraints are applied first. Semantic ranking can reorder valid candidates, but it cannot introduce a car that violates those constraints.

This is important because a vague preference such as "sporty" should affect relevance ranking without silently becoming an invented constraint such as a specific make, body style, or year.

### Semantic Retrieval

Semantic retrieval uses:

```text
gemini-embedding-001
```

Listing documents are constructed from source-backed structured fields plus title and description.

Document and query vectors use the appropriate retrieval task types. Cosine similarity is calculated locally.

For approximately 100 listings, a dedicated vector database would add unnecessary complexity. Embeddings are instead cached locally and reused when the underlying listing text and provider configuration have not changed.

## Conversational Agent

The conversational layer is isolated behind a provider interface.

For the final live demonstration, OpenAI function calling is used for conversational orchestration while Gemini remains responsible for semantic embeddings.

The model does not directly execute Python functions.

It may request one of the registered tools, after which the application:

1. checks the requested tool against an allowlist
2. validates its arguments
3. executes the corresponding Python service
4. returns the structured result to the model
5. validates the final response

Stateful conversations expose these tools:

```text
search_inventory
get_car_details
book_test_drive
save_lead
```

The agent cannot execute arbitrary Python functions.

## Inventory Grounding

Inventory answers are grounded in tool results rather than model knowledge.

The conversational model can produce user-facing prose and select listing IDs, but structured car cards are built by the backend from actual inventory records.

This prevents the model from creating listings that do not exist in the supplied dataset.

Missing information also remains missing.

For example, if a listing contains no warranty information, the assistant should say:

```text
The listing does not specify warranty information.
```

rather than:

```text
The car has no warranty.
```

The same rule applies to price, mileage, specifications, and features.

Listing images are used only for display. No image understanding or visual attribute inference is performed.

## Short-Term Memory

Each conversation has explicit session state.

After a search, the backend stores the ordered listing IDs that were displayed to the user.

For example:

```text
User:
Show me Hondas.

Assistant:
1. Honda ...
2. Honda ...
3. Honda ...

User:
What's the mileage on the first one?
```

"The first one" is resolved against the stored result ordering rather than relying only on language-model conversation history.

When a user focuses on one listing, its ID becomes the active listing. A follow-up such as:

```text
Does it have warranty?
```

can therefore refer to the same vehicle.

Ambiguous references are clarified rather than guessed.

## Long-Term Memory

Users are identified by a stable `user_id`.

Every new conversation receives a new `session_id`, but selected automotive preferences are persisted in SQLite.

For example:

```text
Session 1:
I'm looking for something comfortable for long drives under AED 100,000.
```

After starting a completely new session with the same User ID:

```text
What was I looking for last time?
```

the assistant can recall the previously saved preference.

Only source-supported automotive preferences are retained. The system does not infer personal or sensitive characteristics.

## Test-Drive Booking

Test-drive bookings are simulated locally.

The booking tool requires:

- a real inventory listing
- a future date and time
- Dubai local time
- Monday through Saturday
- between 08:00 and 20:00 inclusive

Validation is performed in Python using:

```text
Asia/Dubai
```

Examples:

```text
Book the first one Saturday at 3pm.
```

is valid if the date is in the future.

```text
Book it Sunday at 2pm.
```

is rejected.

```text
Book it Saturday at 10pm.
```

is also rejected.

Successful bookings are stored in SQLite. There is no real dealership or calendar integration.

## Lead Qualification

The assistant can qualify prospective buyers by gathering:

- price range
- automotive needs
- optional preferred make
- optional preferred model

A qualified lead is appended to a local CSV file:

```text
data/leads.csv
```

The CSV includes:

```text
user_id
budget_min_aed
budget_max_aed
needs
preferred_make
preferred_model
created_at
```

The assistant does not fabricate phone numbers, email addresses, or other contact details.

Lead persistence is performed by the backend, not Streamlit.

## Intent Recognition and Guardrails

The assistant supports:

- greetings and light chit-chat
- general automotive questions
- inventory searches
- vehicle comparisons
- test-drive bookings
- lead qualification

It politely declines unrelated requests such as:

```text
Write me a Python sorting algorithm.
```

or:

```text
Help me write a history essay.
```

It also avoids recommending or comparing competing used-car marketplaces.

Vehicle-brand comparisons remain allowed.

Inventory descriptions and stored user preferences are treated as data, not executable instructions, so text within a listing cannot override the agent's system rules or tool permissions.

## Demo Walkthrough

A simple end-to-end demonstration is:

### Inventory Search

```text
Show me Mercedes-Benz cars from 2020 onwards that are luxurious and well equipped.
```

### Short-Term Memory

```text
Tell me more about the first one.
```

Then:

```text
Does it have warranty?
```

The assistant should continue referring to the same listing.

### Booking

```text
Book the first one this Saturday at 3pm.
```

Invalid Sunday or out-of-hours requests should be rejected.

### Lead Qualification

```text
I'm interested in buying a car.
```

Then provide:

```text
Up to AED 90,000. Something comfortable for long drives.
```

The assistant can save the qualified enquiry through the backend tool.

### Returning User

In one session:

```text
I want something comfortable for long drives under AED 100,000.
```

Then select **New conversation** while keeping the same User ID.

Ask:

```text
What was I looking for last time?
```

The saved preference should be recalled from SQLite.

## Testing

Run the complete offline test suite:

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

Offline persistence smoke:

```bash
uv run python -m dubizzle_cars.smoke_state
```

The automated tests use temporary SQLite databases, temporary CSV files, fake model providers, and mocked HTTP transports. They do not require real API credentials.

Live model checks are intentionally separate from the automated test suite because they consume provider quota and depend on external service availability.

## Design Choices

Structured constraints are deliberately applied before semantic ranking. This keeps user requirements deterministic while still allowing natural-language preferences such as "comfortable", "sporty", or "luxurious" to improve relevance.

Explicit application state is also used instead of relying solely on the model's transcript memory. Ordered search-result IDs, active listings, persisted preferences, bookings, and leads therefore remain reproducible and testable independently of generated language.

The conversational provider and embedding provider are separate by design. This allowed the application to use Gemini embeddings for retrieval while switching the chat layer when Gemini generation encountered temporary quota and capacity limitations, without changing the retrieval, memory, or business-logic layers.

## Limitations and Future Scope

This is a local prototype rather than a production deployment.

User IDs are caller supplied and are not authenticated. A production version would use authenticated identities, encrypted storage, retention and deletion controls, stronger observability, and idempotent action handling.

The current inventory is small enough for in-memory filtering and local embedding ranking. At larger scale, retrieval could move to hosted relational and vector infrastructure while keeping the same structured-first search design.

Bookings are currently local simulations. A production implementation could integrate real dealership availability, calendars, CRM systems, notifications, and lead-management workflows.

The local embedding cache and CSV coordination are designed for a single application worker. Multi-worker deployment would require shared persistence and coordinated locking.

## Submission Notes

Runtime-generated files are intentionally excluded from version control:

```text
.env
.venv/
.cache/
data/dubizzle_assistant.db
data/dubizzle_assistant.db-*
data/leads.csv
__pycache__/
.pytest_cache/
.ruff_cache/
```

The repository includes the supplied read-only workbook, source code, tests, lockfile, README, and an empty `.env.example`.

No API credentials are included.