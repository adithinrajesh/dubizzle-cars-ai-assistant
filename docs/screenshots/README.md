# Live Submission Screenshots

These screenshots were captured from the live FastAPI + Streamlit application and demonstrate the inventory retrieval and memory capabilities required by the take-home assignment.

## Multi-turn inventory conversation

The first four screenshots show a single conversation where the assistant searches the supplied inventory and maintains vehicle context across follow-up questions.

### 1. Inventory search

**File:** `Screenshot 2026-09-10 151629.png`

The user asks:

> Show me Mercedes-Benz cars from 2020 onwards that are luxurious and well equipped.

The assistant retrieves matching cars from the supplied inventory and presents source-grounded listing information and vehicle images.

![Inventory search](<Screenshot 2026-09-10 151629.png>)

### 2. Positional follow-up

**File:** `Screenshot 2026-09-10 151657.png`

The user follows up with:

> Tell me more about the first one.

The assistant resolves "the first one" using the ordered search results stored in the current session and returns details for the same listing.

![Positional follow-up](<Screenshot 2026-09-10 151657.png>)

### 3. Contextual warranty follow-up

**File:** `Screenshot 2026-09-10 151719.png`

The user then asks:

> Does it have warranty?

The assistant keeps the previously selected vehicle in context without requiring the user to restate the listing.

Because the source listing does not specify warranty information, the assistant reports that the information is not specified rather than inventing an answer.

![Warranty follow-up](<Screenshot 2026-09-10 151709.png>)

## Returning-user memory

### 4. Preference persistence across sessions

**File:** `Screenshot 2026-09-10 151833.png`

This screenshot demonstrates long-term memory for a returning user.

A new conversation has been created for the same stable user ID, while previously stored automotive preferences remain available in the sidebar. The remembered state includes structured and semantic preferences such as make, year, budget, and comfort-related requirements.

This demonstrates that user preferences persist independently of the individual conversation session.

![Returning-user memory](<Screenshot 2026-09-10 151833.png>)

---

Together, these screenshots demonstrate:

- natural-language inventory retrieval
- source-grounded vehicle results
- short-term conversational memory
- positional references such as "the first one"
- follow-up references such as "it"
- grounded handling of missing listing information
- stable user identification
- long-term preference persistence across new sessions
