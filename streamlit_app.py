"""Evaluator-facing Streamlit presentation. All application work goes through HTTP."""

from html import escape
from urllib.parse import urlsplit

import streamlit as st

from dubizzle_cars.client.api import ApiClient, ClientError


def image_url(value):
    if not value:
        return None
    try:
        parsed = urlsplit(value.strip())
        if parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password:
            return value.strip()
    except ValueError:
        pass
    return None


def render_car(car, position):
    with st.container(border=True):
        heading = " ".join(str(car[k]) for k in ("year", "make", "model") if car.get(k))
        url = image_url(car.get("photo_url"))
        if url:
            st.html(
                f'<img src="{escape(url, quote=True)}" alt="Vehicle photo unavailable" '
                f'loading="lazy" referrerpolicy="no-referrer" '
                f'style="width:100%;height:160px;object-fit:cover;border-radius:8px;">'
            )
        else:
            st.markdown(
                '<div style="width:100%;height:160px;background:#1a1d24;border-radius:8px;'
                'display:flex;align-items:center;justify-content:center;color:#666;font-size:0.85rem;">'
                'No photo available</div>',
                unsafe_allow_html=True,
            )
        st.markdown(f"**{position}. {heading or 'Vehicle listing'}**")
        metadata = car["metadata"]
        price = metadata["sale_price_aed"]
        mileage = metadata["mileage_km"]
        badge_line = []
        if car.get("trim"):
            badge_line.append(car["trim"])
        badge_line.append(f"AED {price}" if price is not None else "Price not specified")
        badge_line.append(f"{mileage:,} km" if mileage is not None else "Mileage not specified")
        st.caption(" | ".join(badge_line))
        with st.expander("Details"):
            st.text(f"Listing ID: {car['listing_id']}")
            st.text(f"Regional specs: {metadata['regional_specs'] or 'Not specified'}")
            warranty = metadata["warranty_mention"]
            st.text(
                f"Warranty mention (source): {warranty}"
                if warranty
                else "Warranty: Not specified in listing"
            )


def render_message(entry):
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        columns = st.columns(3) if entry.get("cars") else []
        for index, car in enumerate(entry.get("cars", [])):
            with columns[index % 3]:
                render_car(car, index + 1)


def start(client, user_id, display_name):
    session = client.create_session(user_id, display_name)
    st.session_state.update(
        user_id=session.user_id,
        display_name=display_name,
        session_id=session.session_id,
        returning_user=session.returning_user,
        messages=[],
        preferences=None,
        error=None,
    )
    try:
        st.session_state.preferences = client.profile(session.user_id).preferences
    except ClientError:
        pass  # Profile display is optional; a successful session is still usable.


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp { background-color: #0e1117; }
        section[data-testid="stSidebar"] {
            background-color: #12151c;
            border-right: 1px solid rgba(255,255,255,0.06);
            min-width: 320px !important;
        }
        div[data-testid="stChatMessage"] { margin-bottom: 0.6rem; }
        div[data-testid="column"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px;
            padding: 0.6rem;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.07) !important;
        }
        h1 { font-size: 1.7rem !important; margin-bottom: 0.1rem !important; }
        h3 { font-size: 1rem !important; margin-bottom: 0.3rem !important; }
        .stCaption, [data-testid="stCaptionContainer"] { color: #9aa0aa !important; }
        div[data-testid="stChatInput"] textarea { border-radius: 18px !important; }
        div[data-testid="stExpander"] {
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        button[kind="secondary"] {
            border-radius: 8px !important;
            text-align: left !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Dubizzle Cars AI Assistant",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    st.title("Dubizzle Cars AI Assistant")
    st.caption("Explore used cars, remember preferences, and book a test drive.")
    st.caption("Bookings are simulated and local. This prototype has no dealership integration.")

    for key, value in {
        "messages": [],
        "session_id": None,
        "user_id": None,
        "returning_user": False,
        "preferences": None,
        "error": None,
    }.items():
        st.session_state.setdefault(key, value)

    try:
        client = ApiClient()
    except ClientError as exc:
        st.error(str(exc))
        return

    with client:
        with st.sidebar:
            st.header("Your conversation")
            with st.form("identity"):
                user = st.text_input("User ID", placeholder="demo-user")
                name = st.text_input("Display name (optional)")
                begin = st.form_submit_button("Start conversation", use_container_width=True)
            if begin:
                if not user.strip():
                    st.warning("Enter a stable user ID to begin.")
                else:
                    try:
                        start(client, user.strip(), name.strip())
                    except ClientError as exc:
                        st.error(str(exc))

            if st.session_state.session_id:
                st.divider()
                st.text(f"User: {st.session_state.user_id}")
                st.caption(f"Session: {st.session_state.session_id}")
                if st.session_state.returning_user:
                    st.info("Welcome back. Your saved car preferences are available in this new conversation.")
                else:
                    st.caption("New user. Preferences will be remembered when supported by your searches.")
                if st.button("New conversation", use_container_width=True):
                    try:
                        start(client, st.session_state.user_id, st.session_state.get("display_name"))
                        st.rerun()
                    except ClientError as exc:
                        st.error(str(exc))
                with st.expander("Remembered preferences"):
                    if st.button("Refresh preferences", use_container_width=True):
                        try:
                            st.session_state.preferences = client.profile(st.session_state.user_id).preferences
                        except ClientError as exc:
                            st.warning(str(exc))
                    if st.session_state.preferences:
                        st.json(st.session_state.preferences)
                    else:
                        st.caption("No preferences loaded. Refresh to check the backend.")

            st.divider()
            if "health" not in st.session_state or st.button("Check backend", use_container_width=True):
                try:
                    st.session_state.health = client.health().inventory_count
                except ClientError:
                    st.session_state.health = None
            if st.session_state.health is None:
                st.warning("Backend unavailable. Start FastAPI before using the assistant.")
            else:
                st.caption(f"Backend: connected. Inventory: {st.session_state.health} listings.")
            st.caption("Health does not check generation availability.")

            with st.expander("Local setup help"):
                st.code("uv run --env-file .env uvicorn main:app --reload", language="bash")
                st.caption("Use a second terminal for Streamlit.")

        suggestions = [
            "Show me Mercedes-Benz cars from 2020 onwards that are luxurious and well equipped.",
            "Tell me more about the first one.",
            "Does it have warranty?",
            "Book the first one this Saturday at 3pm.",
            "I want something comfortable for long drives under AED 100,000.",
            "What was I looking for last time?",
        ]
        clicked_suggestion = None
        with st.expander("Try asking...", expanded=not st.session_state.messages):
            cols = st.columns(2)
            for i, prompt in enumerate(suggestions):
                with cols[i % 2]:
                    if st.button(
                        prompt,
                        key=f"sugg_{i}",
                        use_container_width=True,
                        disabled=not st.session_state.session_id,
                    ):
                        clicked_suggestion = prompt

        for entry in st.session_state.messages:
            render_message(entry)

        if st.session_state.error:
            st.warning(st.session_state.error)
            st.caption("No automatic retry. You may resubmit explicitly; a timed-out action may already have completed.")

        typed_message = st.chat_input(
            "Ask about cars...", disabled=not st.session_state.session_id, max_chars=8000
        )
        message = clicked_suggestion or typed_message

        if not st.session_state.session_id:
            st.info("Enter your user ID and select Start conversation in the sidebar.")

        if message:
            entry = {"role": "user", "content": message, "cars": []}
            st.session_state.messages.append(entry)
            render_message(entry)
            st.session_state.error = None
            try:
                with st.spinner("Checking with the assistant..."):
                    response = client.chat(message, st.session_state.user_id, st.session_state.session_id)
                if response.session_id:
                    st.session_state.session_id = response.session_id
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response.reply,
                        "cars": [car.model_dump() for car in response.cars],
                    }
                )
            except ClientError as exc:
                st.session_state.error = str(exc)
            st.rerun()


if __name__ == "__main__":
    main()