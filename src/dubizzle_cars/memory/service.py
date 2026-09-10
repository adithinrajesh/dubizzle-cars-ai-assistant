"""Identity-aware chat orchestration, keeping persistence out of HTTP routes."""

import json
import os
import re

from ..agent import AgentService
from ..agent.models import AgentReply
from ..agent.policy import preflight_reply
from .agent import ContextProvider, SessionTools, context_for
from .repository import StateError


def safe_message(message):
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        message = message.replace(key, "[redacted]")
    return re.sub(
        r"AIza[\w-]+|(?:api[_ -]?key|authorization|bearer)\s*[:=]?\s*\S+",
        "[redacted]",
        message,
        flags=re.I,
    )


class StatefulChat:
    def __init__(self, state, leads, inventory, search, provider, **tool_options):
        self.state, self.leads, self.inventory = state, leads, inventory
        self.search, self.provider, self.tool_options = search, provider, tool_options

    def chat(self, message, user_id, session_id=None):
        if not isinstance(message, str) or not message.strip() or len(message) > 8000:
            raise StateError("Message must contain 1–8000 characters")
        message = safe_message(message.strip())
        with self.state.lock:
            if session_id is None:
                session_id = self.state.create_session(user_id)["session_id"]
            self.state.session(user_id, session_id)
            context = context_for(self.state, user_id, session_id)
            self.state.message(user_id, session_id, "user", message)
            blocked = preflight_reply(message, stateful=True)
            if blocked:
                result = AgentReply(blocked)
            elif re.search(
                r"\b(?:interested in buying|want to buy)\b", message, re.I
            ) and not re.search(r"\d", message):
                result = AgentReply(
                    "What AED budget range are you considering, and what are you looking for in the car?"
                )
            elif re.search(
                r"\b(?:last time|remember|previous preferences|looking for before)\b", message, re.I
            ):
                preferences = context["preferences_untrusted"]
                result = AgentReply(
                    "Your saved automotive preferences: "
                    + json.dumps(preferences, ensure_ascii=True)
                    if preferences
                    else "You have no saved automotive preferences yet."
                )
            else:
                reference, clarification = self._reference(message, context)
                if clarification:
                    result = AgentReply(clarification)
                else:
                    if reference:
                        context["resolved_listing_id"] = reference
                        self.state.update_reference(user_id, session_id, active=reference)
                        context["active_listing_id"] = reference
                    tools = SessionTools(
                        self.inventory,
                        self.search,
                        self.state,
                        self.leads,
                        user_id,
                        session_id,
                        message,
                        context,
                        **self.tool_options,
                    )
                    result = AgentService(
                        ContextProvider(self.provider, context), tools, stateful=True
                    ).chat(message)
                    if (
                        re.search(
                            r"\b(?:book|schedule|reserve|test[- ]drive|viewing)\b", message, re.I
                        )
                        and "book_test_drive" not in result.tool_names
                    ):
                        result = AgentReply(
                            "Please provide the listing and a Dubai-local date/time for a simulated booking. Nothing has been booked."
                        )
                    elif (
                        re.search(r"\b(?:buy|buying|enquiry)\b", message, re.I)
                        and not result.tool_names
                    ):
                        result = AgentReply(
                            "What AED budget range are you considering, and what are you looking for in the car? No enquiry has been saved."
                        )
                    if tools.search_ids is not None:
                        self.state.update_reference(
                            user_id, session_id, results=[car.listing_id for car in result.cars]
                        )
                        if len(result.cars) == 1:
                            self.state.update_reference(
                                user_id, session_id, active=result.cars[0].listing_id
                            )
            result = AgentReply(safe_message(result.reply), result.cars, result.tool_names)
            self.state.message(user_id, session_id, "assistant", result.reply)
            return result, session_id

    def _reference(self, message, context):
        if len(re.findall(r"\b(?:first|second|third|fourth|fifth|last)\b", message, re.I)) > 1:
            return (
                None,
                "Please identify the listing IDs you mean; I cannot select one reference unambiguously.",
            )
        match = re.search(
            r"\b(first|second|third|fourth|fifth|last)\s+(?:one|car|result)\b", message, re.I
        )
        if match:
            word = match[1].lower()
            index = (
                -1
                if word == "last"
                else ["first", "second", "third", "fourth", "fifth"].index(word)
            )
            results = context["last_search_result_ids"]
            if results and index < len(results):
                return results[index], None
            return (
                None,
                "Please specify a listing ID; that result is not available in this session.",
            )
        if re.search(r"\b(?:it|that one|this car)\b", message, re.I):
            active = context["active_listing_id"]
            if active and self.inventory.get_car_details(active):
                return active, None
            return None, "Which listing do you mean? Please provide its ID or result position."
        return None, None
