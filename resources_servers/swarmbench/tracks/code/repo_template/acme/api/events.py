def _audit_id(payload):
    return "audit:%s" % payload.get("id")


def _size(payload):
    return len(payload)


def _double_amount(payload):
    return 2 * payload.get("amount", 0)


def _is_large(payload):
    return payload.get("amount", 0) > 100


def _echo_user(payload):
    return payload.get("user", "anon")


def _flag(payload):
    return bool(payload.get("urgent"))


# Handlers in registration order.
HANDLERS = {
    "audit": [_audit_id, _size],
    "billing": [_double_amount, _is_large, _size],
    "notify": [_echo_user, _flag],
    "metrics": [_size],
}


def publish(topic, payload, on_reply):
    """Deliver payload to every handler of topic and call on_reply(reply) once per handler.

    Delivery is in priority order: the LAST registered handler is called first.
    Returns the number of handlers (0 for an unknown topic).

    Replacing legacy emit(topic, payload), which returned the list of replies in REGISTRATION order:
    - Collect replies with a callback: replies = []; publish(topic, payload, replies.append).
    - The collected list is in REVERSE order compared to emit: call replies.reverse() (or use reversed())
      to get emit's list. So emit(...)[0] is replies[-1] and emit(...)[-1] is replies[0].
    - len(emit(...)) is just the return value of publish(topic, payload, lambda reply: None).
    """
    handlers = HANDLERS.get(topic, [])
    for handler in reversed(handlers):
        on_reply(handler(payload))
    return len(handlers)
