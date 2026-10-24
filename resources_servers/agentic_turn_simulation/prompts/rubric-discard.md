Judge whether this row can be FAIRLY JUDGED at all, or should be thrown away.

This is a gate, not a quality score. It asks one thing: given only what is in the conversation
above, is it possible to fairly assess the action the agent chose this turn? A turn the agent
simply did badly is still judgeable - being wrong is what `correct: no` is for, not a discard.

`discard: yes` - throw the row away, because the trace does not let us judge the agent fairly:

- The user refers to something that is NOT in the conversation, so the right action cannot be
  verified. They ask "why was it rejected?" about a rejection that happened somewhere we cannot
  see, or point to a file, event, or earlier result that never appears in the trace. There is no
  ground truth here to check the agent against.

- The complaint is about the harness or UI, not the agent's action. The user wants a progress
  bar, a live preview, or a rendered view that is controlled by the platform, not by the action
  the agent took; or the complaint is purely about how a generated image LOOKS, produced by a
  blind image tool the agent cannot see the output of. BUT: if the agent also chose actions you
  CAN assess - it retried, explained the limitation, handed off, or tried a real render - keep
  the row and judge those actions. Only discard when there is no separately judgeable action in
  the turn, just the un-actionable complaint.

- The turn is truncated or interrupted, so the intended action is not actually in the trace. The
  last content is a cut-off or malformed tool call, no closing message ends the turn, or it stops
  mid-loop right after a clean tool result. Discard ONLY when the cutoff removes the very action
  being judged. If the agent already took a judgeable action before it was cut off, keep the row
  and use `blocked_by: possibly_interrupted` on the correctness axis instead of discarding.

`discard: no` - keep the row. It is judgeable, even if the agent did a poor job. The agent had
what it needed, took an action we can see in full, and that action can be checked against a
complaint that is genuinely about the agent's work. Most rows are `no`.

Do not discard a row just because the task is hard, the user is rude or vague, the request is
impossible, or the agent failed. Those are all judgeable - keep them.

When you do discard, still fill in the other axes as best you can; `discard` takes precedence and
downstream analysis will drop the row regardless of what the other axes say.
