Judge STEERABILITY: does the model correctly understand what the user actually wants, and
engage with that intent directly?

This is about comprehension and engagement, NOT about whether the model succeeded and NOT about
whether its action was the right one. A model can understand the user perfectly, engage head-on,
and still do a bad job or take the wrong action - that is what CORRECTNESS is for. Keep the two
separate.

`yes` - the model read the user's actual intent and engaged with it directly:
- it did (or tried to do) the thing the user asked for
- OR it clearly declined a request it should not satisfy, naming what was asked and why not - a
  clean refusal IS engaging the intent, not dodging it
- OR it asked a focused question that shows it understood the request and needs one thing to
  proceed
The quality of the engagement does not matter here. Bad advice, a botched attempt, a refusal, a
partial answer - all are `yes` as long as the model was genuinely responding to what the user
meant.

`no` - the model did not track the user's intent:
- it misread the request and answered a different question
- it ignored the correction, or repeated verbatim the thing the user just rejected
- it deflected, stalled, or gave a non-answer without engaging the actual ask
- it was confused about what was being asked, or about its own capabilities, and so addressed
  the wrong thing

The test: strip away whether the model was right or successful. Ask only - did it correctly
grasp what the user wanted and respond to THAT? If yes, `yes`, however badly it then executed.
