You are evaluating what an AI agent did at one point in a real conversation with a user.

The agent works in a Linux sandbox rooted at /home/user, shared live with the user. It can
read, write and edit files, run bash, search and fetch the web, generate images and speech,
open a file in the user's viewer, and ask the user a clarifying question. Its work is the files
it leaves behind and what it tells the user.

The conversation so far is between <conversation> and </conversation>. Each block is one of:

- <message from="User"> ... </message>      what the user said
- <message from="Agent"> ... </message>     what the agent said
- <tool_call name="..."> ... </tool_call>   a tool the agent invoked, with its arguments
- <tool_result> ... </tool_result>          what that tool returned

Long tool arguments and results may be shortened; an elision marker says so where it happened.
A block wrapped in <rejected_turn> is an agent turn the user was unhappy with, which the user's
following message is a correction of.$system_prompt_note

$scope_note

The tools available to the agent are between <tools> and </tools>.

$tool_definitions

The criteria for this evaluation are between <criteria> and </criteria>.

<criteria>
$domain_policy
</criteria>

Return JSON matching this schema:

$evaluation_schema
