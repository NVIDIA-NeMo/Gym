Given a problem, your task is to produce a step-by-step solution strategy that a downstream solver model can execute.
Return the plan in XML between <plan> and </plan> tags.

Here is an example of the format you must follow:

<plan>
Step 1: Identify the knowns, unknowns, and any constraints.
Step 2: Select relevant definitions, theorems, or formulas to apply.
Step 3: Set up equations or constructions needed to reach the unknowns.
Step 4: Specify how to manipulate or transform the setup to isolate the targets.
Step 5: State checks for correctness (units, edge cases, invariants, verification method).
</plan>

Rules for the plan:
1. Do not compute, simplify, or give the final answer.
2. Make each step specific, executable, and in logical order.
3. Preserve the original problem's information; do not alter its meaning.
4. Name key variables explicitly and note the formulas to be used.
5. If branching is needed, outline the cases clearly using lines like "Case 1:" and "Case 2:".

Now, provide the plan for this problem:
<problem>
{problem}
</problem>

DO NOT SOLVE THE PROBLEM. ONLY PROVIDE THE PLAN.
Provide exactly 1 plan. Ensure to follow the XML format exactly.