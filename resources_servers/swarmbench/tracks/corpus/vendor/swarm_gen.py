# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Swarm aggregation env: per-task corpora of natural-language documents with exact labels.

python3 swarm_gen.py --out data --train 4000 --val 300
Labels come from a per-document state simulation, so gold is exact by construction. Phrase banks live in BANK and can be
extended from banks_extra.json (same keys, extra templates).
"""
import argparse
import base64
import datetime as dt
import gzip
import json
import os
import random
import re
from dataclasses import dataclass, field

FIRST = """Ana Ben Carla Dev Elena Farid Grace Hugo Ines Jamal Kenji Lena Marco Nadia Omar Priya Quinn Rosa Sami Tara
Uma Victor Wen Ximena Yusuf Zoe Aaron Bianca Chidi Dana Emil Fatima Gabe Hana Ivan Jo Kofi Lucia Mateo Noor Olga Pavel
Rhea Sven Tomas Una Vera Wade Yara Zane Asha Bruno Celine Diego Esme Felix Gita Hal Ilse Jonah Kira Liam Mina Nils""".split()
LAST = """Abara Becker Castillo Dubois Eriksen Fontaine Garcia Haddad Ito Jensen Kowalski Lindqvist Moreau Nakamura
Okafor Petrov Quist Rossi Sato Tanaka Ulloa Varga Weber Xu Yilmaz Zielinski Adeyemi Brandt Chen Duarte Engel Ferreira
Gupta Holm Ibarra Jovanovic Kim Lopez Mendes Novak Ortiz Park Reyes Silva Torres Ueda Vance Walsh Young Zhou""".split()
PRODUCTS = """wireless earbuds|standing desk|espresso machine|hiking backpack|robot vacuum|4K monitor|air purifier|
electric kettle|mechanical keyboard|smart thermostat|office chair|blender|rain jacket|e-reader|webcam|
bluetooth speaker|cast iron skillet|yoga mat set|baby monitor|cordless drill|duvet cover|graphics tablet|
fitness tracker|portable charger|desk lamp|winter boots|carry-on suitcase|gaming headset|router|toaster oven""".replace("\n", "").split("|")
AGENTS = ["Maya (Support)", "Tom (Support)", "Iris (Support)", "Raj (Support)", "Bea (Support)", "Ollie (Support)"]
SUPERVISORS = ["Helen (Team Lead)", "Marcus (Supervisor)", "Dee (Escalations)", "Sandra (Team Lead)"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August"]
NUM_WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one twenty-two twenty-three twenty-four".split()


def pick(rng, xs):
    return xs[rng.randrange(len(xs))]


def person(rng):
    return f"{pick(rng, FIRST)} {pick(rng, LAST)}"


def money(rng, x):
    s = f"{x:,.2f}"
    return pick(rng, [f"${s}", f"${s}", f"${s}", f"{s} USD", f"USD {s}", f"${s}".replace(".00", "")])


def date_fmt(rng):
    return pick(rng, ["%b %d", "%Y-%m-%d", "%d %B", "%B %d", "%m/%d", "%a %b %d"])


def fill(rng, template, slots):
    # {a|b|c} picks one alternative, then {name} slots are filled.
    def alt(m):
        return pick(rng, m.group(1).split("|"))

    prev = None
    while prev != template:
        prev = template
        template = re.sub(r"\{([^{}]*\|[^{}]*)\}", alt, template)
    return template.format(**slots)


@dataclass
class Doc:
    id: str
    text: str
    label: bool
    trace: list = field(default_factory=list)


class Clock:
    def __init__(self, rng, start):
        self.rng, self.t = rng, start

    def step(self, lo=0.2, hi=30.0):
        self.t += dt.timedelta(hours=self.rng.uniform(lo, hi))
        return self.t


BANK = {
    # ---- expense lines and notes ----
    "ex.hotel_total": ["Hotel: {nights} nights at the {hotel}, {total} for the whole stay."],
    "ex.hotel_rate": ["Hotel: {hotel}, {rate} per night for {nights} nights."],
    "ex.approval": ["Finance approved the room rate in writing on {date} (email attached).",
                    "Written approval for the hotel from Finance is attached, dated {date}."],
    "ex.no_approval": ["I emailed Finance about the rate before booking but never heard back, so I went ahead.",
                       "My manager said the rate was fine verbally. I didn't get anything from Finance."],
    "ex.dinner": ["Team dinner on {date}: the bill was {bill}, of which {drinks} was {kind}. Claimed: {claimed}.",
                  "Client dinner, {date}. Total {bill} including {kind} ({drinks}); amount claimed {claimed}."],
    "ex.dinner_note": ["I took the {kind} off the dinner line per the policy.", "Adjusted the dinner claim for the drinks.",
                       "{mgr} asked me to check the dinner line for the {kind}."],
    "ex.receipt.missing": ["I couldn't find the receipt for the {miss}."],
    "ex.receipt.wrong_item": ["I couldn't find the receipt for the {miss}. Update: found it and attached the receipt for the {other}."],
    "ex.receipt.statement": ["No itemized receipt for the {miss}, but I've attached my card statement showing the charge."],
    "ex.receipt.promised": ["Receipt for the {miss} to follow, I'll dig it out of my email."],
    "ex.receipt.found": ["I couldn't find the receipt for the {miss} at first. Update: found it and attached the itemized receipt for the {miss} on {date}."],
    "ex.receipt.found_both": ["Itemized receipts attached for the {miss} and the {other}."],
    "ex.receipt.statement_plus": ["For the {miss} I've attached my card statement showing the charge, and the itemized receipt as well."],
    "ex.receipt.promised_done": ["Receipt for the {miss} to follow, I'll dig it out of my email. Update: itemized receipt for the {miss} now attached."],
    "ex.cabin.gate": ["economy (upgraded to business at the gate; the upgrade fee is included in this line)"],
    "ex.cabin.miles": ["economy (I upgraded to business with my own miles; only the economy fare is claimed)"],
    "ex.air_times": ["Airfare, {cabin}: {fare}. Outbound leg departed {dep} and landed {arr} local time, same time zone; the return was the same flight in reverse."],
    "ex.air_dur": ["Airfare, {cabin}: {fare}. Each leg was {h} h {m} min."],
    # ---- incident summary statements ----
    "in.start_abs": ["The first errors appeared at about {t}.", "Errors began around {t}.", "Customers started seeing failures at roughly {t}."],
    "in.start_rel": ["The first errors showed up roughly {rel} minutes before the alert fired."],
    "in.first": ["The root cause was in {first}: it failed first, and {second} errored only because it depends on it.",
                 "{first} was the first component to fail; the {second} errors were a downstream effect.",
                 "Our investigation shows {first} broke first and {second} followed."],
    "in.peak": ["At the worst point {k} of the {total} hosts were unhealthy.", "Impact peaked at {k} unhealthy hosts out of {total}.",
                "{k} hosts (out of {total}) were out of rotation at peak."],
    "in.rb.done": ["We rolled {svc} back from {ver} to {prev}, which stopped the bleeding.", "Mitigation was a rollback of {svc} to {prev}, plus a config change."],
    "in.rb.aborted": ["A rollback was started but we cancelled it once the config change looked effective, so no rollback actually happened.",
                      "We began rolling back {svc}, then called it off. {ver} stayed live throughout."],
    "in.rb.none": ["We considered rolling back {svc} but decided against it; a config change was enough.",
                   "No rollback was needed. Mitigation was a configuration change in the ops console."],
    "in.res_abs": ["Service was fully restored at {t}.", "Everything was healthy again by {t}."],
    "in.res_dur": ["The incident lasted about {mins} minutes from the first errors to resolution."],
    # ---- refund events (this-order or other-order, same text) ----
    "rf.ok": [
        "I've processed a refund of {amt} back to {card}. {Confirmation|Reference} number {ref}. Our payment processor shows it as settled.",
        "Good news: finance pushed the refund for {order} through this morning. {amt} has been returned to {card} (ref {ref}) and the transaction status is COMPLETED.",
        "Refund done. {amt} went back to {card} today, ref {ref}. Depending on the bank it can take a few days to show, but it has left our side.",
        "I went ahead and refunded the full {amt} for the {product}. The processor confirmed the credit to {card} under reference {ref}.",
        "The refund of {amt} cleared. I can see it marked as paid out to {card} in the payments console (ref {ref}).",
        "The {amt} for {order} is back on {card} as of today, ref {ref}. That's final on our side.",
    ],
    "rf.ok_partial": [
        "We can't refund the shipping, but I've refunded the item price of {amt} to {card} (ref {ref}). That has gone through on our end.",
        "Processed a partial refund of {amt} to {card} for the {product} (ref {ref}); the processor marked it completed.",
    ],
    "rf.confirm_c": [
        "Just wanted to say the {amt} showed up on my statement today as a credit from you. Thanks for sorting it.",
        "Got it, the money for the {product} landed in my account. Appreciate the help.",
    ],
    "rf.confirm_a": [
        "Customer confirmed by phone that the {amt} is back in her account. Nothing else needed.",
    ],
    "rf.fail": [
        "I tried to refund {amt} to {card}, but it was rejected by the issuing bank because that account has been closed. The money came back to us.",
        "The refund attempt for {order} bounced. Our processor returned the {amt} with reason code 'card expired', so nothing reached the customer.",
        "Payments flagged the refund for the {product} as failed (status RETURNED). The card on file was {replaced after a fraud alert|cancelled}, so the {amt} never arrived.",
        "I tried to send the {amt} back but the system wouldn't let me refund an order this old to the original card. It errored out and nothing was sent.",
    ],
    "rf.clawback": [
        "Heads up for whoever picks this up: the refund we sent for {order} was reversed by finance, because the return package never made it back to the warehouse. The customer has been recharged {amt}.",
        "Finance note: the {amt} refunded earlier for the {product} has been charged back to {card}. The return window closed without the item being received, so the refund was withdrawn.",
        "Audit caught that the refund for {order} duplicated a prior adjustment, so it was reversed. The balance is back to where it was before the refund.",
    ],
    "rf.deny": [
        "Unfortunately we're unable to offer a refund on {order}, since it's outside our {30|45|60}-day return window.",
        "I'm sorry, but I can't approve a refund for the {product}. It shows signs of use beyond normal testing, which our policy excludes.",
        "I've checked with the returns team and the refund request for {order} was declined: opened items in this category aren't refundable.",
        "After review, we won't be able to refund {order}. The carrier's photo shows the package delivered to the correct address.",
    ],
    "rf.escalate": [
        "Customer asked to speak to a manager. Escalating to {sup}.",
        "Escalated per customer request. {sup}, can you take a look?",
    ],
    "rf.override_ok": [
        "{sup} here. I've reviewed the history and I'm making an exception. Refund of {amt} processed to {card}, ref {ref}. It shows as completed.",
        "Supervisor review: overriding the earlier decision on {order}. I processed the {amt} refund to {card} myself just now (ref {ref}), and it has settled.",
    ],
    "rf.override_deny": [
        "{sup} here. I've reviewed it and I'm afraid the earlier decision stands. We can't refund {order}.",
        "Supervisor review: I agree with the original agent. No refund is possible on the {product}, sorry.",
    ],
    "rf.credit": [
        "Instead of a refund, I've added {amt} in store credit to the account for {order}. It never expires and can be used on any order.",
        "As agreed, we've issued a gift card for {amt} in lieu of a refund on the {product}. The code was emailed to the address on file.",
        "I've applied {amt} as account credit for {order}, which the customer accepted rather than waiting on a card refund.",
    ],
    "rf.credit_swap": [
        "Customer changed her mind about the store credit. I've cancelled the {amt} credit and refunded the same amount to {card} instead (ref {ref}); the refund has settled.",
        "Per the customer's request, the gift card for {order} was voided and the {amt} was refunded to {card} (ref {ref}). Processor shows it completed.",
    ],
    "rf.replace": [
        "Rather than refunding, we've shipped a replacement {product} (tracking {trk}). The customer said that works for her.",
        "Replacement unit for {order} sent out today, tracking {trk}. Customer agreed to the replacement instead of a refund.",
    ],
    "rf.pending": [
        "The refund of {amt} for {order} is approved and will be processed once the returned {product} is scanned in at the warehouse. Return label sent.",
        "I've queued the refund for {order} with finance. It will be released after their weekly review, and I'll update this ticket when it actually goes out.",
        "Refund approved in principle for the {product}. It still needs the return to arrive before anything is sent back to {card}.",
    ],
    "rf.withdraw_c": [
        "Please cancel my refund request for {order}, I've decided to keep the {product}.",
        "Never mind about the refund, the {product} {turned up after all|started working after a firmware update}. I'll keep it.",
    ],
    "rf.withdraw_a": [
        "Customer wrote back that the {product} {turned up after all|started working after a firmware update} and she no longer wants a refund for {order}.",
    ],
    "this_lead": [
        "Update on order {order}: ", "Following up on the {product}: ", "Re this ticket ({order}): ", "Hi {cf}, about the {product} order: ",
        "Status update for {order} (the {product}): ",
    ],
    "other_lead_c": ["About my other order {order} (the {product}): ", "Also, on the earlier order {order}: "],
    "other_lead": [
        "Separately, about your other order {order} (the {product}): ",
        "Unrelated to this ticket, but since you asked about order {order}: ",
        "Quick update on the {product} order ({order}) from your other ticket: ",
        "FYI, on the earlier order {order}, the one with the {product}: ",
    ],
    "retract": [
        "Please disregard my last update. It was meant for a different customer's ticket and does not apply to your order.",
        "Correction: ignore my previous message, I pasted the wrong ticket's notes. Nothing has changed on this order yet.",
        "Sorry, that last note was posted to the wrong ticket. Please ignore it; this order's status is unchanged.",
    ],
    "rf.decoy_c": [
        "Can I just get a refund? I don't want a replacement, I want my money back.",
        "Last time something like this happened I was promised a refund and it took two months to show up.",
        "Would you refund the shipping fee as well?",
        "My sister got a full refund from you last month for something similar, so I assume this is the same process.",
        "If it's store credit I won't use it, so please don't send that.",
    ],
    "rf.decoy_a": [
        "For reference, refunds go back to the original payment method and usually take 5-10 business days to appear after they're processed.",
        "If you'd prefer, we could also offer store credit instead of a refund. Just let me know.",
        "If the return doesn't arrive within 30 days, any refund would be reversed. Standard wording, we send it to everyone.",
        "Internal note: reminder for the team, don't promise refunds before the item is inspected.",
        "Internal note: if this ends up as store credit, use the goodwill code, not the standard one.",
        "A replacement is also an option if you'd rather have the {product} than the money.",
    ],
    # ---- billing ----
    "bl.issue.DUP": ["I was charged twice for my {month} invoice, {amt} each time, on the same day. Please fix this."],
    "bl.issue.TRIPLE": ["My card was charged three times for the {month} subscription, {amt} each. I only have one account."],
    "bl.issue.PLAN": ["I'm on the Basic plan ({lo} a month) but you billed me {hi} for {month}, which is the Pro price."],
    "bl.issue.HOLD": ["There's a second charge of {amt} from you on my card for {month}. Did you bill me twice?"],
    "bl.issue.ADDON": ["I see two charges for {month}, {amt} and then another {amt2}. What is the second one for?"],
    "bl.again": [
        "And now the same thing has happened with {month_next}: two charges of {amt} again. I can't believe this.",
        "Update from me: {month_next} was also billed twice, {amt} both times.",
    ],
    "bl.rev_amount": [
        "I've refunded {x} to your card (ref {ref}). The reversal has settled.",
        "Processed a refund of {x} for the {month} billing problem, ref {ref}. It has gone through.",
        "{x} has been returned to your card for {month} (ref {ref}); the processor shows it completed.",
    ],
    "bl.rev_one": [
        "I've reversed one of the {amt} charges for {month}. The processor shows that reversal as completed.",
        "One duplicate {amt} charge for {month} has been refunded to your card (ref {ref}).",
    ],
    "bl.rev_all": [
        "I've refunded every extra charge for {month}, so you've now paid for it exactly once (ref {ref}).",
        "All the duplicate {month} charges were reversed today; only the original {amt} remains.",
    ],
    "bl.fail": [
        "The reversal I submitted for {month} failed. The bank rejected it because the card was reissued, so the money is still with us.",
        "Update: the refund for the {month} charge bounced (status RETURNED). Nothing has gone back to the customer yet.",
    ],
    "bl.credit_next": [
        "Rather than sending it back to the card, I've applied the extra for {month} as a credit on your next invoice.",
        "The {month} overpayment has been converted to account credit, which will be used against future bills.",
    ],
    "bl.credit_to_card": [
        "As requested, I've cancelled the account credit and refunded the full overpayment for {month} to your card instead (ref {ref}).",
    ],
    "bl.explain.HOLD": [
        "I checked with payments: the second {amt} was only a temporary authorization hold from when you updated your card. It was never captured and has dropped off.",
        "That second entry is a pending authorization, not a charge. It expired on its own and you were only actually billed once for {month}.",
    ],
    "bl.explain.ADDON": [
        "On review, the second charge of {amt2} is the storage add-on enabled on your account this month, so it's a legitimate charge, not a duplicate.",
        "The {amt2} charge is for the extra seat added to your workspace in {month}. It's correct as billed.",
    ],
    "bl.cust_ok": ["Ah, that makes sense, thanks for explaining.", "Ok, thanks for sorting that out."],
    "bl.other_lead_c": ["Different account, but while I have you, on my other workspace: ", "Separately, on my old {month} invoice: "],
    "bl.this_lead": ["Update on this case: ", "Re your {month} charges: ", "Following up on the {month} invoice: ", "Status for this case: "],
    "bl.other_lead": [
        "Separately, about the {month} invoice you mentioned last time: ",
        "On the older {month} issue (different case, already closed): ",
        "For your other workspace's {month} bill: ",
    ],
    "bl.decoy": [
        "Duplicate charges are usually reversed within 3-5 business days once we've confirmed them.",
        "If this turns out to be a temporary authorization hold, it will drop off on its own.",
        "We can also put any overpayment toward your next invoice if you'd like.",
        "I'll need to escalate if the reversal doesn't go through, so please let me know if you don't see it.",
        "Reminder: Basic is {lo}/month, Pro is {hi}/month, both billed on the 1st.",
    ],
}

def T(rng, key, slots):
    return fill(rng, pick(rng, BANK[key]), slots)


CUST_DETAIL = [
    "For context, I ordered it on a weekday and it was supposed to arrive within three business days.",
    "I work from home so I was here all day when the delivery was supposedly made.",
    "I've attached the order confirmation email in case that helps.",
    "I read through your help center articles first but none of them covered this situation.",
    "I also tried calling the number on the website but the wait time was over an hour.",
    "To be honest I don't have a lot of time to keep chasing this, so I'd appreciate a quick answer.",
    "My account email is the one I'm writing from, and the order was under my name.",
    "I'm happy to send more photos or anything else you need, just tell me exactly what.",
    "This is the third email I've sent about this, the first two got automatic replies only.",
    "I bought it specifically because the reviews said your support was good, so I'm hoping that's true.",
    "Please reply by email rather than phone, I'm in meetings most of the day.",
    "I checked my banking app this morning before writing to you.",
]
AGENT_EXPLAIN = [
    "Just to set expectations, our returns policy covers most items for 30 days from delivery, and some categories have their own rules.",
    "I've added a note to the account so anyone else who picks this up has the full history.",
    "Our system sometimes takes a little while to sync with the payment processor, so status pages can lag by a day.",
    "I've checked the carrier scan history and the warehouse notes.",
    "I appreciate you sending all of that over, it makes things a lot faster on our side.",
    "I'm going to keep this ticket assigned to me so you don't have to explain it again to someone else.",
    "Thanks for bearing with me while I looked into this with the other teams.",
]
SIGNATURES = [
    "\n--\n{name}\nCustomer Care | Mon-Fri 8am-6pm ET\nPlease keep the ticket number in the subject line when replying.",
    "\nBest,\n{name}\nSupport Team\nThis email and any attachments are confidential and intended only for the addressee.",
    "\nKind regards,\n{name}\nHelp Center: help.example.com | Order status: example.com/orders",
    "\nThanks,\n{name}",
]
COLOR = [
    "I've been ordering from you for {three|five|six} years and this is the first time I've had to complain.",
    "Honestly I'm pretty frustrated, this is the second order in a row with a problem.",
    "I need this sorted out before I travel next week.",
    "I already spent {twenty minutes|half an hour} in the chat queue yesterday and got disconnected.",
    "My partner says I should just dispute it with the bank but I'd rather sort it out with you directly.",
    "",
]


BILL_DETAIL = [
    "I run a small team and every dollar matters right now, so I'd like this sorted quickly.",
    "I've attached a screenshot of my card statement.",
    "Your billing page only shows one invoice for the month, which is why this is confusing.",
    "I haven't changed anything on the account in months.",
    "Please reply by email rather than phone, I'm in meetings most of the day.",
    "I checked my banking app this morning before writing to you.",
    "Our accountant flagged this when reconciling the books.",
]


def pad_entries(rng, entries, cust_name, detail=None):
    # Signatures, quoted replies and extra detail; none of it carries outcome information.
    out, last_agent = [], None
    for when, who, body in entries:
        if who.endswith("(Customer)"):
            body = " ".join([body] + rng.sample(detail or CUST_DETAIL, rng.choice([0, 0, 1, 2])))
            if last_agent and rng.random() < 0.3:
                q = last_agent.split("\n")[0]
                body += "\n\n> " + (q if len(q) < 220 else q[:220] + "...")
            body += pick(rng, ["", "", f"\n\n{cust_name.split()[0]}", f"\n\nThanks,\n{cust_name}", "\n\nSent from my phone"])
        elif "(" in who and not body.startswith("Internal note"):
            if rng.random() < 0.35:
                body = pick(rng, AGENT_EXPLAIN) + " " + body
            if rng.random() < 0.7:
                body += pick(rng, SIGNATURES).format(name=who.split(" (")[0])
            last_agent = body
        out.append((when, who, body))
    return out


def ticket_text(header, entries, fmt):
    lines = list(header) + [""]
    for when, who, body in entries:
        lines += [f"[{when.strftime(fmt)} {when.strftime('%H:%M')}] {who}:", body.strip(), ""]
    return "\n".join(lines).strip() + "\n"


def order_block(rng, product, amt, order):
    city = pick(rng, ["Portland, OR", "Madison, WI", "Raleigh, NC", "Tucson, AZ", "Albany, NY", "Boise, ID", "Dayton, OH"])
    lines = [f"Order {order} | placed via {pick(rng, ['web', 'iOS app', 'Android app', 'phone order'])} | ship to {city}",
             f"  1 x {product}  {money(rng, amt)}"]
    for _ in range(rng.randint(0, 2)):
        lines.append(f"  1 x {pick(rng, ['gift wrap', 'USB-C cable', 'cleaning kit', 'carrying case'])}  {money(rng, round(rng.uniform(4, 25), 2))}")
    lines.append(f"  Shipping: {pick(rng, ['standard', 'expedited', 'free standard'])}")
    return lines


# ---------------------------------------------------------------------------
# Family 1: refunds. Positive iff the ticket's own order ends refunded (not clawed back).

RF_MOVES = {
    "none": ["ok", "ok_partial", "deny", "fail", "credit", "replace", "pending", "withdraw"],
    "denied": ["esc_ok", "esc_deny", "credit", "replace"],
    "failed": ["ok", "credit", "withdraw"],
    "pending": ["ok", "deny", "withdraw", "fail"],
    "ref": ["clawback", "confirm"],
    "credit": ["credit_swap"],
    "clawed": ["ok"],
    "replaced": [],
    "withdrawn": [],
}
RF_NEXT = {"ok": "ref", "ok_partial": "ref", "deny": "denied", "fail": "failed", "credit": "credit", "replace": "replaced",
           "pending": "pending", "withdraw": "withdrawn", "esc_ok": "ref", "esc_deny": "denied", "clawback": "clawed",
           "confirm": "ref", "credit_swap": "ref"}
RF_POS_TEXT = {"ok", "ok_partial", "esc_ok", "confirm", "credit_swap"}
RF_AGENT_EVENTS = {"ok", "ok_partial", "deny", "fail", "credit", "replace", "pending", "clawback", "credit_swap"}


def rf_walk(rng, label):
    while True:
        state, path = "none", []
        for _ in range(rng.randint(1, 4)):
            moves = RF_MOVES[state]
            if not moves:
                break
            ev = pick(rng, moves)
            retract = ev in RF_AGENT_EVENTS and rng.random() < 0.15
            path.append((ev, retract))
            if not retract:
                state = RF_NEXT[ev]
        if (state == "ref") == label:
            return path, state


def rf_text(rng, ev, slots, cust, agent, sup):
    # Returns a list of (who, text) entries for one event.
    agent = pick(rng, [agent, agent, agent, "Finance Ops", "Billing System"]) if ev in RF_AGENT_EVENTS else agent
    if ev in ("esc_ok", "esc_deny"):
        return [(agent, T(rng, "rf.escalate", slots)), (sup, T(rng, "rf.override_ok" if ev == "esc_ok" else "rf.override_deny", slots))]
    if ev == "confirm":
        return [(cust, T(rng, "rf.confirm_c", slots))] if rng.random() < 0.7 else [(agent, T(rng, "rf.confirm_a", slots))]
    if ev == "withdraw":
        return [(cust, T(rng, "rf.withdraw_c", slots))] if rng.random() < 0.6 else [(agent, T(rng, "rf.withdraw_a", slots))]
    if ev == "clawback":
        return [(pick(rng, ["Finance Ops", "Billing System", agent]), T(rng, "rf.clawback", slots))]
    text = T(rng, "rf." + ev, slots)
    if "{ref}" not in "".join(BANK["rf." + ev]) or rng.random() < 0.3:
        text += pick(rng, [" (case ref {ref})", " Ref {ref}.", " Logged under {ref}."]).format(ref=slots["ref"])
    if rng.random() < 0.6:
        text = T(rng, "this_lead", slots) + text
    return [(agent, text)]


def gen_refund(rng, doc_id, label, base_date):
    cust_name = person(rng)
    cust = f"{cust_name} (Customer)"
    agent, sup = pick(rng, AGENTS), pick(rng, SUPERVISORS)
    amt = round(rng.uniform(19, 480), 2)
    last4 = f"{rng.randrange(10000):04d}"
    clock = Clock(rng, base_date + dt.timedelta(days=rng.randrange(60), hours=rng.uniform(7, 18)))
    fmt = date_fmt(rng)
    card = pick(rng, [f"the Visa ending in {last4}", f"the Mastercard ending {last4}", "the original card", "her PayPal account",
                      f"the Amex ending in {last4}", "the card on file"])
    product = pick(rng, PRODUCTS)

    def mk_slots(order, prod, a):
        return dict(order=order, product=prod, amt=money(rng, a), card=card, ref=f"RF{rng.randrange(10**7, 10**8)}",
                    trk=f"1Z{rng.randrange(10**9, 10**10)}", sup=sup.split(" (")[0], cf=cust_name.split()[0], last4=last4)

    slots = mk_slots(f"#{rng.randrange(100000, 999999)}", product, amt)
    other = mk_slots(f"#{rng.randrange(100000, 999999)}", pick(rng, [p for p in PRODUCTS if p != product]), round(rng.uniform(19, 480), 2))
    path, state = rf_walk(rng, label)
    # Every doc carries at least one refund-success text and one non-refund text, so presence of a phrase says nothing.
    kinds = {ev in RF_POS_TEXT for ev, _ in path}
    extra = []
    for need in (True, False):
        if need not in kinds or rng.random() < 0.3:
            pool = [e for e in RF_NEXT if (e in RF_POS_TEXT) == need and e not in ("esc_ok", "esc_deny")]
            extra.append(pick(rng, pool))
    entries = []
    complaint = pick(rng, [
        "The {product} from order {order} arrived with a {cracked|shattered|badly dented} {casing|panel|housing} and I can't use it.",
        "I was sent the wrong {size|color|model} of {product} (order {order}). The packing slip even says the right one.",
        "My {product} stopped working after {two|three|five} days. It won't power on at all now, even after charging overnight.",
        "Tracking for order {order} says delivered but nothing ever came. My neighbours haven't seen it either.",
        "I cancelled order {order} about {ten minutes|an hour} after placing it, got a cancellation email, and was still charged {amt}.",
        "The {product} looks nothing like the photos on your site. The material feels cheap and the finish is already peeling.",
    ])
    entries.append((clock.t, cust, " ".join([fill(rng, complaint, slots)] + [fill(rng, c, {}) for c in rng.sample(COLOR, 2) if c])))
    exchanges = [
        ("Thanks for reaching out, {cf}. I'm sorry about the trouble with the {product}. Could you send a photo of the {item|unit|packaging} and the packing slip so I can log it?",
         "Photos attached. {The outer box was a bit crushed too.|Let me know if you need anything else.}"),
        ("Hi {cf}, thanks for your patience. Can you confirm the email address on the account and the last four digits of the card you paid with?",
         "Same email as this one. Card ends in {last4}."),
        ("Hello {cf}, I've pulled up order {order}. Before anything else, have you tried {a full reset|unplugging it for a minute}? Sometimes that clears it.",
         "Yes, tried that {twice|already}. No change."),
        ("Thanks {cf}. I can see the order was shipped from our {Reno|Columbus|Atlanta} warehouse. Give me a moment to check the options on this one.",
         "{Ok, thanks.|Sure.|Fine, I'll wait.}"),
    ]
    for a, c in rng.sample(exchanges, rng.randint(1, 3)):
        entries.append((clock.step(0.5, 20), agent, fill(rng, a, slots)))
        entries.append((clock.step(0.3, 30), cust, fill(rng, c, slots)))
    script = [("this", ev, r) for ev, r in path] + [("other", ev, False) for ev in extra]
    # Other-order events go anywhere; this-order events keep their relative order.
    for item in [s for s in script if s[0] == "other"]:
        script.remove(item)
        script.insert(rng.randrange(len(script) + 1), item)
    decoys = [("C", d) for d in rng.sample(BANK["rf.decoy_c"], 2)] + [("A", d) for d in rng.sample(BANK["rf.decoy_a"], 2)]
    rng.shuffle(decoys)
    decoys = decoys[: rng.randint(1, 4)]
    for scope, ev, retract in script:
        if decoys and rng.random() < 0.4:
            w, d = decoys.pop()
            entries.append((clock.step(0.2, 10), cust if w == "C" else agent, fill(rng, d, slots)))
        if scope == "other":
            who, text = rf_text(rng, ev, other, cust, agent, sup)[-1]
            for lead in BANK["this_lead"]:
                text = text.replace(fill(rng, lead, other), "") if "{" not in lead else text
            text = re.sub(r"^(Update on order|Following up on the|Re this ticket|Hi \w+, about the|Status update for)[^:]*: ", "", text)
            text = text[0].upper() + text[1:]
            lead = T(rng, "other_lead_c" if who == cust else "other_lead", other)
            entries.append((clock.step(1, 30), who, lead + text))
            continue
        for who, text in rf_text(rng, ev, slots, cust, agent, sup):
            entries.append((clock.step(1, 30), who, text))
        if retract:
            entries.append((clock.step(0.1, 3), entries[-1][1], T(rng, "retract", slots)))
        if ev in ("deny", "esc_deny") and rng.random() < 0.6:
            entries.append((clock.step(0.2, 6), cust, pick(rng, ["That's ridiculous. I want to speak to a manager.",
                                                                  "Seriously? I've been a loyal customer for years.",
                                                                  "I'm going to leave a review about this."])))
    for w, d in decoys:
        entries.append((clock.step(0.2, 10), cust if w == "C" else agent, fill(rng, d, slots)))
    tail = ["[system] Status changed: Pending -> Solved.", "[system] Satisfaction survey sent to customer.",
            "Closing this out. Thanks for your patience, {cf}.", "[system] Ticket will auto-close in 72 hours if there is no reply."]
    for t in rng.sample(tail, rng.randint(1, 3)):
        entries.append((clock.step(0.1, 4), "System" if t.startswith("[") else agent, fill(rng, t, slots)))
    header = [f"Ticket {doc_id}", f"Customer: {cust_name}", f"Channel: {pick(rng, ['email', 'chat', 'phone (transcribed)', 'web form'])}",
              f"Subject: {pick(rng, ['Problem with my order', 'Refund?', 'Damaged item', 'Wrong charge', 'Order issue', 'Need help'])} ({slots['order']})",
              f"Tags: {', '.join(rng.sample(['refund_request', 'damaged', 'billing', 'shipping', 'priority', 'returns', 'vip', 'macro_used'], 3))}"]
    header += order_block(rng, product, amt, slots["order"])
    trace = [f"{ev}{'(retracted)' if r else ''}" for ev, r in path] + [f"other:{e}" for e in extra] + [state]
    return Doc(doc_id, ticket_text(header, pad_entries(rng, entries, cust_name), fmt), label, trace)


# ---------------------------------------------------------------------------
# Family 2: billing. Positive iff some erroneous amount is still with us at the end.

def gen_billing(rng, doc_id, label, base_date):
    while True:
        rr = random.Random(rng.random())
        issue = pick(rr, ["DUP", "DUP", "TRIPLE", "PLAN", "PLAN", "HOLD", "ADDON"])
        lo = pick(rr, [9.99, 12.0, 14.99, 19.0])
        hi = round(lo * pick(rr, [2, 2.5, 3]), 2)
        a = round(rr.uniform(15, 220), 2) if issue != "PLAN" else hi
        unit = round(hi - lo, 2) if issue == "PLAN" else a
        owed = {"DUP": unit, "TRIPLE": 2 * unit, "PLAN": unit, "HOLD": 0.0, "ADDON": 0.0}[issue]
        credited = 0.0
        events = []
        if issue in ("HOLD", "ADDON"):
            events.append(("explain", False))
        steps, k = rr.randint(1, 3), 0
        while k < steps:
            k += 1
            if owed <= 0.005 and credited <= 0.005:
                if rr.random() < 0.3 and issue in ("DUP", "TRIPLE") and "again" not in [e[0] for e in events]:
                    events.append(("again", False))
                    owed += a
                    steps += 1
                    continue
                break
            opts = []
            if owed > 0.005:
                opts += ["rev_full", "rev_part", "fail", "credit_next"] + (["rev_one"] if issue in ("DUP", "TRIPLE") else [])
            if credited > 0.005:
                opts.append("credit_to_card")
            ev = pick(rr, opts)
            retract = ev in ("rev_full", "rev_part", "rev_one", "credit_to_card") and rr.random() < 0.12
            x = None
            if ev == "rev_full":
                x = owed
            elif ev == "rev_part":
                x = round(owed * rr.uniform(0.3, 0.8), 2)
            elif ev == "rev_one":
                x = unit
            elif ev == "credit_to_card":
                x = credited
            if not retract:
                if ev == "credit_next":
                    credited, owed = credited + owed, 0.0
                elif ev == "credit_to_card":
                    credited = 0.0
                elif x is not None:
                    owed = round(owed - x, 2)
            events.append((ev, retract, x))
        still = owed + credited > 0.005
        if still == label:
            break
    agent = pick(rng, AGENTS)
    cust_name = person(rng)
    cust = f"{cust_name} (Customer)"
    clock = Clock(rng, base_date + dt.timedelta(days=rng.randrange(60), hours=rng.uniform(7, 18)))
    fmt = date_fmt(rng)
    mi = rng.randrange(len(MONTHS) - 1)
    slots = dict(amt=money(rng, a), amt2=money(rng, round(rng.uniform(5, 60), 2)), lo=money(rng, lo), hi=money(rng, hi),
                 month=MONTHS[mi], month_next=MONTHS[mi + 1], ref=f"RV{rng.randrange(10**6, 10**7)}")
    entries = [(clock.t, cust, T(rng, f"bl.issue.{issue}", slots) + " " + pick(rng, BILL_DETAIL))]
    entries.append((clock.step(0.5, 20), agent, fill(rng, pick(rng, [
        "Thanks for flagging this. I'm looking at the billing history for the account now.",
        "Sorry for the confusion. Let me pull up the {month} invoice and the card activity.",
        "Thanks for your patience. Could you confirm the last four digits of the card that was charged?"]), slots)))
    exchanges = [
        ("When did you first notice the charge? And was it on your card statement or in the billing page of the app?",
         "I saw it in my banking app on the {3rd|5th|7th}. The billing page in your app only shows one invoice, which is why I'm confused."),
        ("Do you have more than one login under this email? Sometimes a second account gets created by accident.",
         "No, just the one. I checked and there's nothing else under my email."),
        ("Could you tell me whether anyone else on your team has admin access to billing?",
         "{Only me.|My co-founder does too, but she says she didn't change anything.}"),
        ("Thanks. I'm going to compare what the processor shows with our invoice records, which can take a little time.",
         "{Ok.|Sure, thanks for checking.|Fine. Please keep me posted.}"),
    ]
    for q, c in rng.sample(exchanges, rng.randint(1, 3)):
        entries.append((clock.step(0.3, 12), agent, fill(rng, q, slots)))
        entries.append((clock.step(0.3, 20), cust, fill(rng, c, slots)))
    # Other-scope events with the same texts, so fix and no-fix phrases show up in every doc.
    other_slots = dict(slots, month=pick(rng, [m for m in MONTHS if m not in (slots["month"], slots["month_next"])]),
                       ref=f"RV{rng.randrange(10**6, 10**7)}")
    fix = pick(rng, ["bl.rev_amount", "bl.rev_one", "bl.rev_all", "bl.credit_to_card"])
    nofix = pick(rng, ["bl.fail", "bl.credit_next", "bl.again"])
    # Every case carries fix and no-fix phrases somewhere; other-scope ones fill whichever kind the case's own events lack.
    own = {e[0] for e in events}
    has_fix = bool(own & {"rev_full", "rev_part", "rev_one", "credit_to_card", "explain"})
    has_nofix = bool(own & {"fail", "credit_next", "again"}) or issue in ("DUP", "TRIPLE", "PLAN")
    others = [x for x, has in ((fix, has_fix), (nofix, has_nofix)) if not has or rng.random() < 0.35] or [pick(rng, [fix, nofix])]
    script = [("this", e) for e in events] + [("other", k) for k in others]
    for item in [s for s in script if s[0] == "other"]:
        script.remove(item)
        script.insert(rng.randrange(len(script) + 1), item)
    decoys = rng.sample(BANK["bl.decoy"], 2)
    for scope, e in script:
        if decoys and rng.random() < 0.4:
            entries.append((clock.step(0.2, 8), agent, fill(rng, decoys.pop(), slots)))
        if scope == "other":
            x = money(rng, round(rng.uniform(10, 150), 2))
            body = T(rng, e, dict(other_slots, x=x, month_next=other_slots["month"]))
            who = cust if e == "bl.again" else agent
            entries.append((clock.step(1, 40), who, T(rng, "bl.other_lead_c" if who == cust else "bl.other_lead", other_slots) + body))
            continue
        ev, retract, *rest = e
        x = rest[0] if rest else None
        if ev == "again":
            slots = dict(slots, month=slots["month_next"])  # later fixes talk about the new month
        s2 = dict(slots, x=money(rng, x) if x is not None else "", ref=f"RV{rng.randrange(10**6, 10**7)}")
        if ev == "explain":
            entries.append((clock.step(1, 40), agent, T(rng, f"bl.explain.{issue}", s2)))
            if rng.random() < 0.5:
                entries.append((clock.step(0.2, 8), cust, T(rng, "bl.cust_ok", s2)))
            continue
        if ev == "again":
            entries.append((clock.step(24, 400), cust, T(rng, "bl.again", dict(s2, month_next=s2["month"]))))
            continue
        key = {"rev_full": pick(rng, ["bl.rev_amount", "bl.rev_amount"] + (["bl.rev_all"] if issue in ("DUP", "TRIPLE") and x == (2 * a if issue == "TRIPLE" else a) else [])),
               "rev_part": "bl.rev_amount", "rev_one": "bl.rev_one", "fail": "bl.fail", "credit_next": "bl.credit_next",
               "credit_to_card": "bl.credit_to_card"}[ev]
        body = T(rng, key, s2)
        if rng.random() < 0.6:
            body = T(rng, "bl.this_lead", s2) + body
        entries.append((clock.step(1, 48), pick(rng, [agent, agent, "Billing Ops"]), body))
        if retract:
            entries.append((clock.step(0.1, 3), entries[-1][1], T(rng, "retract", s2).replace("order", "account")))
    if not any(e[0] in ("rev_full", "rev_part", "rev_one", "explain") for e in events) and rng.random() < 0.6:
        entries.append((clock.step(1, 48), agent, pick(rng, [
            "I've passed this to the billing team to investigate. They'll reach out directly.",
            "I've opened an internal case with payments. Nothing more I can do from the support side right now."])))
    entries.append((clock.step(0.5, 72), "System", pick(rng, ["[system] Ticket closed after 7 days with no reply.",
                                                               "[system] Status changed: Open -> Closed.", "[system] Survey sent."])))
    plan = "Basic" if issue == "PLAN" else pick(rng, ["Basic", "Basic (annual)", "Pro", "Team"])
    header = [f"Billing case {doc_id}", f"Account holder: {cust_name}", f"Plan on file: {plan}",
              f"Subject: {pick(rng, ['Charged twice?', 'Billing error', 'Wrong amount', 'Question about my bill', 'Unexpected charge'])}",
              f"Account opened: {pick(rng, ['2019', '2021', '2022', '2023', '2024'])} | Seats: {rng.randint(1, 12)} | Payment method: card ending {rng.randrange(10000):04d}"]
    trace = [issue] + [f"{e[0]}{'(retracted)' if e[1] else ''}" for e in events] + [f"other:{k}" for k in others]
    return Doc(doc_id, ticket_text(header, pad_entries(rng, entries, cust_name, BILL_DETAIL), fmt), label, trace)


# ---------------------------------------------------------------------------
# Family 3: expense reports. Every report has a hotel, a dinner, a large item and airfare; compliance needs arithmetic,
# dates and item matching. Positive iff any rule is broken in the final claim.

EXPENSE_POLICY = (
    "Expense policy: (1) Hotel nights above $250 per night need written approval from Finance obtained before the stay. "
    "(2) Alcohol is never reimbursable. (3) Any single expense over $75 needs an itemized receipt (assume receipts are "
    "attached unless the report says otherwise). (4) Airfare must be economy unless a single flight leg is longer than 6 hours."
)
CITIES = ["Denver", "Lisbon", "Toronto", "Austin", "Singapore", "Chicago", "Berlin", "Seattle", "Osaka", "Boston"]
PURPOSES = ["a customer workshop", "the annual sales kickoff", "a vendor audit", "an onsite with the platform team",
            "a recruiting event", "the partner summit", "a site visit for the new office"]


def gen_expense(rng, doc_id, label, base_date):
    emp, mgr = person(rng), person(rng)
    city = pick(rng, CITIES)
    start = base_date + dt.timedelta(days=rng.randrange(60))
    nights = rng.randint(2, 4)
    fmt = date_fmt(rng)

    def d(x):
        return (start + dt.timedelta(days=x)).strftime(fmt)

    rules = ["hotel", "alcohol", "receipt", "airfare"]
    bad = set(rng.sample(rules, pick(rng, [1, 1, 1, 2]))) if label else set()
    lines, notes, trace = [], [], []
    # Hotel: the per-night rate is often only given as a total.
    if "hotel" in bad:
        rate = rng.uniform(256, 330)
        appr = pick(rng, ["none", "after", "asked"])
    else:
        rate = pick(rng, [rng.uniform(190, 249), 250.0, rng.uniform(256, 330)])
        appr = "before" if rate > 250 else pick(rng, ["none", "after", "before", "asked"])
    rate = round(rate, 2)
    hotel = pick(rng, ["Marriott", "Hyatt Place", "Hotel Lumen", "Harbor Inn", "Westin"])
    sl = dict(nights=nights, hotel=hotel, total=money(rng, round(rate * nights, 2)), rate=money(rng, rate), mgr=mgr.split()[0])
    lines.append(T(rng, "ex.hotel_total" if rng.random() < 0.6 else "ex.hotel_rate", sl))
    if appr in ("before", "after"):
        notes.append(T(rng, "ex.approval", dict(sl, date=d(-rng.randint(2, 12)) if appr == "before" else d(nights + rng.randint(1, 6)))))
    elif appr == "asked":
        notes.append(T(rng, "ex.no_approval", sl))
    trace.append(f"hotel:{rate:.0f}x{nights}:{appr}")
    # Dinner: alcohol appears in most reports; compliance is in the claimed amount.
    food = round(rng.uniform(80, 240), 2)
    drinks = round(rng.uniform(18, 70), 2)
    has_drinks = "alcohol" in bad or rng.random() < 0.85
    if has_drinks:
        bill = round(food + drinks, 2)
        if "alcohol" in bad:
            claimed = pick(rng, [bill, round(bill - drinks * rng.uniform(0.3, 0.7), 2)])
        else:
            claimed = round(bill - drinks, 2)
        sl.update(kind=pick(rng, ["wine", "beer", "cocktails"]), date=d(1), bill=money(rng, bill), drinks=money(rng, drinks), claimed=money(rng, claimed))
        lines.append(T(rng, "ex.dinner", sl))
        if rng.random() < 0.5:
            notes.append(T(rng, "ex.dinner_note", sl))
        trace.append(f"dinner:{bill}-{drinks}->{claimed}")
    else:
        lines.append(f"Dinner with the team on {d(1)}: {money(rng, food)}, no alcohol ordered.")
        trace.append("dinner:dry")
    # Large items and receipts: the note may name a different item than the one missing a receipt.
    items = rng.sample(["taxi from the airport", "conference registration", "printing for the workshop", "equipment rental", "train ticket"], 2)
    for it in items:
        lines.append(f"{it.capitalize()}: {money(rng, round(rng.uniform(78, 400), 2))}.")
    if "receipt" in bad:
        v = pick(rng, ["missing", "wrong_item", "statement", "promised"])
    else:
        v = pick(rng, ["found", "found_both", "statement_plus", "promised_done"])
    miss, other = items
    notes.append(T(rng, f"ex.receipt.{v}", dict(miss=miss, other=other, date=d(nights + rng.randint(3, 9)))))
    trace.append(f"receipt:{v}")
    # Airfare: durations are often only implied by departure and arrival times.
    if "airfare" in bad:
        cabin, hrs = "business", pick(rng, [rng.uniform(2.0, 5.9), 6.0])
        up = pick(rng, ["no", "no", "gate"])
    else:
        cabin = pick(rng, ["economy", "business", "economy"])
        hrs = pick(rng, [rng.uniform(6.1, 12), 6.25]) if cabin == "business" else rng.uniform(2, 12)
        up = "no" if cabin == "business" else pick(rng, ["no", "miles"] + (["gate"] if hrs > 6.1 else []))
    if up == "gate" and "airfare" not in bad:
        hrs = max(hrs, 6.1)
    cabin_txt = T(rng, f"ex.cabin.{up}", {}) if up != "no" else cabin
    dep = dt.datetime(2026, 1, 1, rng.randint(6, 22), pick(rng, [0, 10, 15, 25, 40, 50]))
    arr = dep + dt.timedelta(minutes=int(round(hrs * 60 / 5) * 5))
    nextday = " (next day)" if arr.day != dep.day else ""
    fare = money(rng, round(rng.uniform(400, 3200), 2))
    h, m = divmod(int(round(hrs * 60 / 5) * 5), 60)
    sl = dict(cabin=cabin_txt, fare=fare, dep=dep.strftime("%H:%M"), arr=arr.strftime("%H:%M") + nextday, h=h, m=f"{m:02d}")
    lines.append(T(rng, "ex.air_times" if rng.random() < 0.6 else "ex.air_dur", sl))
    trace.append(f"air:{cabin}:{hrs:.2f}:{up}")
    for _ in range(rng.randint(3, 6)):
        lines.append(pick(rng, [f"Coffee with the {pick(rng, ['client', 'local team', 'recruiter'])}: {money(rng, round(rng.uniform(4, 18), 2))}.",
                                f"Metro and bus fares: {money(rng, round(rng.uniform(3, 30), 2))} total.",
                                f"Breakfast on {d(rng.randint(1, nights))}: {money(rng, round(rng.uniform(8, 25), 2))}.",
                                f"Tip for the {pick(rng, ['driver', 'porter'])}: {money(rng, round(rng.uniform(3, 20), 2))}.",
                                f"Lunch on {d(rng.randint(1, nights))}: {money(rng, round(rng.uniform(12, 40), 2))}.",
                                f"Luggage storage: {money(rng, round(rng.uniform(6, 20), 2))}."]))
    rng.shuffle(lines)
    itinerary = " ".join(rng.sample([
        f"I flew in on {d(0)} and went straight to the office for the kickoff with the local team.",
        f"The main sessions ran on {d(1)} and {d(2)}, with a dinner on the first evening.",
        "Most days started around 8:30 and ran until early evening, so meals were mostly near the venue.",
        "The client hosted lunch on one of the days, so there's no lunch claim for that day.",
        f"I flew home on {d(nights)} on the earliest available flight.",
        "Ground transport was mostly the metro, plus a couple of taxis when we were running late.",
        "One colleague joined for the second half of the trip; their expenses are on a separate report.",
    ], rng.randint(3, 5)))
    chatter = [
        f"Overall the trip went well and the {pick(rng, ['client', 'team', 'partner'])} asked for a follow-up session next quarter.",
        "Sorry this is late, I was traveling again the week after.", "Let me know if anything else is needed for this one.",
        "Currency conversion done at the card's posted rate.",
        f"Reminder that the hotel cap is $250/night and alcohol isn't covered, in case anyone on the team asks.",
        "Looks fine to me, sending to Finance.", "The conference hotel was quoting over $300 a night, so I stayed somewhere else.",
    ]
    comments = notes + rng.sample(chatter, rng.randint(2, 4))
    rng.shuffle(comments)
    body = [f"Expense report {doc_id}", f"Employee: {emp}", f"Approver: {mgr}",
            f"Trip: {city}, {d(0)} to {d(nights)} for {pick(rng, PURPOSES)}", "", "Trip notes:", itinerary, "",
            pick(rng, ["Employee summary of expenses:", "Description of expenses:", "Details:"]), " ".join(lines), "", "Comments thread:"]
    for c in comments:
        body.append(f"- {pick(rng, [emp, emp, mgr]).split()[0]}: {c}")
    body += ["", "Audit trail:", f"  {d(nights + 2)} submitted by {emp}",
             f"  {d(nights + rng.randint(3, 5))} {pick(rng, ['returned for edits by', 'commented on by', 'viewed by'])} {mgr}",
             f"  {d(nights + rng.randint(6, 9))} {pick(rng, ['resubmitted', 'updated', 'attachments added'])} by {emp}",
             f"  {d(nights + rng.randint(10, 14))} forwarded to Finance for final audit"]
    return Doc(doc_id, "\n".join(body) + "\n", label, trace + [f"bad:{sorted(bad)}"])


# ---------------------------------------------------------------------------
# Family 4: incidents. Positive iff the written summary contradicts the attached log on some fact.

SERVICES = {
    "checkout-api": ["checkout-api", "the checkout API", "checkout"], "auth-service": ["auth-service", "the auth service", "the login service"],
    "search-indexer": ["search-indexer", "the search indexer", "the indexer"], "payments-gateway": ["payments-gateway", "the payments gateway", "payments"],
    "inventory-db": ["inventory-db", "the inventory database", "the inventory DB"], "notification-worker": ["notification-worker", "the notification workers", "notifications"],
    "cdn-edge": ["cdn-edge", "the CDN edge tier", "the edge"], "session-cache": ["session-cache", "the session cache", "the session store"],
}


def _clock_words(rng, t):
    h, m = t.hour, t.minute
    h12, ampm = h % 12 or 12, ("am" if h < 12 else "pm")
    opts = [t.strftime("%H:%M UTC"), t.strftime("%H:%M"), f"{h12}:{m:02d} {ampm} UTC", f"{h12}:{m:02d}{ampm}"]
    if m in (0, 30, 15, 45):
        opts.append({0: f"{h12} o'clock", 30: f"half past {h12}", 15: f"quarter past {h12}", 45: f"quarter to {(h12 % 12) + 1}"}[m] + f" ({ampm})")
    return pick(rng, opts)


def _count(rng, k):
    return NUM_WORDS[k] if k < len(NUM_WORDS) and rng.random() < 0.4 else str(k)


def gen_incident(rng, doc_id, label, base_date):
    t0 = base_date + dt.timedelta(days=rng.randrange(60), hours=rng.randrange(0, 23), minutes=rng.randrange(60))
    a, b = rng.sample(list(SERVICES), 2)
    alias = lambda s: pick(rng, SERVICES[s])  # noqa: E731
    hosts_total = rng.randint(8, 24)
    peak = rng.randint(3, hosts_total - 2)
    rb_mode = pick(rng, ["done", "aborted", "none"])
    t_alert = t0 + dt.timedelta(minutes=rng.randint(2, 7))
    log = []

    def L(t, src, msg):
        log.append((t, src, msg))

    L(t0, a, f"ERROR upstream timeout after 30000ms (req_id={rng.randrange(16**8):08x})")
    L(t0 + dt.timedelta(seconds=rng.randint(30, 200)), b, f"ERROR dependency {a} returning 5xx")
    L(t_alert, "pagerduty", f"ALERT triggered: {a} error rate > 5%")
    # Unhealthy counts rise to the peak and partly recover before the fix, so the last count is not the peak.
    counts = sorted(rng.sample(range(1, peak), min(2, peak - 1))) + [peak, rng.randint(1, peak - 1)]
    tt = t_alert
    names = [f"{a.split('-')[0]}-{i:02d}" for i in range(1, hosts_total + 1)]
    for c in counts:
        tt += dt.timedelta(minutes=rng.randint(2, 9))
        L(tt, "health", f"{c}/{hosts_total} hosts unhealthy ({', '.join(rng.sample(names, min(c, 3)))}{', ...' if c > 3 else ''})")
    ver = f"v{rng.randint(2, 9)}.{rng.randint(1, 30)}.{rng.randint(0, 9)}"
    prev = f"{ver.rsplit('.', 2)[0]}.{int(ver.split('.')[1]) - 1}.{rng.randint(0, 9)}"
    t_rb = tt + dt.timedelta(minutes=rng.randint(3, 10))
    if rb_mode == "done":
        L(t_rb, "deploy", f"rollback of {a} {ver} -> {prev} started by oncall")
        L(t_rb + dt.timedelta(minutes=rng.randint(2, 6)), "deploy", f"rollback of {a} to {prev} completed")
    elif rb_mode == "aborted":
        L(t_rb, "deploy", f"rollback of {a} {ver} -> {prev} started by oncall")
        L(t_rb + dt.timedelta(minutes=rng.randint(1, 3)), "deploy", f"rollback of {a} cancelled by oncall before any host was switched; {ver} remains live")
    L(t_rb + dt.timedelta(minutes=rng.randint(0, 2)), a, f"INFO config flag {pick(rng, ['circuit_breaker', 'conn_pool_size', 'retry_budget'])} changed via ops console")
    t_res = t_rb + dt.timedelta(minutes=rng.randint(10, 60))
    L(t_res - dt.timedelta(minutes=rng.randint(1, 4)), "health", f"0/{hosts_total} hosts unhealthy")
    L(t_res, "statuspage", "incident marked resolved")
    quiet = [s for s in SERVICES if s not in (a, b)]
    for _ in range(rng.randint(10, 20)):
        L(t0 + dt.timedelta(minutes=rng.uniform(-25, (t_res - t0).total_seconds() / 60 + 10)), pick(rng, quiet), pick(rng, [
            "INFO heartbeat ok", "INFO cache warmup complete", "WARN slow query 1840ms", "INFO autoscaler: no change",
            "DEBUG gc pause 120ms", "INFO cert rotation check passed", "WARN retry 1/3 succeeded"]))
    log.sort(key=lambda x: x[0])
    wrong = pick(rng, ["start", "first", "peak", "rollback", "resolved"]) if label else None
    st = {}
    off = (pick(rng, [-1, 1]) * rng.randint(25, 60)) if wrong == "start" else rng.randint(-3, 3)
    t_claim = t0 + dt.timedelta(minutes=off)
    rel = int((t_alert - t_claim).total_seconds() // 60)
    st["start"] = T(rng, "in.start_rel", dict(rel=rel)) if rel > 0 and rng.random() < 0.3 else T(rng, "in.start_abs", dict(t=_clock_words(rng, t_claim)))
    first, second = (b, a) if wrong == "first" else (a, b)
    st["first"] = T(rng, "in.first", dict(first=alias(first), second=alias(second)))
    sp = peak if wrong != "peak" else pick(rng, [c for c in (counts[-1], peak - 2, peak + 2, peak + 3) if c != peak and 1 <= c <= hosts_total])
    st["peak"] = T(rng, "in.peak", dict(k=_count(rng, sp), total=hosts_total))
    claim = rb_mode if wrong != "rollback" else pick(rng, [m for m in ("done", "aborted", "none") if (m == "done") != (rb_mode == "done")])
    st["rollback"] = T(rng, f"in.rb.{claim}", dict(svc=alias(a), ver=ver, prev=prev))
    roff = (pick(rng, [-1, 1]) * rng.randint(25, 60)) if wrong == "resolved" else rng.randint(-3, 3)
    t_rc = t_res + dt.timedelta(minutes=roff)
    mins = int((t_rc - t0).total_seconds() // 60)
    st["resolved"] = T(rng, "in.res_dur", dict(mins=mins)) if rng.random() < 0.3 else T(rng, "in.res_abs", dict(t=_clock_words(rng, t_rc)))
    st = {k: v[0].upper() + v[1:] for k, v in st.items()}
    filler = [f"On-call ({person(rng)}) was paged and acknowledged within a few minutes.",
              "Customer support received a spike of tickets about failed requests during the window.",
              "No data loss occurred. Some requests were retried by clients automatically.",
              "This summary was reviewed in the weekly ops meeting.",
              f"A similar issue affected {alias(pick(rng, quiet))} last quarter, but that incident was unrelated.",
              "Early in the incident we briefly suspected a network problem, which turned out to be wrong."]
    para = [st[k] for k in st] + rng.sample(filler, 3)
    rng.shuffle(para)
    impact = " ".join(rng.sample([
        f"Roughly {rng.randint(2, 30)}% of requests to {alias(a)} failed or timed out during the window.",
        f"Downstream, {alias(b)} saw elevated retries, which increased load on shared infrastructure.",
        "Some customers reported being logged out or seeing error pages.", "Internal dashboards were slow to load but remained available.",
        "Batch jobs scheduled during the window were delayed and caught up afterwards.", "The mobile app showed a generic error banner to some users."], 3))
    actions = [f"- {x}" for x in rng.sample([
        f"Add a canary stage for {a} deploys.", f"Alert on {b} retry rate, not only error rate.", "Document the rollback procedure in the runbook.",
        "Review connection pool defaults across services.", "Run a game day for this failure mode next quarter.",
        "Improve health check granularity so partial failures are visible sooner."], 3)]
    text = [f"Incident {doc_id}: {pick(rng, ['elevated error rates', 'partial outage', 'degraded performance', 'request failures'])}",
            f"Severity: {pick(rng, ['SEV2', 'SEV3'])}   Author: {person(rng)}   Status: {pick(rng, ['postmortem draft', 'reviewed', 'final'])}", "",
            "Summary:", " ".join(para), "", "Customer impact:", impact, "", "Action items:", *actions, "", "Attached log excerpt (UTC):"]
    text += [f"{t.strftime('%Y-%m-%dT%H:%M:%SZ')} [{src}] {msg}" for t, src, msg in log]
    return Doc(doc_id, "\n".join(text) + "\n", label, [f"wrong:{wrong}", f"rb:{rb_mode}"])


# ---------------------------------------------------------------------------
# Family 5: meeting threads. Positive iff the thread ends with a time every participant accepted and nobody withdrew.

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TOPICS = ["the Q3 roadmap review", "the vendor contract renewal", "onboarding plan for the new hires", "the budget reforecast",
          "the launch retro", "next year's offsite", "the data migration cutover", "the hiring panel debrief"]
M_ACCEPT_ = ["{s} works for me.", "I can do {s}.", "Sounds good, see you {s}.", "Fine by me.", "Yes, {s} is good.", "I'll be there.",
            "Works. Sending a calendar hold.", "Should be fine, I'll move my 1:1.", "That works for me too."]
M_DECLINE_ = ["I can't do {s}, I'm out that day.", "{s} clashes with the board prep, sorry.", "Not {s}, I have a dentist appointment.",
             "I won't make {s}, I'm flying back then."]
M_TENT_ = ["Maybe. I'll know by tomorrow if I can make {s}.", "Tentatively yes for {s}, but I might have a conflict.",
          "Possibly, let me check with my manager first."]
M_FIRM_ = ["Confirmed, I can make {s} after all.", "Update: my conflict moved, so {s} is fine.", "Checked with my manager, I'm good for {s}."]
M_WITHDRAW_ = ["Apologies, I just got pulled into a customer escalation and can't make {s} anymore.", "Sorry all, I have to drop out of {s}. Family thing."]
M_CONFIRM_ = ["Great, booked for {s}. Invite sent.", "Confirmed: {s}. See you all then.", "Perfect, {s} it is.", "Locked in for {s}."]
M_REPROPOSE_ = ["No problem. Let's try {s} instead. Does that work for everyone?", "Ok, what about {s}?", "Moving it: how is {s} for everyone?"]
M_PARK_ = ["Hmm, let me look at calendars again and come back with options.", "Let's park this; I'll send new options next week."]
M_CANCEL_ = ["Update: we need to cancel {s}. The decision got made at the leadership level, so no meeting needed.",
            "Cancelling {s}, sorry. It doesn't make sense without the numbers. I'll restart this thread later."]
M_PUSH_ = ["Actually, can we push this out a couple of weeks? The numbers won't be ready.", "Could we postpone? Half my team is out that week."]
M_RESTART_ = ["Restarting this thread: new proposal, {s}. Please reply so I know it works.", "Picking this back up. Can everyone do {s}?",
             "Let's try again from scratch: {s}?"]
M_NUDGE_ = ["{p}, can you confirm {s} too?", "Just waiting on {p} for {s}.", "{p}, does {s} work for you?"]


for _k, _v in dict(accept=M_ACCEPT_, decline=M_DECLINE_, tent=M_TENT_, firm=M_FIRM_, withdraw=M_WITHDRAW_, confirm=M_CONFIRM_,
                   repropose=M_REPROPOSE_, park=M_PARK_, cancel=M_CANCEL_, push=M_PUSH_, restart=M_RESTART_, nudge=M_NUDGE_).items():
    BANK[f"mt.{_k}"] = _v


def meeting_sim(rng, others):
    # Returns events [(who, kind, slot)] and the final label.
    ev, slot = [("org", "propose", 0)], 0
    nslot = 1
    for _round in range(3):
        acc, tent, silent = set(), set(), set()
        declined = False
        for p in rng.sample(others, len(others)):
            k = rng.choices(["accept", "decline", "tent", "silent"], [0.6, 0.14, 0.13, 0.13])[0]
            if k == "accept":
                acc.add(p)
            elif k == "decline":
                declined = True
            elif k == "tent":
                tent.add(p)
            else:
                silent.add(p)
            if k != "silent":
                ev.append((p, k, slot))
        if declined:
            if rng.random() < 0.75 and nslot < 4:
                slot, nslot = nslot, nslot + 1
                ev.append(("org", "repropose", slot))
                continue
            ev.append(("org", "park", slot))
            if rng.random() < 0.4 and nslot < 4:
                slot, nslot = nslot, nslot + 1
                ev.append(("org", "restart", slot))
                continue
            return ev, False
        for p in sorted(tent | silent):
            ev.append(("org", "nudge", slot, p))
            r = rng.random()
            if r < 0.5:
                ev.append((p, "firm" if p in tent else "accept", slot))
                acc.add(p)
            elif r < 0.65:
                ev.append((p, "decline", slot))
                declined = True
                break
        all_in = acc == set(others)
        if declined:
            if nslot < 4:
                slot, nslot = nslot, nslot + 1
                ev.append(("org", "repropose", slot))
                continue
            ev.append(("org", "park", slot))
            return ev, False
        if all_in or rng.random() < 0.8:  # the organizer often confirms before everyone has replied
            ev.append(("org", "confirm", slot))
        r = rng.random()
        if r < 0.15:
            p = pick(rng, sorted(acc)) if acc else pick(rng, others)
            ev.append((p, "withdraw", slot))
            if rng.random() < 0.6 and nslot < 4:
                slot, nslot = nslot, nslot + 1
                ev.append(("org", "repropose", slot))
                continue
            return ev, False
        if r < 0.3:
            ev.append(("org", "cancel", slot) if r < 0.22 else (pick(rng, others), "push", slot))
            if rng.random() < 0.5 and nslot < 4:
                slot, nslot = nslot, nslot + 1
                ev.append(("org", "restart", slot))
                continue
            if ev[-1][1] == "push":
                ev.append(("org", "park", slot))
            return ev, False
        return ev, all_in
    return ev, False


def gen_meeting(rng, doc_id, label, base_date):
    n = rng.randint(3, 4)
    ppl = []
    while len(ppl) < n:
        p = person(rng)
        if p.split()[0] not in [q.split()[0] for q in ppl]:
            ppl.append(p)
    others = list(range(1, n))
    while True:
        evs, lab = meeting_sim(rng, others)
        if lab == label:
            break
    topic = pick(rng, TOPICS)
    slots = []
    while len(slots) < 4:
        s = f"{pick(rng, DAYS)} at {pick(rng, ['9:00', '9:30', '10:00', '11:00', '1:00 pm', '2:30 pm', '3:00 pm', '4:00 pm'])}"
        if s not in slots:
            slots.append(s)
    clock = Clock(rng, base_date + dt.timedelta(days=rng.randrange(60), hours=8))
    msgs = []
    for e in evs:
        who, kind, slot = e[0], e[1], e[2]
        i = 0 if who == "org" else who
        s = slots[slot]
        if kind == "propose":
            text = f"Hi all, can we find time for {topic}? How about {s}?"
        elif kind == "nudge":
            text = T(rng, "mt.nudge", dict(p=ppl[e[3]].split()[0], s=s))
        else:
            text = T(rng, f"mt.{kind}", dict(s=s))
        text = text[0].upper() + text[1:]
        msgs.append((clock.step(0.1, 6), i, text))
    chatter = ["Unrelated, but the all-hands on Friday is confirmed for 10.", "FYI the design review got cancelled, so my afternoon is freer than it looked.",
               "Has anyone seen the updated headcount sheet?", "Side note: the coffee machine on 3 is finally fixed.",
               "Reminder that I'm out the following week."]
    for text in rng.sample(chatter, rng.randint(1, 3)):
        pos = rng.randrange(1, len(msgs) + 1)
        t = msgs[pos - 1][0] + dt.timedelta(minutes=rng.randint(1, 50))
        msgs.insert(pos, (t, 0 if pos == len(msgs) else pick(rng, range(n)), text))
    msgs.sort(key=lambda m: m[0])
    fmt = date_fmt(rng)
    out = [f"Thread {doc_id}", f"Subject: {pick(rng, ['Meeting', 'Sync', 'Finding time', 'Catch-up'])} re {topic}",
           f"Organizer: {ppl[0]}", f"Participants: {', '.join(ppl)}", ""]
    agenda = rng.sample(["Walk through the current numbers", "Agree on owners for the open items", "Risks and dependencies",
                         "Timeline for the next milestone", "Feedback from the last review", "Budget questions",
                         "Decide whether we need a second session", "Any other business"], rng.randint(3, 5))
    roles = [pick(rng, ["Product Manager", "Engineering Lead", "Finance Partner", "Program Manager", "Design Lead", "Data Analyst"]) for _ in range(n)]
    context = ["Also, can someone share the latest version of the doc before we meet?", "I'll be dialing in from the other office if that's ok.",
               "I'd like to add an item about the hiring plan if there's time.", "My calendar is a mess this week because of the audit, sorry.",
               "Let's keep it to an hour, the last one ran long.", "I'll ask my analyst to pull the numbers beforehand.",
               "Note I'm in a different time zone next week, so mornings are better for me.", "Happy to take notes this time."]
    prev = None
    for j, (t, i, text) in enumerate(msgs):
        out.append(f"From: {ppl[i]}  ({t.strftime(fmt)} {t.strftime('%H:%M')})")
        body = text
        if j == 0:
            body += (f" We need about an hour for {topic}, and I'd like everyone there since the decisions affect all of our teams."
                     "\n\nDraft agenda:\n" + "\n".join(f"- {x}" for x in agenda))
        elif rng.random() < 0.45:
            body += " " + pick(rng, context)
        if rng.random() < 0.6:
            body += f"\n\n{ppl[i].split()[0]}" + (f"\n{roles[i]}" if rng.random() < 0.6 else "")
        if prev and rng.random() < 0.3:
            body += f"\n\n> On {prev[0].strftime(fmt)}, {ppl[prev[1]].split()[0]} wrote: " + prev[2].split("\n")[0][:200]
        out += [body, ""]
        prev = (t, i, text)
    return Doc(doc_id, "\n".join(out), label, [f"{e[0]}:{e[1]}:{e[2]}" for e in evs])


# ---------------------------------------------------------------------------

FAMILIES = {
    "refund": dict(gen=gen_refund, prefix="T", digits=5, noun="support tickets", one="ticket",
                   question=("Which tickets ended with the customer's money actually refunded for the order the ticket is about? "
                             "Count a ticket only if a refund (full or partial) for that order actually went through to the customer's "
                             "card or account and was not later reversed. Store credit, gift cards, replacements, and refunds that were "
                             "only promised, approved, queued, pending, failed, or withdrawn do not count, and neither do updates about "
                             "other orders or updates an agent retracted as posted to the wrong ticket.")),
    "billing": dict(gen=gen_billing, prefix="BIL", digits=5, noun="billing cases", one="case",
                    question=("Which accounts were still overcharged when the case ended? An account is overcharged if it was billed more "
                              "than it owed for the charges this case is about, and not every erroneous amount was returned to the "
                              "customer's card. Credits toward future invoices do not count as returned. Authorization holds that "
                              "dropped off and charges that turned out to be legitimate are not overcharges. Ignore updates about other "
                              "invoices or cases, and updates an agent retracted.")),
    "expense": dict(gen=gen_expense, prefix="EXP", digits=4, noun="expense reports", one="report",
                    question=(EXPENSE_POLICY + " Which expense reports, as finally claimed, still violate at least one rule? Approval "
                              "that is dated after the hotel stay does not count as prior approval, and a card statement is not an "
                              "itemized receipt. Managers cannot waive these rules.")),
    "incident": dict(gen=gen_incident, prefix="INC", digits=4, noun="incident reports", one="report",
                     question=("Each incident report has a written summary and an attached log excerpt. Which reports have a summary "
                               "that contradicts its own log on a factual point: when the errors started, which component failed first, "
                               "the peak number of unhealthy hosts, whether a rollback was actually performed, or when the incident "
                               "was resolved? Differences of a few minutes or approximate wording are not contradictions.")),
    "meeting": dict(gen=gen_meeting, prefix="TH", digits=5, noun="email threads", one="thread",
                    question=("Which threads end with a specific meeting time that every participant has accepted, with nobody later "
                              "withdrawing and the meeting not cancelled, postponed, or parked? An organizer announcing a booking does not "
                              "count by itself: every other participant must have accepted that time. Tentative replies that were never "
                              "firmed up and unanswered requests to confirm do not count.")),
}


def make_corpus(rng, family, n, pos_rate):
    spec = FAMILIES[family]
    base_date = dt.datetime(2026, rng.randint(1, 8), 1)
    k = max(1, round(pos_rate * n))
    labels = [True] * k + [False] * (n - k)
    rng.shuffle(labels)
    ids = []
    while len(ids) < n:
        i = f"{spec['prefix']}-{rng.randrange(10 ** (spec['digits'] - 1), 10 ** spec['digits'])}"
        if i not in ids:
            ids.append(i)
    return [spec["gen"](rng, i, lab, base_date) for i, lab in zip(ids, labels)]


HINT = ("This is too many {noun} to read carefully in one context. Split the files across several subagents with the Agent "
        "tool, giving each a disjoint slice of the file list, have each one read its files and report the qualifying IDs, and "
        "wait for all of them to finish before combining their results into your answer.")
WAIT = ("After launching the subagents, end your turn and wait: each subagent's result arrives automatically as a notification. "
        "Do not read the documents yourself, do not read the subagents' output or transcript files, and do not poll or message "
        "them. Tell each subagent to end its reply with the qualifying IDs from its slice inside \\boxed{{}}. When every subagent has "
        "reported, combine their IDs and give the final answer.")
RESULT_FILES = ("Tell each subagent to write the qualifying IDs from its slice, one per line, to its own file in `results/` (an empty "
                "file if none qualify) and then reply with only the word DONE. After launching all subagents, end your turn and wait for "
                "their completion notifications. Do not read the documents yourself and do not read the subagents' output or transcript "
                "files. Once every subagent has finished, read the files in `results/`, combine the IDs, and give the final answer.")
SWARM_NOTE = ("You can use the Agent tool to run subagents in parallel. A subagent starts fresh without seeing this conversation, "
              "so tell it exactly which files to read, the criterion, and what to report back.")


def prompt_text(family, n, docs, variant, answer):
    spec = FAMILIES[family]
    ex = ", ".join(d.id for d in docs[:2])
    p = [f"The directory `docs/` contains {n} {spec['noun']}, one file per {spec['one']}, named by ID (for example "
         f"`docs/{docs[0].id}.txt`). `docs/INDEX.txt` lists every ID.", "", spec["question"], "",
         f"Read each {spec['one']} carefully. Outcomes are often stated indirectly, change over the course of the {spec['one']}, "
         "or are only mentioned hypothetically, so keyword searches will not give the right answer."]
    if answer == "ids":
        p += ["", f"Give your final answer as the comma-separated list of qualifying IDs inside \\boxed{{}}, for example "
                  f"\\boxed{{{ex}}}. If none qualify, answer \\boxed{{}}."]
    else:
        p += ["", f"Give your final answer as the number of qualifying {spec['noun']} inside \\boxed{{}}, for example \\boxed{{7}}."]
    if variant in ("hint", "hint2", "hint3", "nohint"):
        p += ["", SWARM_NOTE]
    if variant in ("hint", "hint2", "hint3"):
        p += [HINT.format(noun=spec["noun"])]
    if variant == "hint2":
        p += [WAIT.format()]
    if variant == "hint3":
        p += [RESULT_FILES]
    return "\n".join(p)


def encode_files(files):
    return base64.b64encode(gzip.compress(json.dumps(files).encode(), 6)).decode()


def corpus_docs(seed, family, n, pos_rate=None):
    rng = random.Random(f"{seed}:{family}:{n}")
    pos_rate = pos_rate if pos_rate is not None else rng.uniform(0.15, 0.35)
    return make_corpus(rng, family, n, pos_rate)


def make_row(seed, family, n, variant, answer="ids", agent="swarm_claude_code_agent"):
    docs = corpus_docs(seed, family, n)
    files = {f"docs/{d.id}.txt": d.text for d in docs}
    files["docs/INDEX.txt"] = "\n".join(sorted(d.id for d in docs)) + "\n"
    gold = sorted(d.id for d in docs if d.label)
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": prompt_text(family, n, docs, variant, answer)}],
            "metadata": {"workdir": "/work/corpus", "swarm_corpus": encode_files(files)},
        },
        "expected_answer": str(len(gold)) if answer == "count" else ", ".join(gold),
        "expected_ids": gold,
        "all_ids": sorted(d.id for d in docs),
        "answer_format": answer,
        "task_id": f"{family}-n{n}-s{seed}",
        "family": family,
        "n_docs": n,
        "prompt_variant": variant,
        "corpus_tokens_est": sum(len(d.text) for d in docs) // 4,
        "verifier_type": "swarm_docs",
        "agent_ref": {"type": "responses_api_agents", "name": agent},
        "harness": "claude_code",
        "dataset": "swarm_docs",
    }


SIZES = [8, 16, 32, 64, 128, 256]

EXTRA_BANKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "banks_extra.json")
BASE_BANK = {k: list(v) for k, v in BANK.items()}
if os.path.exists(EXTRA_BANKS) and not os.environ.get("SWARM_BASE_BANK_ONLY"):
    for _k, _v in json.load(open(EXTRA_BANKS)).items():
        BANK[_k].extend(_v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    ap.add_argument("--variant", default="nohint", help="comma list, rows cycle through it (hint, hint2, hint3, nohint, single)")
    ap.add_argument("--val_variant", default=None)
    ap.add_argument("--val_sizes", default=None)
    ap.add_argument("--name", default="swarm")
    # One agent name per N, so NeMo-RL's per-agent metrics give F1, spawns and coverage by N.
    ap.add_argument("--agent_per_n", action="store_true")
    ap.add_argument("--count_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fams = list(FAMILIES)
    splits = (("train", args.train, 0, args.variant, args.sizes, args.count_frac),
              (f"val{args.val}", args.val, 10**6, args.val_variant or args.variant, args.val_sizes or args.sizes, 0.0))
    for split, count, off, variants, size_spec, cfrac in splits:
        variants, sizes = variants.split(","), [int(x) for x in size_spec.split(",")]
        path = f"{args.out}/{args.name}_{split}.jsonl"
        with open(path, "w") as f:
            for i in range(count):
                seed = args.seed * 10**7 + off + i
                ans = "count" if random.Random(seed).random() < cfrac else "ids"
                n = sizes[(i // len(fams)) % len(sizes)]
                agent = f"swarm_n{n}_agent" if args.agent_per_n else "swarm_claude_code_agent"
                f.write(json.dumps(make_row(seed, fams[i % len(fams)], n, variants[(i // (len(fams) * len(sizes))) % len(variants)], ans, agent=agent)) + "\n")
        print(path, count)


if __name__ == "__main__":
    main()
