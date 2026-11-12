# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Business/business_135/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Business-135-I2: Brooklyn Jazz Symposium 2026

Checks: 17 weighted checks across pretix and twenty.
Strategy: docker exec (DB queries) for both sites.

Required env vars:
  SERVER_HOSTNAME, PRETIX_PORT, PRETIX_CONTAINER, PRETIX_DB_CONTAINER,
  TWENTY_PORT, TWENTY_CONTAINER, TWENTY_DB_CONTAINER.
"""

import os
import sys
import subprocess

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

PRETIX_PORT = os.getenv("PRETIX_PORT")
PRETIX_CONTAINER = os.getenv("PRETIX_CONTAINER")
PRETIX_DB_CONTAINER = os.getenv("PRETIX_DB_CONTAINER")
TWENTY_PORT = os.getenv("TWENTY_PORT")
TWENTY_CONTAINER = os.getenv("TWENTY_CONTAINER")
TWENTY_DB_CONTAINER = os.getenv("TWENTY_DB_CONTAINER")

for var in ("PRETIX_PORT", "PRETIX_CONTAINER", "PRETIX_DB_CONTAINER",
            "TWENTY_PORT", "TWENTY_CONTAINER", "TWENTY_DB_CONTAINER"):
    if not os.getenv(var):
        print(f"FATAL: {var} not set", file=sys.stderr)
        sys.exit(1)


# ── Result accumulator ────────────────────────────────────────────────────────
_checks: list[tuple[str, int, bool, str]] = []


def check(label: str, weight: int, passed: bool, detail: str = "") -> None:
    _checks.append((label, weight, passed, detail))
    status = "PASS" if passed else "FAIL"
    tail = f"  ({detail})" if detail else ""
    print(f"[{status}] ({weight}pt) {label}{tail}", file=sys.stderr)


# ── Helpers ───────────────────────────────────────────────────────────────────
def pretix_sql(query: str, timeout: int = 15) -> str:
    """Run a SQL query against the Pretix PostgreSQL database."""
    r = subprocess.run(
        ["docker", "exec", PRETIX_DB_CONTAINER,
         "psql", "-U", "pretix", "-d", "pretix", "-t", "-A", "-c", query],
        capture_output=True, text=True, timeout=timeout,
    )
    return r.stdout.strip()


def twenty_sql(query: str, timeout: int = 15) -> str:
    """Run a SQL query against the Twenty PostgreSQL database."""
    r = subprocess.run(
        ["docker", "exec", TWENTY_DB_CONTAINER,
         "psql", "-U", "postgres", "-d", "default", "-t", "-A", "-c", query],
        capture_output=True, text=True, timeout=timeout,
    )
    return r.stdout.strip()


def get_twenty_workspace_schema() -> str:
    """Discover the workspace schema name in Twenty's DB."""
    result = twenty_sql(
        "SELECT schema_name FROM information_schema.schemata "
        "WHERE schema_name LIKE 'workspace_%' ORDER BY schema_name LIMIT 1;"
    )
    return result.strip()


# ── Pretix checks ─────────────────────────────────────────────────────────────

def check_1_event_basics() -> None:
    """Event exists with correct name, slug, date, currency, and is live (EO #1, #11)."""
    try:
        row = pretix_sql(
            "SELECT e.slug, e.name::text, e.date_from::text, e.currency, e.live "
            "FROM pretixbase_event e "
            "JOIN pretixbase_organizer o ON e.organizer_id = o.id "
            "WHERE o.slug = 'urban-music' AND e.slug = 'bklyn-jazz-symposium-2026';"
        )
        if not row:
            check("1. Event basics + live", 2, False, "event not found")
            return
        parts = row.split("|")
        slug, name_json, date_from, currency, live = (
            parts[0], parts[1], parts[2], parts[3], parts[4]
        )
        ok = (
            slug == "bklyn-jazz-symposium-2026"
            and "Brooklyn Jazz Symposium 2026" in name_json
            and date_from.startswith("2026-10-18")
            and currency == "USD"
            and live == "t"
        )
        check("1. Event basics + live", 2, ok,
              f"slug={slug}, date={date_from[:10]}, currency={currency}, live={live}")
    except Exception as e:
        check("1. Event basics + live", 2, False, f"exception: {e}")


def check_2_categories() -> None:
    """Categories 'General Admission' and 'Experience Add-ons' exist (EO #2)."""
    try:
        rows = pretix_sql(
            "SELECT ic.name::text FROM pretixbase_itemcategory ic "
            "JOIN pretixbase_event e ON ic.event_id = e.id "
            "JOIN pretixbase_organizer o ON e.organizer_id = o.id "
            "WHERE o.slug = 'urban-music' AND e.slug = 'bklyn-jazz-symposium-2026';"
        )
        has_general = "General Admission" in rows
        has_addons = "Experience Add-ons" in rows
        check("2. Categories", 1, has_general and has_addons,
              f"general={'found' if has_general else 'missing'}, addons={'found' if has_addons else 'missing'}")
    except Exception as e:
        check("2. Categories", 1, False, f"exception: {e}")


def check_3_products() -> None:
    """Three products exist with correct prices and categories (EO #3)."""
    try:
        rows = pretix_sql(
            "SELECT i.name::text, i.default_price, ic.name::text "
            "FROM pretixbase_item i "
            "JOIN pretixbase_event e ON i.event_id = e.id "
            "JOIN pretixbase_organizer o ON e.organizer_id = o.id "
            "LEFT JOIN pretixbase_itemcategory ic ON i.category_id = ic.id "
            "WHERE o.slug = 'urban-music' AND e.slug = 'bklyn-jazz-symposium-2026';"
        )
        lines = [l for l in rows.split("\n") if l.strip()]
        expected = [
            ("General Admission Pass", "85.00", "General Admission"),
            ("VIP Backstage Pass", "220.00", "General Admission"),
            ("Jam Session Workshop", "60.00", "Experience Add-ons"),
        ]
        issues = []
        for name, price, cat in expected:
            matched = False
            for line in lines:
                if name in line:
                    matched = True
                    if price not in line:
                        issues.append(f"{name}: wrong price")
                    if cat not in line:
                        issues.append(f"{name}: wrong category")
                    break
            if not matched:
                issues.append(f"{name}: not found")
        check("3. Products", 2, not issues,
              "all 3 correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("3. Products", 2, False, f"exception: {e}")


def check_4_quotas() -> None:
    """Quotas with correct sizes: Main Venue Capacity=350, Jam Session Quota=50 (EO #4)."""
    try:
        rows = pretix_sql(
            "SELECT q.name, q.size FROM pretixbase_quota q "
            "JOIN pretixbase_event e ON q.event_id = e.id "
            "JOIN pretixbase_organizer o ON e.organizer_id = o.id "
            "WHERE o.slug = 'urban-music' AND e.slug = 'bklyn-jazz-symposium-2026';"
        )
        lines = [l for l in rows.split("\n") if l.strip()]
        quotas = {}
        for line in lines:
            parts = line.split("|")
            if len(parts) >= 2:
                quotas[parts[0].strip()] = int(parts[1].strip())
        venue_ok = quotas.get("Main Venue Capacity") == 350
        workshop_ok = quotas.get("Jam Session Quota") == 50
        check("4. Quotas", 2, venue_ok and workshop_ok,
              f"venue={quotas.get('Main Venue Capacity')}, workshop={quotas.get('Jam Session Quota')}")
    except Exception as e:
        check("4. Quotas", 2, False, f"exception: {e}")


def check_5_questions() -> None:
    """Custom questions 'Company Affiliation' (text) and 'Dietary Preference' (choice) (EO #5)."""
    try:
        rows = pretix_sql(
            "SELECT q.question::text, q.type, q.required FROM pretixbase_question q "
            "JOIN pretixbase_event e ON q.event_id = e.id "
            "JOIN pretixbase_organizer o ON e.organizer_id = o.id "
            "WHERE o.slug = 'urban-music' AND e.slug = 'bklyn-jazz-symposium-2026';"
        )
        lines = [l for l in rows.split("\n") if l.strip()]
        issues = []
        found_affiliation = False
        found_dietary = False
        for line in lines:
            parts = line.split("|")
            if len(parts) >= 3:
                qtext, qtype, qreq = parts[0], parts[1].strip(), parts[2].strip()
                if "Company Affiliation" in qtext:
                    found_affiliation = True
                    # S = short text (one line), T = long text
                    if qtype not in ("S", "T"):
                        issues.append(f"Company Affiliation type={qtype}, expected S")
                    if qreq != "t":
                        issues.append("Company Affiliation not required")
                if "Dietary Preference" in qtext:
                    found_dietary = True
                    # C = choice single
                    if qtype != "C":
                        issues.append(f"Dietary Preference type={qtype}, expected C")
                    if qreq != "t":
                        issues.append("Dietary Preference not required")
        if not found_affiliation:
            issues.append("Company Affiliation not found")
        if not found_dietary:
            issues.append("Dietary Preference not found")
        check("5. Custom questions", 2, not issues,
              "both correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("5. Custom questions", 2, False, f"exception: {e}")


def check_6_vouchers() -> None:
    """Vouchers VIPJAZZ2026 (30%, max 8) and GROUPJAZZ40 (40 USD flat, max 25) (EO #6)."""
    try:
        rows = pretix_sql(
            "SELECT v.code, v.price_mode, v.value, v.max_usages, v.valid_until::text "
            "FROM pretixbase_voucher v "
            "JOIN pretixbase_event e ON v.event_id = e.id "
            "JOIN pretixbase_organizer o ON e.organizer_id = o.id "
            "WHERE o.slug = 'urban-music' AND e.slug = 'bklyn-jazz-symposium-2026' "
            "AND v.code IN ('VIPJAZZ2026', 'GROUPJAZZ40');"
        )
        lines = [l for l in rows.split("\n") if l.strip()]
        vouchers: dict[str, dict[str, str]] = {}
        for line in lines:
            parts = line.split("|")
            if len(parts) >= 5:
                vouchers[parts[0].strip()] = {
                    "price_mode": parts[1].strip(),
                    "value": parts[2].strip(),
                    "max_usages": parts[3].strip(),
                    "valid_until": parts[4].strip(),
                }
        issues = []
        vip = vouchers.get("VIPJAZZ2026")
        if not vip:
            issues.append("VIPJAZZ2026 not found")
        else:
            if vip["price_mode"] != "percent":
                issues.append(f"VIPJAZZ2026 mode={vip['price_mode']}")
            if "30" not in vip["value"]:
                issues.append(f"VIPJAZZ2026 value={vip['value']}, expected 30")
            if vip["max_usages"] != "8":
                issues.append(f"VIPJAZZ2026 max_usages={vip['max_usages']}")
        grp = vouchers.get("GROUPJAZZ40")
        if not grp:
            issues.append("GROUPJAZZ40 not found")
        else:
            if grp["price_mode"] != "subtract":
                issues.append(f"GROUPJAZZ40 mode={grp['price_mode']}")
            if "40" not in grp["value"]:
                issues.append(f"GROUPJAZZ40 value={grp['value']}, expected 40")
            if grp["max_usages"] != "25":
                issues.append(f"GROUPJAZZ40 max_usages={grp['max_usages']}")
        check("6. Vouchers", 2, not issues,
              "both correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("6. Vouchers", 2, False, f"exception: {e}")


def check_7_discount_rule() -> None:
    """Discount rule 'Jazz Early Bird' with 20% off applied to GA Pass (EO #7)."""
    try:
        rows = pretix_sql(
            "SELECT d.internal_name, d.benefit_discount_matching_percent "
            "FROM pretixbase_discount d "
            "JOIN pretixbase_event e ON d.event_id = e.id "
            "JOIN pretixbase_organizer o ON e.organizer_id = o.id "
            "WHERE o.slug = 'urban-music' AND e.slug = 'bklyn-jazz-symposium-2026';"
        )
        found = "Jazz Early Bird" in rows
        pct_ok = "20" in rows if found else False
        check("7. Discount rule", 1, found and pct_ok,
              f"found={found}, 20%={'yes' if pct_ok else 'no'}")
    except Exception as e:
        check("7. Discount rule", 1, False, f"exception: {e}")


def check_8_checkin_lists() -> None:
    """Check-in lists 'Main Venue Check-In' and 'Jam Session Check-In' exist (EO #8)."""
    try:
        rows = pretix_sql(
            "SELECT cl.name FROM pretixbase_checkinlist cl "
            "JOIN pretixbase_event e ON cl.event_id = e.id "
            "JOIN pretixbase_organizer o ON e.organizer_id = o.id "
            "WHERE o.slug = 'urban-music' AND e.slug = 'bklyn-jazz-symposium-2026';"
        )
        has_main = "Main Venue Check-In" in rows
        has_jam = "Jam Session Check-In" in rows
        check("8. Check-in lists", 1, has_main and has_jam,
              f"main={'found' if has_main else 'missing'}, jam={'found' if has_jam else 'missing'}")
    except Exception as e:
        check("8. Check-in lists", 1, False, f"exception: {e}")


def check_9_display_settings() -> None:
    """Display settings: primary_color=#7C3AED and front page text (EO #9)."""
    try:
        rows = pretix_sql(
            "SELECT s.key, s.value FROM pretixbase_event_setting s "
            "JOIN pretixbase_event e ON s.event_id = e.id "
            "JOIN pretixbase_organizer o ON e.organizer_id = o.id "
            "WHERE o.slug = 'urban-music' AND e.slug = 'bklyn-jazz-symposium-2026' "
            "AND s.key IN ('primary_color', 'frontpage_text');"
        )
        has_color = "#7C3AED" in rows or "#7c3aed" in rows
        has_text = "Brooklyn Jazz Symposium 2026" in rows
        check("9. Display settings", 1, has_color and has_text,
              f"color={'found' if has_color else 'missing'}, text={'found' if has_text else 'missing'}")
    except Exception as e:
        check("9. Display settings", 1, False, f"exception: {e}")


def check_10_invoice_settings() -> None:
    """Invoice settings: auto-generation enabled, prefix BJS2026- (EO #10)."""
    try:
        rows = pretix_sql(
            "SELECT s.key, s.value FROM pretixbase_event_setting s "
            "JOIN pretixbase_event e ON s.event_id = e.id "
            "JOIN pretixbase_organizer o ON e.organizer_id = o.id "
            "WHERE o.slug = 'urban-music' AND e.slug = 'bklyn-jazz-symposium-2026' "
            "AND (s.key LIKE 'invoice%');"
        )
        has_prefix = "BJS2026-" in rows
        # invoice_generate can be 'True', 'paid', 'user', etc. - any non-false means enabled
        has_auto = False
        for line in rows.split("\n"):
            if "invoice_generate" in line:
                val = line.split("|")[-1].strip().strip('"') if "|" in line else ""
                if val and val.lower() not in ("false", "no", "never", "False", ""):
                    has_auto = True
        check("10. Invoice settings", 1, has_auto and has_prefix,
              f"auto={'enabled' if has_auto else 'disabled'}, prefix={'found' if has_prefix else 'missing'}")
    except Exception as e:
        check("10. Invoice settings", 1, False, f"exception: {e}")


def check_11_customers() -> None:
    """Customer accounts Helena Vasquez and Dominic Ferrara exist (EO #12)."""
    try:
        rows = pretix_sql(
            "SELECT c.email FROM pretixbase_customer c "
            "JOIN pretixbase_organizer o ON c.organizer_id = o.id "
            "WHERE o.slug = 'urban-music' "
            "AND c.email IN ('helena.vasquez@jazzpremier.com', "
            "'dominic.ferrara@soundwavecorp.com');"
        )
        has_helena = "helena.vasquez@jazzpremier.com" in rows
        has_dominic = "dominic.ferrara@soundwavecorp.com" in rows
        check("11. Customer accounts", 1, has_helena and has_dominic,
              f"helena={'found' if has_helena else 'missing'}, dominic={'found' if has_dominic else 'missing'}")
    except Exception as e:
        check("11. Customer accounts", 1, False, f"exception: {e}")


def check_12_membership() -> None:
    """Membership type 'Urban Music VIP Patron' exists; Helena has membership 2026-03-01 to 2027-02-28 (EO #13)."""
    try:
        mt_rows = pretix_sql(
            "SELECT mt.id FROM pretixbase_membershiptype mt "
            "JOIN pretixbase_organizer o ON mt.organizer_id = o.id "
            "WHERE o.slug = 'urban-music' AND mt.name::text LIKE '%Urban Music VIP Patron%';"
        )
        if not mt_rows.strip():
            check("12. Membership", 2, False, "membership type 'Urban Music VIP Patron' not found")
            return
        m_rows = pretix_sql(
            "SELECT m.date_start::text, m.date_end::text "
            "FROM pretixbase_membership m "
            "JOIN pretixbase_membershiptype mt ON m.membership_type_id = mt.id "
            "JOIN pretixbase_customer c ON m.customer_id = c.id "
            "WHERE c.email = 'helena.vasquez@jazzpremier.com' "
            "AND mt.name::text LIKE '%Urban Music VIP Patron%';"
        )
        if not m_rows.strip():
            check("12. Membership", 2, False, "Helena's membership not found")
            return
        parts = m_rows.split("|")
        start_ok = parts[0].strip().startswith("2026-03-01") if len(parts) > 0 else False
        end_ok = parts[1].strip().startswith("2027-02-28") if len(parts) > 1 else False
        check("12. Membership", 2, start_ok and end_ok,
              f"start={parts[0].strip()}, end={parts[1].strip() if len(parts) > 1 else '?'}")
    except Exception as e:
        check("12. Membership", 2, False, f"exception: {e}")


# ── Twenty CRM checks ────────────────────────────────────────────────────────

_ws_schema: str = ""


def get_ws() -> str:
    global _ws_schema
    if not _ws_schema:
        _ws_schema = get_twenty_workspace_schema()
    return _ws_schema


def check_13_twenty_companies() -> None:
    """Four companies exist with correct domains in Twenty (EO #14)."""
    try:
        ws = get_ws()
        if not ws:
            check("13. Twenty companies", 2, False, "workspace schema not found")
            return
        rows = twenty_sql(
            f'SELECT name, "domainName"::text FROM "{ws}".company '
            f"WHERE name IN ('Jazz Premier Group', 'Soundwave Corporation', "
            f"'Rhythm House Productions', 'Blue Note Ventures');"
        )
        expected = {
            "Jazz Premier Group": "jazzpremier.com",
            "Soundwave Corporation": "soundwavecorp.com",
            "Rhythm House Productions": "rhythmhouseprod.com",
            "Blue Note Ventures": "bluenoteven.com",
        }
        issues = []
        for name, domain in expected.items():
            if name not in rows:
                issues.append(f"{name} missing")
            elif domain not in rows:
                issues.append(f"{name} domain wrong")
        check("13. Twenty companies", 2, not issues,
              "all 4 correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("13. Twenty companies", 2, False, f"exception: {e}")


def check_14_twenty_people() -> None:
    """Four people exist with correct emails and company links (EO #15)."""
    try:
        ws = get_ws()
        if not ws:
            check("14. Twenty people", 2, False, "workspace schema not found")
            return
        rows = twenty_sql(
            f'SELECT p."name"::text, p."emails"::text, p."jobTitle", c.name '
            f'FROM "{ws}".person p '
            f'LEFT JOIN "{ws}".company c ON p."companyId" = c.id '
            f"WHERE p.\"emails\"::text LIKE '%helena.vasquez@jazzpremier.com%' "
            f"OR p.\"emails\"::text LIKE '%dominic.ferrara@soundwavecorp.com%' "
            f"OR p.\"emails\"::text LIKE '%amara.diallo@rhythmhouseprod.com%' "
            f"OR p.\"emails\"::text LIKE '%stefan.kowalczyk@bluenoteven.com%';"
        )
        expected_people = [
            ("helena.vasquez@jazzpremier.com", "Helena", "Jazz Premier Group"),
            ("dominic.ferrara@soundwavecorp.com", "Dominic", "Soundwave Corporation"),
            ("amara.diallo@rhythmhouseprod.com", "Amara", "Rhythm House Productions"),
            ("stefan.kowalczyk@bluenoteven.com", "Stefan", "Blue Note Ventures"),
        ]
        issues = []
        for email, first_name, company in expected_people:
            if email not in rows:
                issues.append(f"{first_name} ({email}) missing")
            elif company not in rows:
                # Loose check — company name should appear somewhere in the result set
                issues.append(f"{first_name} not linked to {company}")
        check("14. Twenty people", 2, not issues,
              "all 4 correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("14. Twenty people", 2, False, f"exception: {e}")


def check_15_waitlist_tasks() -> None:
    """Two waitlist notification tasks with correct titles, due dates, body keywords (EO #16)."""
    try:
        ws = get_ws()
        if not ws:
            check("15. Waitlist tasks", 2, False, "workspace schema not found")
            return
        rows = twenty_sql(
            f'SELECT t.title, t."dueAt"::text, t.body '
            f'FROM "{ws}".task t '
            f"WHERE t.title LIKE '%Notify when capacity opens%Brooklyn Jazz Symposium%';"
        )
        issues = []
        if "Rhythm House Productions" not in rows:
            issues.append("Rhythm House Productions task missing")
        if "Blue Note Ventures" not in rows:
            issues.append("Blue Note Ventures task missing")
        if rows.strip() and "2026-09-15" not in rows:
            issues.append("due date not 2026-09-15")
        if rows.strip():
            for keyword in ["GROUPJAZZ40", "350"]:
                if keyword not in rows:
                    issues.append(f"body missing '{keyword}'")
        check("15. Waitlist tasks", 2, not issues,
              "both tasks correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("15. Waitlist tasks", 2, False, f"exception: {e}")


def check_16_vip_task() -> None:
    """VIP invitation task with correct title, due date, and body content (EO #17)."""
    try:
        ws = get_ws()
        if not ws:
            check("16. VIP invite task", 2, False, "workspace schema not found")
            return
        rows = twenty_sql(
            f'SELECT t.title, t."dueAt"::text, t.body '
            f'FROM "{ws}".task t '
            f"WHERE t.title LIKE '%Send VIP invitations%Brooklyn Jazz Symposium%';"
        )
        issues = []
        if not rows.strip():
            issues.append("task not found")
        else:
            if "2026-08-20" not in rows:
                issues.append("due date not 2026-08-20")
            for keyword in ["Helena Vasquez", "Dominic Ferrara", "VIPJAZZ2026",
                            "helena.vasquez@jazzpremier.com",
                            "dominic.ferrara@soundwavecorp.com"]:
                if keyword not in rows:
                    issues.append(f"body missing '{keyword}'")
        check("16. VIP invite task", 2, not issues,
              "task correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("16. VIP invite task", 2, False, f"exception: {e}")


def check_17_note() -> None:
    """Capacity & Pricing Summary note with all required content (EO #18)."""
    try:
        ws = get_ws()
        if not ws:
            check("17. Summary note", 3, False, "workspace schema not found")
            return
        rows = twenty_sql(
            f'SELECT n.title, n.body '
            f'FROM "{ws}".note n '
            f"WHERE n.title LIKE '%Brooklyn Jazz Symposium 2026%';"
        )
        issues = []
        if not rows.strip():
            issues.append("note not found")
        else:
            required = [
                "Brooklyn Jazz Symposium 2026", "2026-10-18", "350",
                "General Admission Pass", "VIP Backstage Pass", "50",
                "85", "220", "60", "VIPJAZZ2026", "GROUPJAZZ40",
                "Helena Vasquez", "Dominic Ferrara", "BJS2026-",
            ]
            for kw in required:
                if kw not in rows:
                    issues.append(f"missing '{kw}'")
        check("17. Summary note", 3, not issues,
              "note correct" if not issues else "; ".join(issues[:5]))
    except Exception as e:
        check("17. Summary note", 3, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_event_basics()
    check_2_categories()
    check_3_products()
    check_4_quotas()
    check_5_questions()
    check_6_vouchers()
    check_7_discount_rule()
    check_8_checkin_lists()
    check_9_display_settings()
    check_10_invoice_settings()
    check_11_customers()
    check_12_membership()
    check_13_twenty_companies()
    check_14_twenty_people()
    check_15_waitlist_tasks()
    check_16_vip_task()
    check_17_note()

    total = sum(w for _, w, _, _ in _checks)
    earned = sum(w for _, w, p, _ in _checks if p)
    all_pass = all(p for _, _, p, _ in _checks) and bool(_checks)
    score = (earned / total) if total else 0.0

    print(
        f"SCORE: {score:.3f}  PASS: {all_pass}  ({earned}/{total})",
        file=sys.stderr,
    )
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
