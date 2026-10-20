"""Build site/deadlines.html: paper submission deadlines of the venues this site covers, for one year from CHECKED.

Every date was read on CHECKED from the official page linked next to it. Editions that had announced
nothing show the previous edition's deadline, labelled as such; nothing on the page is estimated.
To update: edit VENUES / CLOSED / JOURNALS, set CHECKED, run this script, copy site/deadlines.html to the repository.
"""
import datetime as dt
import html
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SITE = "ML & CV Topic Trends"
TITLE = "ML & CV Deadlines"
SCOPE = "the six venues this site covers (NeurIPS, ICML, ICLR, CVPR, ICCV and ECCV)"
CHECKED = dt.date(2026, 10, 20)
UNTIL = dt.date(2027, 10, 20)
OFFSET = {"AoE": -12, "HST": -10, "UTC": 0, "PST": -8}  # hours from UTC

# steps: (what, date, time or None, time zone or None); a venue without steps has not announced its deadline
VENUES = [
    dict(name="CVPR 2027",
         steps=[("Paper enrollment", "2026-11-10", "23:59", "AoE"),
                ("Full paper", "2026-11-16", "23:59", "AoE"),
                ("Supplementary material", "2026-11-23", None, "AoE")],
         remark="The call: “Date is fixed, no extension will be given.” All co-authors need an up-to-date OpenReview "
                "profile.",
         conf=("20–25 June 2027", "Seattle, WA, USA"),
         links=[("Call for papers", "https://cvpr.thecvf.com/Conferences/2027/CallForPapers"),
                ("Conference site", "https://cvpr.thecvf.com/Conferences/2027")]),
    dict(name="ICML 2027", steps=[], order="2027-01-28",
         prev="ICML 2026: abstract 23 January 2026, full paper 28 January 2026 (AoE).",
         conf=("Not announced", "South America (city not announced)"),
         links=[("Future meetings", "https://icml.cc/Conferences/FutureMeetings"),
                ("ICML 2026 dates", "https://icml.cc/Conferences/2026/Dates")]),
    dict(name="ICCV 2027", steps=[], order="2027-03-07",
         prev="ICCV 2025: paper registration 3 March 2025, full paper 7 March 2025, 23:59 HST.",
         conf=("3–7 October 2027", "Hong Kong (HKCEC)"),
         links=[("ICCV 2027 dates", "https://iccv.thecvf.com/Conferences/2027/Dates"),
                ("Conference site", "https://iccv.thecvf.com/Conferences/2027"),
                ("ICCV 2025 dates", "https://iccv.thecvf.com/Conferences/2025/Dates")]),
    dict(name="NeurIPS 2027", steps=[], order="2027-05-06",
         prev="NeurIPS 2026: abstract 4 May 2026, full paper 6 May 2026 (AoE).",
         conf=("Not announced", "Europe (city not announced)"),
         links=[("Future meetings", "https://neurips.cc/Conferences/FutureMeetings"),
                ("NeurIPS 2026 dates", "https://neurips.cc/Conferences/2026/Dates")]),
    dict(name="ICLR 2028", steps=[], order="2027-09-25",
         prev="ICLR 2027: abstract 18 September 2026, full paper 25 September 2026 (AoE).",
         conf=("Not announced", ""),
         links=[("ICLR 2027 dates", "https://iclr.cc/Conferences/2027/Dates"),
                ("Future meetings", "https://iclr.cc/Conferences/FutureMeetings")]),
]
CLOSED = [("ICLR 2027 (main conference 26–28 April 2027) closed on 25 September 2026, AoE.",
           "https://iclr.cc/Conferences/2027/Dates")]
LATER = ("ECCV meets in even years. ECCV 2028 has announced nothing; ECCV 2026 closed on 5 March 2026, 22:00 UTC "
         "(eccv.ecva.net/Conferences/2026/Dates). CVPR 2028 has announced nothing either; CVPR 2027 closes in "
         "November 2026. If they keep those months, both deadlines fall after 20 October 2027.")
JOURNALS = []

E = html.escape


def day(s):
    d = dt.date.fromisoformat(s)
    return f"{d.day} {d:%B %Y}"


def time_tag(date, time, tz):
    if tz is None:  # no time zone published: date only, countdown in calendar days
        return (f'<span class="when"><time datetime="{date}" data-date="{date}">{day(date)}</time> '
                f'<span class="tz">time zone not stated</span></span>')
    hh, mm = map(int, (time or "23:59").split(":"))
    local = dt.datetime.fromisoformat(date).replace(hour=hh, minute=mm)
    utc = local - dt.timedelta(hours=OFFSET[tz])
    shown = f"{day(date)}, {time} {tz}" if time else f"{day(date)}, {tz} (no time given)"
    return f'<span class="when"><time datetime="{utc:%Y-%m-%dT%H:%MZ}" data-utc="{utc:%Y-%m-%dT%H:%M:00Z}">{E(shown)}</time></span>'


def first_date(v):
    return v["steps"][0][1] if v["steps"] else v["order"]


for v in VENUES:  # every announced step must fall inside the page window or after CHECKED
    for _, d, _, _ in v["steps"]:
        assert CHECKED <= dt.date.fromisoformat(d), (v["name"], d)
    assert dt.date.fromisoformat(first_date(v)) <= UNTIL, v["name"]

rows = []
for v in sorted(VENUES, key=first_date):
    if v["steps"]:
        items = "".join(f'<li><span class="what">{E(w)}</span> {time_tag(d, t, z)}'
                        f'<span class="left" aria-live="off"></span><span class="local"></span></li>'
                        for w, d, t, z in v["steps"])
        ddl = f'<ul class="steps">{items}</ul>'
        if v.get("remark"):
            ddl += f'<p class="remark">{E(v["remark"])}</p>'
    else:
        ddl = f'<p class="tba">Not announced yet</p><p class="remark">Previous edition, for orientation only: {E(v["prev"])}</p>'
    when, where = v["conf"]
    conf = E(when) + (f'<span class="where">{E(where)}</span>' if where else "")
    links = "".join(f'<li><a href="{E(u)}" target="_blank" rel="noopener">{E(t)}</a></li>' for t, u in v["links"])
    cls = "" if v["steps"] else ' class="tbarow"'
    rows.append(f'<tr{cls}><th scope="row" data-label="Venue">{E(v["name"])}</th>'
                f'<td data-label="Deadlines">{ddl}</td><td data-label="Conference">{conf}</td>'
                f'<td data-label="Official pages"><ul class="links">{links}</ul></td></tr>')

closed = "".join(f'<li>{E(t)} <a href="{E(u)}" target="_blank" rel="noopener">Official notice</a></li>' for t, u in CLOSED)
journals = "".join(f'<li><strong>{E(a)}</strong> <span class="muted">{E(n)}</span> · '
                   f'<a href="{E(u)}" target="_blank" rel="noopener">{E(t)}</a></li>' for a, n, t, u in JOURNALS)
journal_block = (f'<h2>Journals</h2><p class="note">No fixed deadline: regular papers can be submitted at any time. '
                 f'Special issues set their own deadlines on each journal’s site.</p><ul class="journals">{journals}</ul>'
                 if JOURNALS else "")

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="Paper submission deadlines of {E(SCOPE)}, {day(CHECKED.isoformat())} to {day(UNTIL.isoformat())}, each with its official page.">
<title>{E(TITLE)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
/* Same tokens as index.html; one table of venues (cards on phones), then notes and journals */
:root {{
  --bg: #f3f5f4; --surface: #ffffff; --fg: #16191c; --muted: #59616a; --line: #dbe0de;
  --accent: #4a3aa7; --accent-soft: #ece9f9; --accent-ink: #ffffff;
  --rise: #2a78d6; --fall: #d63c3b; --flat: #6f6e6a; --bar: #c9c3ee;
  --font-display: "IBM Plex Sans Condensed", "Arial Narrow", "Helvetica Neue", sans-serif;
  --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-data: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
  --radius: 6px;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #121518; --surface: #1a1e22; --fg: #e6e8ea; --muted: #9aa3ab; --line: #2c3238;
    --accent: #9a8ff0; --accent-soft: #292545; --accent-ink: #121518;
    --rise: #4a95ec; --fall: #ec6f6e; --flat: #959ba1; --bar: #4a4380; color-scheme: dark;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #121518; --surface: #1a1e22; --fg: #e6e8ea; --muted: #9aa3ab; --line: #2c3238;
  --accent: #9a8ff0; --accent-soft: #292545; --accent-ink: #121518;
  --rise: #4a95ec; --fall: #ec6f6e; --flat: #959ba1; --bar: #4a4380; color-scheme: dark;
}}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.5 var(--font-body); margin: 0; }}
.wrap {{ max-width: 1320px; margin: 0 auto; padding-inline: 20px; padding-block: 0 48px; }}
a {{ color: var(--accent); }}
a:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
header.top {{ padding-block: 28px 18px; display: grid; gap: 6px; }}
.back {{ font-size: 0.9rem; text-decoration: none; }}
h1 {{ font: 600 2rem/1.1 var(--font-display); letter-spacing: 0.01em; margin: 0; text-wrap: balance; }}
h2 {{ font: 600 1.25rem/1.2 var(--font-display); margin: 28px 0 8px; }}
.lede {{ color: var(--muted); max-width: 75ch; margin: 0; }}
.note {{ font-size: 0.85rem; color: var(--muted); margin: 0; max-width: 90ch; }}
.muted {{ color: var(--muted); }}
.next {{ margin: 6px 0 16px; padding: 10px 14px; background: var(--accent-soft); border-radius: var(--radius); font-weight: 500; }}
.next:empty {{ display: none; }}
.tablewrap {{ background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; }}
th, td {{ padding: 12px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
thead th {{ font: 600 0.75rem var(--font-body); text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
tbody th {{ font: 600 1.05rem/1.25 var(--font-display); white-space: nowrap; }}
tbody tr:last-child > * {{ border-bottom: 0; }}
tr.tbarow th {{ color: var(--muted); }}
ul.steps, ul.links, ul.journals, ul.closed {{ list-style: none; margin: 0; padding: 0; }}
ul.steps li {{ display: grid; grid-template-columns: minmax(10rem, 15rem) 1fr; column-gap: 12px; padding-block: 2px; }}
ul.steps li.past {{ color: var(--muted); text-decoration: line-through; }}
.what {{ color: var(--muted); }}
time {{ font-family: var(--font-data); font-variant-numeric: tabular-nums; font-size: 0.88rem; }}
.when {{ grid-column: 2; }}
.tz {{ font-size: 0.8rem; color: var(--muted); white-space: nowrap; }}
.left {{ grid-column: 2; font-size: 0.82rem; font-weight: 600; color: var(--accent); }}
.left:empty, .local:empty {{ display: none; }}
.local {{ grid-column: 2; font-size: 0.78rem; color: var(--muted); }}
.remark {{ font-size: 0.82rem; color: var(--muted); margin: 6px 0 0; max-width: 60ch; }}
.tba {{ margin: 0; font-weight: 600; }}
.where {{ display: block; color: var(--muted); font-size: 0.85rem; }}
ul.links li + li {{ margin-top: 2px; }}
ul.journals li, ul.closed li {{ padding-block: 4px; }}
footer {{ margin-top: 32px; }}
@media (max-width: 760px) {{
  .wrap {{ padding-inline: 16px; }}
  thead {{ display: none; }}
  table, tbody, tr, th, td {{ display: block; width: 100%; }}
  tbody tr {{ border-bottom: 1px solid var(--line); padding: 12px 14px; }}
  tbody tr:last-child {{ border-bottom: 0; }}
  th, td {{ border: 0; padding: 4px 0; }}
  td::before {{ content: attr(data-label); display: block; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
  ul.steps li {{ grid-template-columns: 1fr; }}
  .when, .left, .local {{ grid-column: 1; }}
}}
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <a class="back" href="./">← {E(SITE)}</a>
    <h1>Submission deadlines</h1>
    <p class="lede">Paper deadlines of {E(SCOPE)}, from {day(CHECKED.isoformat())} to {day(UNTIL.isoformat())}. Every date was read on the official page linked in its row on {day(CHECKED.isoformat())}.</p>
    <p class="note">Deadlines can move: check the official page before submitting. Where an edition has announced nothing yet, the row shows the previous edition’s deadline for orientation only; it is not a forecast. AoE = Anywhere on Earth (UTC−12): the deadline has passed everywhere once it passes there. Countdowns and “your time” use this browser’s clock and time zone.</p>
  </header>
  <p class="next" id="next"></p>
  <h2>Conferences</h2>
  <div class="tablewrap">
  <table>
    <thead><tr><th scope="col">Venue</th><th scope="col">Deadlines</th><th scope="col">Conference</th><th scope="col">Official pages</th></tr></thead>
    <tbody>
    {"".join(rows)}
    </tbody>
  </table>
  </div>
  {'<p class="note" style="margin-top:10px">' + E(LATER) + '</p>' if LATER else ''}
  <h2>Closed shortly before {day(CHECKED.isoformat())}</h2>
  <ul class="closed">{closed}</ul>
  {journal_block}
  <footer><p class="note">Checked {day(CHECKED.isoformat())}. Built by <code>scripts/{os.path.basename(__file__)}</code>.</p></footer>
</div>
<script>
(() => {{
  const now = new Date();
  const fmt = (d) => d.toLocaleString(undefined, {{ weekday: "short", day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" }});
  const left = (ms) => ms < 0 ? "closed" : ms < 864e5 ? Math.floor(ms / 36e5) + " h left" : Math.floor(ms / 864e5) + " days left";
  let next = null;
  for (const t of document.querySelectorAll("time[data-utc]")) {{
    const d = new Date(t.dataset.utc), ms = d - now, li = t.closest("li");
    li.querySelector(".left").textContent = left(ms);
    if (ms < 0) li.classList.add("past");
    else {{
      li.querySelector(".local").textContent = "your time: " + fmt(d);
      if (!next || ms < next.ms) next = {{ ms, li }};
    }}
  }}
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  for (const t of document.querySelectorAll("time[data-date]")) {{
    const [y, m, dd] = t.dataset.date.split("-").map(Number);
    const days = Math.round((new Date(y, m - 1, dd) - today) / 864e5), li = t.closest("li");
    li.querySelector(".left").textContent = days < 0 ? "closed" : "about " + days + " days left";
    if (days < 0) li.classList.add("past");
    else if (!next || days * 864e5 < next.ms) next = {{ ms: days * 864e5, li }};
  }}
  if (next) {{
    const venue = next.li.closest("tr").querySelector("th").textContent;
    const what = next.li.querySelector(".what").textContent;
    document.getElementById("next").textContent = "Next: " + venue + ", " + what.toLowerCase() + " — " + next.li.querySelector(".left").textContent + ".";
  }}
}})();
</script>
</body>
</html>
"""
os.makedirs(os.path.join(ROOT, "site"), exist_ok=True)
open(os.path.join(ROOT, "site", "deadlines.html"), "w").write(page)
print("site/deadlines.html", len(page), "bytes;", len(VENUES), "venues")
