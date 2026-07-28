"""
server.py
---------
A small FastAPI surface over the trained network.

    pip install -e ".[serve]"
    python main.py --mode train_eval      # produces data/cpts.json
    uvicorn app.server:app --reload       # or: dictm-serve

Endpoints
---------
GET  /            single self-contained HTML page
GET  /api/schema  the variables the model accepts, with their observed values
POST /api/predict {"evidence": {...}, "track": "ex_ante"} -> posterior

Inference is exact (variable elimination), so responses are deterministic and
take well under a millisecond — there is no sampling noise between two
identical requests.
"""

import os

from src.cpt_builder import CPTBuilder, _sort_key
from src.exact import VariableEliminationEngine
from src.network_structure import (
    DISPOSITION_LABELS,
    FEATURE_SETS,
    NODES,
    to_binary_disposition,
)

MODEL_PATH = os.environ.get(
    "DICTM_MODEL",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cpts.json"),
)

# Human-readable value labels for the fields the form exposes. Codes follow the
# SCDB codebook; anything not listed is rendered as its bare code.
VALUE_LABELS = {
    "issue_area": {
        1: "Criminal Procedure", 2: "Civil Rights", 3: "First Amendment",
        4: "Due Process", 5: "Privacy", 6: "Attorneys", 7: "Unions",
        8: "Economic Activity", 9: "Judicial Power", 10: "Federalism",
        11: "Interstate Relations", 12: "Federal Taxation",
        13: "Miscellaneous", 14: "Private Action",
    },
    "law_type": {
        1: "Constitution", 2: "Constitutional Amendment", 3: "Federal Statute",
        4: "Court Rules", 5: "Other", 6: "Infrequent litigation",
        8: "State or local law", 9: "No legal provision",
    },
    "lower_court_disposition": {
        1: "Stay/petition granted", 2: "Affirmed", 3: "Reversed",
        4: "Reversed and remanded", 5: "Vacated and remanded",
        6: "Affirmed and reversed in part", 7: "Affirmed and reversed in part, remanded",
        8: "Vacated", 9: "Petition denied", 10: "Certification granted",
        11: "No disposition", 12: "Other",
    },
    "decision_type": {
        1: "Opinion of the Court", 2: "Per curiam (no oral argument)",
        4: "Decree", 5: "Equally divided vote", 6: "Per curiam (oral argument)",
        7: "Judgment of the Court",
    },
    "unconstitutional": {
        1: "No declaration", 2: "Federal law unconstitutional",
        3: "State law unconstitutional", 4: "Local law unconstitutional",
    },
    "split_vote": {0: "Unanimous", 1: "One or more dissents"},
    "precedent_alteration": {0: "No", 1: "Yes"},
    "case_supplement": {0: "Normal disposition", 1: "Unusual disposition"},
}


def _load_engine():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"No trained model at {MODEL_PATH}.\n"
            "Run `python main.py --mode train_eval` first."
        )
    return VariableEliminationEngine(CPTBuilder.load(MODEL_PATH))


def build_schema(engine) -> dict:
    """Describe every input variable and the values the model has seen."""
    fields = []
    for node_name in FEATURE_SETS["explanatory"]:
        values = sorted(engine.cpt.get_values(node_name), key=_sort_key)
        labels = VALUE_LABELS.get(node_name, {})

        options = []
        for value in values:
            try:
                label = labels.get(int(value), str(value))
            except (TypeError, ValueError):
                label = str(value)
            options.append({"value": value, "label": label})

        fields.append({
            "name": node_name,
            "description": NODES[node_name].description,
            "ex_ante": node_name in FEATURE_SETS["ex_ante"],
            "options": options,
        })

    return {
        "fields": fields,
        "tracks": {k: list(v) for k, v in FEATURE_SETS.items()},
        "dispositions": {str(k): v for k, v in DISPOSITION_LABELS.items()},
    }


def predict(engine, evidence: dict, track: str = "ex_ante") -> dict:
    """Posterior over dispositions, restricted to the chosen evidence policy."""
    allowed = set(FEATURE_SETS.get(track, FEATURE_SETS["ex_ante"]))
    filtered = {k: v for k, v in evidence.items() if k in allowed and v is not None}

    posterior = engine.query("final_disposition", filtered)

    ranked = []
    for value, prob in sorted(posterior.items(), key=lambda x: -x[1]):
        try:
            label = DISPOSITION_LABELS.get(int(value), str(value))
        except (TypeError, ValueError):
            label = str(value)
        ranked.append({"value": value, "label": label, "probability": round(prob, 6)})

    affirm = sum(p for v, p in posterior.items() if to_binary_disposition(v) == 0)
    reverse = sum(p for v, p in posterior.items() if to_binary_disposition(v) == 1)
    total = affirm + reverse

    return {
        "track": track,
        "evidence_used": filtered,
        "ignored": sorted(set(evidence) - allowed),
        "distribution": ranked,
        "binary": {
            "affirmed": round(affirm / total, 6) if total else None,
            "reversed": round(reverse / total, 6) if total else None,
        },
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel

    class PredictRequest(BaseModel):
        evidence: dict = {}
        track: str = "ex_ante"

    app = FastAPI(title="dictm", description="Supreme Court disposition prediction")
    _engine = None

    def get_engine():
        global _engine
        if _engine is None:
            _engine = _load_engine()
        return _engine

    @app.get("/api/schema")
    def api_schema():
        return build_schema(get_engine())

    @app.post("/api/predict")
    def api_predict(request: PredictRequest):
        if request.track not in FEATURE_SETS:
            raise HTTPException(400, f"Unknown track '{request.track}'")
        return predict(get_engine(), request.evidence, request.track)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

except ImportError:  # FastAPI is an optional extra
    app = None


def run():
    """Console-script entry point: `dictm-serve`."""
    import uvicorn

    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=False)


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>dictm — Supreme Court disposition prediction</title>
<style>
  :root {
    --bg: #ffffff; --fg: #1a1a1a; --muted: #666; --line: #e0e0e0;
    --accent: #2563eb; --bar: #93c5fd; --bar-top: #2563eb;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14161a; --fg: #e8e8e8; --muted: #9aa0a6; --line: #2c2f36;
      --accent: #60a5fa; --bar: #1e3a8a; --bar-top: #60a5fa;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 2rem 1rem; background: var(--bg); color: var(--fg);
    font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  main { max-width: 940px; margin: 0 auto; }
  h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
  .sub { color: var(--muted); margin: 0 0 1.5rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 1rem; }
  label { display: block; font-weight: 600; font-size: .8rem; margin-bottom: .3rem; }
  select {
    width: 100%; padding: .5rem; border: 1px solid var(--line);
    border-radius: 6px; background: var(--bg); color: var(--fg); font-size: .9rem;
  }
  fieldset { border: 1px solid var(--line); border-radius: 8px; padding: 1rem; margin: 0 0 1rem; }
  legend { font-weight: 600; font-size: .85rem; padding: 0 .4rem; }
  .note { color: var(--muted); font-size: .8rem; margin: .4rem 0 1rem; }
  button {
    background: var(--accent); color: #fff; border: 0; border-radius: 6px;
    padding: .6rem 1.2rem; font-size: .95rem; font-weight: 600; cursor: pointer;
  }
  table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
  th, td { text-align: left; padding: .4rem .5rem; border-bottom: 1px solid var(--line); font-size: .88rem; }
  th { color: var(--muted); font-weight: 600; font-size: .78rem; text-transform: uppercase; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
  .bar { height: 9px; border-radius: 4px; background: var(--bar); min-width: 2px; }
  tr:first-child .bar { background: var(--bar-top); }
  .wrap { overflow-x: auto; }
</style>
</head>
<body>
<main>
  <h1>Supreme Court disposition prediction</h1>
  <p class="sub">Exact posterior inference over a Bayesian Network trained on the Supreme Court Database.</p>

  <fieldset>
    <legend>Evidence policy</legend>
    <select id="track">
      <option value="ex_ante">Ex ante — only pre-decision information</option>
      <option value="explanatory">Explanatory — includes variables recorded from the decision</option>
    </select>
    <p class="note" id="trackNote"></p>
  </fieldset>

  <form id="form">
    <fieldset>
      <legend>Case</legend>
      <div class="grid" id="fields"></div>
    </fieldset>
    <button type="submit">Predict</button>
  </form>

  <div id="output"></div>
</main>

<script>
const NOTES = {
  ex_ante: "Only variables knowable before the Court rules. This is a genuine forecast \\u2014 and on this dataset it barely improves on the base rate.",
  explanatory: "Includes decision type, dissent, unconstitutionality and precedent alteration, all recorded from the ruling itself. Descriptive, not predictive."
};

let SCHEMA = null;

function renderFields() {
  const track = document.getElementById("track").value;
  const allowed = new Set(SCHEMA.tracks[track]);
  document.getElementById("trackNote").textContent = NOTES[track];

  document.getElementById("fields").innerHTML = SCHEMA.fields
    .filter(f => allowed.has(f.name))
    .map(f => {
      const opts = ['<option value="">(unknown)</option>']
        .concat(f.options.map(o =>
          `<option value="${encodeURIComponent(JSON.stringify(o.value))}">${o.label}</option>`))
        .join("");
      const pretty = f.name.replace(/_/g, " ");
      return `<div><label for="f_${f.name}">${pretty}</label>
              <select id="f_${f.name}" data-node="${f.name}">${opts}</select></div>`;
    }).join("");
}

function collectEvidence() {
  const evidence = {};
  document.querySelectorAll("#fields select").forEach(sel => {
    if (sel.value) evidence[sel.dataset.node] = JSON.parse(decodeURIComponent(sel.value));
  });
  return evidence;
}

function renderResult(data) {
  const max = Math.max(...data.distribution.map(d => d.probability), 1e-9);
  const rows = data.distribution.map(d => `
    <tr>
      <td>${d.label}</td>
      <td class="num">${(d.probability * 100).toFixed(1)}%</td>
      <td style="width:45%"><div class="bar" style="width:${(d.probability / max * 100).toFixed(1)}%"></div></td>
    </tr>`).join("");

  const b = data.binary;
  const binaryLine = (b.affirmed === null) ? "" :
    `<p class="note">Collapsed to the binary task: affirmed <strong>${(b.affirmed*100).toFixed(1)}%</strong>,
     reversed or vacated <strong>${(b.reversed*100).toFixed(1)}%</strong>.</p>`;

  const used = Object.keys(data.evidence_used).length;
  document.getElementById("output").innerHTML = `
    <h2 style="font-size:1.1rem;margin-top:1.75rem">Posterior</h2>
    <p class="note">${used} evidence variable${used === 1 ? "" : "s"} supplied; the rest were marginalised out.</p>
    ${binaryLine}
    <div class="wrap"><table>
      <thead><tr><th>Disposition</th><th class="num">P</th><th></th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>`;
}

document.getElementById("track").addEventListener("change", renderFields);

document.getElementById("form").addEventListener("submit", async e => {
  e.preventDefault();
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({evidence: collectEvidence(), track: document.getElementById("track").value})
  });
  if (!res.ok) {
    document.getElementById("output").innerHTML = `<p class="note">Request failed (${res.status}).</p>`;
    return;
  }
  renderResult(await res.json());
});

fetch("/api/schema").then(r => r.json()).then(schema => {
  SCHEMA = schema;
  renderFields();
});
</script>
</body>
</html>
"""
