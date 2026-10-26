"""Causal graph of storm-driven orbital decay, and the adjustment set it implies.

Treatment: storm activity (T_ap_int, or T_storm). Outcome: residual decay (Y_residual).

Main paths:
- Storm -> air density -> decay. Air density is the mediator (the way the storm
  works), so adjusting for it would remove the effect we want to measure.
- Sun UV (f107, f107_81d) -> storms (more flares and CMEs near solar maximum) and
  Sun UV -> density / model error -> residual. This is the confounding path.
- Season (day_of_year) -> storms (equinox effect) and season -> density.
- Altitude, inclination, local solar time and drag sensitivity shape how much air
  the object meets and how strongly it reacts; they also decide which objects we
  observe on which days (TLE coverage), so we adjust for them too.
- BSTAR is fitted from the same tracking data and jumps during storms: it is a
  descendant of the storm and of density. Never adjust for it.
- TLE artifacts: TLEs are fitted mean elements, refitted more often during storms,
  and on oval orbits (high ecc) the fit swings and fakes altitude drops. These
  artifacts sit on a path storm -> artifact -> measured residual that is not air
  drag. They are removed by filtering (schema.main_sample: ecc < 0.01, tle_ok = 1),
  never by putting them in a regression.
- Operator manoeuvres react to storms (active satellites only). They are a
  descendant of the treatment: the offset ratio measures them, the main analysis
  avoids them by using passive objects only.

Node names match the analysis-table columns in src/schema.py where one exists.
"""

from src.schema import CONFOUNDERS

# node -> short description
NODES = {
    "f107":             "Daily Sun UV (F10.7)",
    "f107_81d":         "81-day Sun UV",
    "day_of_year":      "Season",
    "alt_km":           "Altitude",
    "inc_deg":          "Inclination",
    "local_solar_time": "Local solar time of the orbit",
    "drag_sensitivity": "Quiet-time drag sensitivity (Cd*A/m)",
    "ecc":              "Eccentricity (orbit shape)",
    "tle_artifact":     "TLE fitting artifact (fake altitude change)",
    "is_active":        "Active satellite (operator present)",
    "T_ap_int":         "Storm activity (integrated ap)",
    "air_density":      "Neutral air density along the orbit",
    "bstar":            "TLE BSTAR",
    "manoeuvre":        "Operator manoeuvre",
    "Y_residual":       "Residual decay (observed - predicted)",
}

EDGES = [
    # Sun UV: confounder of storms and residual
    ("f107_81d", "f107"),
    ("f107", "T_ap_int"), ("f107_81d", "T_ap_int"),
    ("f107", "air_density"), ("f107_81d", "air_density"),
    ("f107", "Y_residual"), ("f107_81d", "Y_residual"),     # physics model error depends on UV
    # Season
    ("day_of_year", "T_ap_int"), ("day_of_year", "air_density"), ("day_of_year", "Y_residual"),
    # Orbit: where the object flies decides the air it meets and the model error there
    ("alt_km", "air_density"), ("inc_deg", "air_density"), ("local_solar_time", "air_density"),
    ("alt_km", "Y_residual"), ("inc_deg", "Y_residual"), ("local_solar_time", "Y_residual"),
    ("drag_sensitivity", "Y_residual"),
    # The storm effect: through density
    ("T_ap_int", "air_density"), ("air_density", "Y_residual"),
    # BSTAR: fitted from tracking, reacts to density and drag sensitivity
    ("air_density", "bstar"), ("drag_sensitivity", "bstar"), ("T_ap_int", "bstar"),
    # TLE fitting artifacts: measurement error in the outcome, worse on oval orbits
    # and during storms (more frequent refits)
    ("ecc", "tle_artifact"), ("T_ap_int", "tle_artifact"), ("tle_artifact", "Y_residual"),
    # Operators: react to storms, change the orbit
    ("T_ap_int", "manoeuvre"), ("is_active", "manoeuvre"), ("manoeuvre", "Y_residual"),
]

TREATMENT = "T_ap_int"
OUTCOME = "Y_residual"
ADJUST = list(CONFOUNDERS)
NEVER_ADJUST = ["air_density", "bstar", "manoeuvre", "tle_artifact"]


def parents(node, edges=EDGES):
    return {a for a, b in edges if b == node}


def children(node, edges=EDGES):
    return {b for a, b in edges if a == node}


def ancestors(nodes, edges=EDGES):
    """All ancestors of a set of nodes (the nodes themselves included)."""
    seen, todo = set(), list(nodes)
    while todo:
        n = todo.pop()
        if n not in seen:
            seen.add(n)
            todo.extend(parents(n, edges))
    return seen


def descendants(node, edges=EDGES):
    """All descendants of one node (itself excluded)."""
    seen, todo = set(), list(children(node, edges))
    while todo:
        n = todo.pop()
        if n not in seen:
            seen.add(n)
            todo.extend(children(n, edges))
    return seen


def is_acyclic(edges=EDGES):
    return all(a not in descendants(a, edges) for a, _ in edges)


def d_separated(x, y, z, edges=EDGES):
    """True if x and y are d-separated given the set z.

    Moral-graph test: keep only ancestors of {x, y} and z, connect ("marry") parents
    with a common child, drop arrow directions, delete z, and check whether x can
    still reach y.
    """
    z = set(z)
    keep = ancestors({x, y} | z, edges)
    sub = [(a, b) for a, b in edges if a in keep and b in keep]
    links = {n: set() for n in keep}
    for a, b in sub:
        links[a].add(b)
        links[b].add(a)
    for child in keep:
        ps = sorted(parents(child, sub))
        for i, p in enumerate(ps):
            for q in ps[i + 1:]:
                links[p].add(q)
                links[q].add(p)

    seen, todo = {x}, [x]
    while todo:
        n = todo.pop()
        if n == y:
            return False
        for m in links[n] - z - seen:
            seen.add(m)
            todo.append(m)
    return True


def satisfies_backdoor(z, treatment=TREATMENT, outcome=OUTCOME, edges=EDGES):
    """Back-door criterion: no member of z descends from the treatment, and z blocks
    every path from treatment to outcome that starts with an arrow into the treatment."""
    if set(z) & descendants(treatment, edges):
        return False
    backdoor_graph = [(a, b) for a, b in edges if a != treatment]
    return d_separated(treatment, outcome, z, backdoor_graph)


def check_adjustment(columns):
    """Raise if a control list uses a forbidden variable or breaks the back-door criterion.

    Call this in every estimator with the controls it is about to use.
    """
    bad = [c for c in columns if c in NEVER_ADJUST or c.lower().startswith(("bstar", "rho"))]
    if bad:
        raise ValueError(f"Never adjust for {bad}: they lie on or after the storm effect")
    if not satisfies_backdoor([c for c in columns if c in NODES]):
        raise ValueError(f"Controls {list(columns)} leave a back-door path open; use {ADJUST}")
    return list(columns)


def to_dot():
    """Graphviz DOT text of the graph (treatment/outcome highlighted, forbidden nodes dashed)."""
    lines = ["digraph icarus {", "  rankdir=LR;", "  node [shape=box, fontname=Helvetica];"]
    for n, label in NODES.items():
        style = ""
        if n in (TREATMENT, OUTCOME):
            style = ", style=bold"
        elif n in NEVER_ADJUST:
            style = ", style=dashed"
        lines.append(f'  {n} [label="{label}"{style}];')
    lines += [f"  {a} -> {b};" for a, b in EDGES]
    lines.append("}")
    return "\n".join(lines)
