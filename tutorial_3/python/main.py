import json
import pytest
import psycopg
from pathlib import Path

DB_URI = "postgresql://gis:secret@localhost:5433/gis_qa"

def load_rules():
    rules = []
    rules_dir = Path(__file__).parent / "test" / "rules"
    if not rules_dir.exists():
        return rules
    for file_path in rules_dir.glob("*.json"):
        with open(file_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                rules.extend(data)
            else:
                rules.append(data)
    return rules

RULES = load_rules()

@pytest.fixture(scope="session")
def db_conn():
    conn = psycopg.connect(DB_URI)
    yield conn
    conn.close()

@pytest.mark.parametrize("rule", RULES, ids=lambda r: r["id"])
def test_spatial_rule(db_conn, rule):
    with db_conn.cursor() as cur:
        cur.execute(rule["query"])
        violations = cur.fetchall()

    if violations:
        formatted_violations = "\n".join(
            [f"  - Record ID: {v[0]} | Reason: {v[1]}" for v in violations]
        )
        pytest.fail(
            f"\n[{rule['severity']}] {rule['id']}: {rule['description']}\n"
            f"Found {len(violations)} violation(s):\n"
            f"{formatted_violations}"
        )
