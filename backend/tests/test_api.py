"""End-to-end smoke test: boots the app in-process and exercises every endpoint
with a real generated leaf image. Run:  python tests/test_api.py
"""
import io, os, sys, numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import app  # noqa

client = TestClient(app)


def a_leaf_png():
    """Grab one real validation image if present, else a green square."""
    p = os.path.join(os.path.dirname(__file__), "..", "train", "data", "X_val.npy")
    if os.path.exists(p):
        arr = np.load(p)[0]
    else:
        arr = np.full((160, 160, 3), (60, 140, 70), dtype="uint8")
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG"); buf.seek(0)
    return buf


def main():
    h = client.get("/health").json()
    print("health:", h)
    assert h["status"] == "ok"

    d = client.get("/api/diseases").json()
    print("diseases in KB:", len(d))
    assert len(d) == 15

    if h["model_loaded"]:
        r = client.post("/api/diagnose",
                        files={"image": ("leaf.png", a_leaf_png(), "image/png")},
                        data={"lang": "hi", "rain": "true", "area_acre": "1.5"})
        assert r.status_code == 200, r.text
        j = r.json()
        print("\n/api/diagnose ->")
        print("  disease   :", j["disease"], f'({j["confidence"]:.2%})')
        print("  severity  :", j["severity"], j["severity_word"])
        print("  medicine  :", j["advisory"]["medicine"], "|", j["advisory"]["dose"])
        print("  economics :", j["economics"])
        print("  spoken(hi):", j["spoken_advice"])
        assert j["economics"]["treatment_cost"] >= 0
    else:
        print("model not loaded — skipping /diagnose (run train/train.py first)")

    # advisory-only (works without a model)
    r = client.post("/api/advisory", data={"label": "Tomato___Late_blight", "lang": "ta", "rain": "false"})
    assert r.status_code == 200, r.text
    print("\n/api/advisory (ta):", r.json()["spoken_advice"])

    # outbreak reports
    rep = client.post("/api/reports", json={"label": "Tomato___Late_blight", "crop": "tomato",
                                            "severity": 80, "lat": 15.5, "lon": 78.2, "village": "Peddur"})
    assert rep.status_code == 200, rep.text
    print("\nreport created:", rep.json()["id"], "risk=", rep.json()["risk"])
    stats = client.get("/api/reports").json()
    print("map stats:", {k: stats[k] for k in ("reports_total", "villages", "alerts")})

    print("\nALL TESTS PASSED ✅")


if __name__ == "__main__":
    main()
