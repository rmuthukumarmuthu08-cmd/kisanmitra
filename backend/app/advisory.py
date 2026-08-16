"""Advisory brain: disease knowledge base + economic engine + spoken advice.

This is the 'decision, not just diagnosis' layer. Given a predicted disease,
crop stage/severity and the local weather, it returns treatment, safety,
a rupee cost-benefit and a spoken recommendation in the farmer's language.
"""
import json, os

_DATA = os.path.join(os.path.dirname(__file__), "data", "diseases.json")
with open(_DATA, encoding="utf-8") as f:
    KB = json.load(f)
CROPS = KB["crops"]
DISEASES = KB["diseases"]

CROP_NAME_LOCAL = {
    "tomato": {"hi": "टमाटर", "ta": "தக்காளி", "te": "టమాటా", "mr": "टोमॅटो"},
    "potato": {"hi": "आलू", "ta": "உருளைக்கிழங்கு", "te": "బంగాళాదుంప", "mr": "बटाटा"},
    "pepper": {"hi": "मिर्च", "ta": "மிளகாய்", "te": "మిరప", "mr": "मिरची"},
}


def severity_word(sev: int) -> str:
    return "Severe" if sev > 68 else "Moderate" if sev > 45 else "Mild"


def economics(entry, crop_key, severity, area_acre=1.0):
    crop = CROPS[crop_key]
    yield_value = crop["yield_kg_per_acre"] * crop["price_per_kg"]
    save = round(yield_value * entry["loss_factor"] * (severity / 100.0) * area_acre)
    cost = 0 if entry["healthy"] else round((340 + severity * 3) * area_acre)
    return {
        "crop_value_at_risk": save,
        "treatment_cost": cost,
        "net_benefit": save - cost,
        "yield_value_per_acre": yield_value,
        "currency": "INR",
    }


def recommendation(entry, econ, rain: bool) -> str:
    if entry["healthy"]:
        return "No disease detected — your crop looks healthy. Keep scouting weekly."
    if rain:
        return (f"Rain is expected — spray this evening so it soaks in first, or wait "
                f"until it clears. Acting protects about ₹{econ['crop_value_at_risk']:,} of your crop.")
    return (f"Weather is dry — spray today for best effect. Acting now protects about "
            f"₹{econ['crop_value_at_risk']:,} of your harvest.")


def local_disease_name(entry, lang):
    return entry.get("name_local", {}).get(lang, entry["name"])


def spoken_advice(entry, crop_key, econ, rain, lang):
    dname = local_disease_name(entry, lang)
    cost = econ["treatment_cost"]
    save = econ["crop_value_at_risk"]
    crop_local = CROP_NAME_LOCAL.get(crop_key, {}).get(lang, CROPS[crop_key]["label"])
    if entry["healthy"]:
        msgs = {
            "en": f"Good news. Your {CROPS[crop_key]['label']} looks healthy. No spray needed. Keep checking every week.",
            "hi": f"अच्छी खबर। आपकी {crop_local} की फसल स्वस्थ है। छिड़काव की जरूरत नहीं। हर हफ्ते जांच करते रहें।",
            "ta": f"நல்ல செய்தி. உங்கள் {crop_local} ஆரோக்கியமாக உள்ளது. மருந்து தேவையில்லை. வாரம்தோறும் பரிசோதிக்கவும்.",
            "te": f"శుభవార్త. మీ {crop_local} ఆరోగ్యంగా ఉంది. మందు అవసరం లేదు. ప్రతి వారం తనిఖీ చేయండి.",
            "mr": f"चांगली बातमी. तुमचे {crop_local} निरोगी आहे. फवारणीची गरज नाही. दर आठवड्याला तपासा.",
        }
        return msgs.get(lang, msgs["en"])
    timing = {
        "en": "Spray this evening before the rain." if rain else "Spray today while it is dry.",
        "hi": "बारिश से पहले आज शाम छिड़काव करें।" if rain else "आज छिड़काव करें।",
        "ta": "மழைக்கு முன் இன்று மாலை தெளிக்கவும்." if rain else "இன்று தெளிக்கவும்.",
        "te": "వర్షానికి ముందు ఈ సాయంత్రం పిచికారీ చేయండి." if rain else "ఈరోజు పిచికారీ చేయండి.",
        "mr": "पावसाआधी आज संध्याकाळी फवारा." if rain else "आज फवारा.",
    }
    t = timing.get(lang, timing["en"])
    if lang == "hi":
        return f"आपकी {crop_local} की फसल में {dname} है। {entry['medicine']} का {entry['dose']} छिड़काव करें। {t} इसकी लागत लगभग {cost} रुपये है और यह करीब {save} रुपये की फसल बचाता है।"
    if lang == "ta":
        return f"உங்கள் {crop_local} பயிரில் {dname} உள்ளது. {entry['medicine']} ஐ {entry['dose']} தெளிக்கவும். {t} இதன் செலவு சுமார் {cost} ரூபாய், சுமார் {save} ரூபாய் பயிரை சேமிக்கும்."
    if lang == "te":
        return f"మీ {crop_local} పంటలో {dname} ఉంది. {entry['medicine']} ను {entry['dose']} పిచికారీ చేయండి. {t} దీని ఖర్చు సుమారు {cost} రూపాయలు, సుమారు {save} రూపాయల పంటను ఆదా చేస్తుంది."
    if lang == "mr":
        return f"तुमच्या {crop_local} पिकात {dname} आहे. {entry['medicine']} चा {entry['dose']} फवारा. {t} याचा खर्च सुमारे {cost} रुपये आहे आणि सुमारे {save} रुपयांचे पीक वाचते."
    return f"Your {CROPS[crop_key]['label']} has {entry['name']}, {severity_word(entry['base_severity']).lower()}. Spray {entry['medicine']}, {entry['dose']}. {t} It costs about {cost} rupees and saves around {save} rupees of your crop."


def safety_text(entry):
    if entry["healthy"]:
        return "No chemical needed. Keep monitoring."
    return (f"Do not harvest for {entry['phi_days']} days after the last spray. "
            f"Wear a mask and gloves. Never exceed the dose — extra chemical harms the crop, soil and your health.")


def build_advisory(label, severity=None, rain=False, lang="en", area_acre=1.0):
    entry = DISEASES[label]
    crop_key = entry["crop"]
    sev = int(severity if severity is not None else entry["base_severity"])
    econ = economics(entry, crop_key, sev, area_acre)
    return {
        "label": label,
        "disease": entry["name"],
        "disease_local": local_disease_name(entry, lang) if lang != "en" else None,
        "crop": crop_key,
        "healthy": entry["healthy"],
        "severity": sev,
        "severity_word": severity_word(sev),
        "advisory": {
            "medicine": entry["medicine"],
            "dose": entry["dose"],
            "frequency": entry["frequency"],
            "pre_harvest_interval_days": entry["phi_days"],
            "safety": safety_text(entry),
            "note": entry["note"],
        },
        "economics": econ,
        "recommendation": recommendation(entry, econ, rain),
        "spoken_advice": spoken_advice(entry, crop_key, econ, rain, lang),
        "language": lang,
    }
