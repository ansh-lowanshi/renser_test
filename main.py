# main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Try multiple possible module names for the GenAI SDK
genai = None
_import_error = None
for mod in ("google.generativeai", "google_genai", "google.genai", "google.genai.v1"):
    try:
        # importlib to import dynamic name
        import importlib
        genai = importlib.import_module(mod)
        loaded_mod = mod
        break
    except Exception as e:
        _import_error = e
        genai = None

if genai is None:
    raise RuntimeError(
        "Could not import a Google GenAI SDK module. Tried: "
        "'google.generativeai', 'google_genai', 'google.genai', 'google.genai.v1'.\n"
        f"Last error: {_import_error}\n\n"
        "Make sure your requirements.txt lists the correct package (google-genai) "
        "and that the package version supports your Python runtime. "
        "Locally you can test `python -c \"import google.generativeai; print('ok')\"` "
        "or check `pip show google-genai`."
    )

# If the SDK expects configure() on a submodule, adapt:
# Common API: google.generativeai.configure(...)
if hasattr(genai, "configure"):
    configure = genai.configure
else:
    # Some package variants expose genai.configure differently; try aliasing
    try:
        configure = getattr(genai, "genai", None) or getattr(genai, "configure", None)
    except Exception:
        configure = None

# Configure using env var if present (safe for render)
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY and configure:
    try:
        configure(api_key=API_KEY)
    except Exception:
        # some modules expect different configure signatures — ignore here
        pass

# Attempt to create a model object in a defensive way
Model = None
for candidate in ("GenerativeModel", "GenModel", "Model"):
    Model = getattr(genai, candidate, None)
    if Model:
        break

# If no class found, keep Model None and use a lower-level call later
app = FastAPI(title="Gemini AI Agent (diagnostic)")

class UserRequest(BaseModel):
    message: str

@app.get("/debug")
def debug():
    return {
        "imported_module": loaded_mod,
        "has_configure": bool(configure),
        "model_class_found": bool(Model),
        "python_version": os.sys.version,
    }

@app.post("/agent")
def agent_endpoint(req: UserRequest):
    try:
        # prefer high-level Model.generate_content if available
        if Model:
            model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            model = Model(model_name)
            # many SDKs offer generate_content or generate
            if hasattr(model, "generate_content"):
                resp = model.generate_content(req.message)
                text = getattr(resp, "text", str(resp))
                return {"reply": text}
            elif hasattr(model, "generate"):
                resp = model.generate(req.message)
                return {"reply": getattr(resp, "text", str(resp))}
        # fallback: try top-level generate_content
        if hasattr(genai, "generate_content"):
            resp = genai.generate_content(req.message)
            return {"reply": getattr(resp, "text", str(resp))}
        # last fallback: attempt a generic API call (may fail)
        raise RuntimeError("No supported generate API found in imported GenAI module.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
