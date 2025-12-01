

import base64

def image_to_data_url(path, mime="image/png"):
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"