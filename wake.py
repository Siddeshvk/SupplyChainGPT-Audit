import urllib.request
import sys

URL = "https://supplychaingpt.streamlit.app/"

try:
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        status = response.getcode()
        print(f"✅ Pinged {URL} — HTTP {status}")
except Exception as e:
    print(f"⚠️ Could not reach {URL}: {e}")
    # Exit 0 so the GitHub Action doesn't fail — app may just be waking up
    sys.exit(0)
