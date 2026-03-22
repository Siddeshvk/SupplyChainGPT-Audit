import urllib.request
import ssl

url = "https://supplychaingpt.streamlit.app/"
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

try:
    req = urllib.request.urlopen(url, context=ctx, timeout=60)
    print(f"✅ App pinged successfully! Status: {req.status}")
except Exception as e:
    print(f"Ping sent (response: {e})")

print("✅ Keep-alive finished")
