"""
Keeps the SupplyChainGPT Streamlit app awake.

A plain HTTP request does NOT work: Streamlit Community Cloud returns HTTP 200
for a static HTML shell while the real Python app stays asleep. We must open
the page in a real browser so the WebSocket connection starts and, if the app
is sleeping, click the "Yes, get this app back up!" button.
"""

import sys
from playwright.sync_api import sync_playwright

URL = "https://supplychaingpt.streamlit.app/"


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(URL, wait_until="domcontentloaded", timeout=120_000)
            page.wait_for_timeout(8_000)

            wake_button = page.get_by_role(
                "button", name="Yes, get this app back up!"
            )
            if wake_button.count() > 0:
                print("App was asleep -> clicking the wake button.")
                wake_button.click()
                page.wait_for_timeout(90_000)  # let the container cold-start
            else:
                print("App was already awake.")

            page.wait_for_timeout(5_000)  # register WebSocket activity
            print(f"Done. Final URL: {page.url}")
            return 0
        except Exception as exc:
            # Never fail the workflow over a flaky load.
            print(f"Warning: could not fully load {URL}: {exc}")
            return 0
        finally:
            browser.close()


if __name__ == "__main__":
    sys.exit(main())
