import time
from playwright.sync_api import sync_playwright

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        print("🌐 Opening https://supplychaingpt.streamlit.app/")
        page.goto("https://supplychaingpt.streamlit.app/", timeout=90000)
        time.sleep(12)
        
        page_content = page.content().lower()
        
        if any(word in page_content for word in ["inactive", "sleep", "wake", "open it", "get this app back up"]):
            print("😴 App was sleeping — clicking wake button")
            buttons = page.get_by_role("button")
            clicked = False
            for btn in buttons.all():
                if btn.is_visible(timeout=8000):
                    try:
                        btn.click()
                        print("✅ Wake button clicked!")
                        clicked = True
                        time.sleep(45)
                        break
                    except:
                        pass
            if not clicked:
                page.locator("button").first.click()
                time.sleep(45)
        else:
            print("👍 App was already awake!")
        
        page.screenshot(path="status.png")
        browser.close()

if __name__ == "__main__":
    main()
    print("✅ Keep-alive finished successfully")
