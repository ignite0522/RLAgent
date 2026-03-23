"""使用无头浏览器打开 URL，执行 JS，探测是否出现 alert 弹窗且内容为 success。"""
from langchain_core.tools import tool


@tool
def check_xss(url: str, wait_ms: int = 3000):
    """
    在真实浏览器中打开 URL，执行页面 JavaScript，探测是否弹出 alert 且内容为 success。
    - url: 完整地址（可带 XSS payload 的查询参数或 hash），例如 https://example.com/page?q=<script>alert('success')</script>
    - wait_ms: 打开页面后等待多少毫秒再判定（用于等待延迟触发的 alert），默认 3000。
    返回 JSON：alert_detected 为 true 表示探测到内容为 success 的 alert；message 为实际弹窗文字；error 为异常信息；summary 为简短说明。
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        err = "未安装 playwright，请执行: pip install playwright && playwright install chromium"
        return {
            "alert_detected": False,
            "message": None,
            "error": err,
            "summary": err,
        }

    captured_message: list = []  # 用 list 以便在闭包中修改

    def on_dialog(dialog):
        captured_message.append(dialog.message)
        dialog.accept()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                ignore_https_errors=True,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )
            page = context.new_page()
            page.on("dialog", on_dialog)
            page.goto(url, wait_until="domcontentloaded", timeout=15000)
            page.wait_for_timeout(wait_ms)
            browser.close()

        msg = captured_message[0] if captured_message else None
        ok = msg is not None and msg.strip() == "success"
        summary = "检测到弹窗，内容为 success" if ok else ("未检测到弹窗" if not captured_message else f"检测到弹窗但内容为: {msg!r}，非 success")
        return {
            "alert_detected": ok,
            "message": msg,
            "error": None,
            "summary": summary,
        }
    except Exception as e:
        err = str(e)
        return {
            "alert_detected": False,
            "message": None,
            "error": err,
            "summary": f"执行异常: {err}",
        }
