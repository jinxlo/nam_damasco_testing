import logging
import os
import requests

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_ALERT_CHAT_ID = os.getenv("TELEGRAM_ALERT_CHAT_ID")

API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage" if TELEGRAM_BOT_TOKEN else None


def send_telegram_message(message: str) -> None:
    """Send a plain text message to the configured Telegram chat."""
    if not API_URL or not TELEGRAM_ALERT_CHAT_ID:
        return
    try:
        requests.post(API_URL, data={"chat_id": TELEGRAM_ALERT_CHAT_ID, "text": message})
    except Exception:
        pass


class TelegramAlertHandler(logging.Handler):
    """Logging handler that sends records to a Telegram chat."""

    def emit(self, record: logging.LogRecord) -> None:
        if not API_URL or not TELEGRAM_ALERT_CHAT_ID:
            return
        try:
            log_entry = self.format(record)
            requests.post(API_URL, data={"chat_id": TELEGRAM_ALERT_CHAT_ID, "text": log_entry})
        except Exception:
            self.handleError(record)
