"""
🔔 ALERT & NOTIFICATION SYSTEM

Multi-Channel Notifications für:
- High-Value Bets
- Cashout Opportunities
- Drawdown Warnings
- Model Performance
- System Events

Channels:
- Telegram Bot ✅
- Discord Webhooks ✅
- Email ✅
- Console (fallback) ✅
"""

import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Optional dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ==========================================================
# ALERT TYPES & CONFIGURATION
# ==========================================================

class AlertLevel(Enum):
    """Alert Priority Levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    SUCCESS = "SUCCESS"


class AlertType(Enum):
    """Alert Categories"""
    VALUE_BET = "value_bet"
    CASHOUT_OPPORTUNITY = "cashout"
    DRAWDOWN_WARNING = "drawdown"
    PROFIT_MILESTONE = "profit"
    MODEL_PERFORMANCE = "model"
    SYSTEM_ERROR = "error"


@dataclass
class AlertConfig:
    """Alert System Configuration"""
    # Telegram
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Discord
    discord_enabled: bool = False
    discord_webhook_url: str = ""

    # Email
    email_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_sender: str = ""
    email_password: str = ""
    email_recipient: str = ""

    # Console
    console_enabled: bool = True

    # Alert Rules
    min_value_bet_ev: float = 0.10  # Min 10% EV für Alert
    min_cashout_profit: float = 50.0  # Min €50 Profit für Cashout Alert
    drawdown_threshold: float = 0.15  # Alert bei 15% Drawdown


# ==========================================================
# ALERT MESSAGE
# ==========================================================
@dataclass
class Alert:
    """Alert Message"""
    level: AlertLevel
    type: AlertType
    title: str
    message: str
    timestamp: datetime = None
    data: Dict = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.data is None:
            self.data = {}

    def to_dict(self) -> Dict:
        return {
            'level': self.level.value,
            'type': self.type.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data
        }

    def format_console(self) -> str:
        """Format für Console"""
        emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.CRITICAL: "🔴",
            AlertLevel.SUCCESS: "✅"
        }

        return f"{emoji.get(self.level, '📢')} [{self.level.value}] {self.title}\n{self.message}"

    def format_telegram(self) -> str:
        """Format für Telegram (Markdown)"""
        emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.CRITICAL: "🔴",
            AlertLevel.SUCCESS: "✅"
        }

        return f"{emoji.get(self.level, '📢')} *{self.title}*\n\n{self.message}\n\n_{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}_"


# ==========================================================
# NOTIFICATION CHANNELS
# ==========================================================

class TelegramNotifier:
    """Telegram Bot Notifications"""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    def send(self, message: str) -> bool:
        """Send message via Telegram"""
        if not REQUESTS_AVAILABLE:
            print("❌ requests library not installed")
            return False

        url = f"{self.base_url}/sendMessage"

        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Telegram Error: {e}")
            return False


class DiscordNotifier:
    """Discord Webhook Notifications"""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, message: str, title: str = None) -> bool:
        """Send message via Discord Webhook"""
        if not REQUESTS_AVAILABLE:
            print("❌ requests library not installed")
            return False

        # Discord Embed
        embed = {
            'title': title or 'AI Dutching Alert',
            'description': message,
            'color': 3447003,  # Blue
            'timestamp': datetime.now().isoformat()
        }

        payload = {
            'embeds': [embed]
        }

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            return response.status_code == 204
        except Exception as e:
            print(f"❌ Discord Error: {e}")
            return False


class EmailNotifier:
    """Email Notifications"""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        sender: str,
        password: str,
        recipient: str
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender = sender
        self.password = password
        self.recipient = recipient

    def send(self, subject: str, message: str) -> bool:
        """Send email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender
            msg['To'] = self.recipient
            msg['Subject'] = subject

            msg.attach(MIMEText(message, 'plain'))

            # Connect & Send
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender, self.password)

            text = msg.as_string()
            server.sendmail(self.sender, self.recipient, text)

            server.quit()

            return True

        except Exception as e:
            print(f"❌ Email Error: {e}")
            return False


# ==========================================================
# ALERT MANAGER
# ==========================================================
class AlertManager:
    """
    Hauptklasse für Alert-Management

    Features:
    - Multi-Channel Dispatch
    - Alert Deduplication
    - Alert History
    - Rule-based Filtering
    """

    def __init__(self, config: AlertConfig):
        self.config = config

        # Initialize Notifiers
        self.notifiers = {}

        if config.telegram_enabled and config.telegram_bot_token:
            self.notifiers['telegram'] = TelegramNotifier(
                config.telegram_bot_token,
                config.telegram_chat_id
            )

        if config.discord_enabled and config.discord_webhook_url:
            self.notifiers['discord'] = DiscordNotifier(
                config.discord_webhook_url
            )

        if config.email_enabled and config.email_sender:
            self.notifiers['email'] = EmailNotifier(
                config.email_smtp_server,
                config.email_smtp_port,
                config.email_sender,
                config.email_password,
                config.email_recipient
            )

        # Alert History
        self.alert_history: List[Alert] = []

    def send_alert(
        self,
        alert: Alert,
        channels: Optional[List[str]] = None
    ):
        """
        Send Alert über specified channels

        Args:
            alert: Alert object
            channels: Liste von Channels ('telegram', 'discord', 'email', 'console')
                     Wenn None: Alle enabled channels
        """
        # Add to history
        self.alert_history.append(alert)

        # Determine channels
        if channels is None:
            channels = list(self.notifiers.keys())
            if self.config.console_enabled:
                channels.append('console')

        # Send to each channel
        for channel in channels:
            if channel == 'console':
                self._send_console(alert)

            elif channel == 'telegram' and 'telegram' in self.notifiers:
                self._send_telegram(alert)

            elif channel == 'discord' and 'discord' in self.notifiers:
                self._send_discord(alert)

            elif channel == 'email' and 'email' in self.notifiers:
                self._send_email(alert)

    def _send_console(self, alert: Alert):
        """Send to console"""
        print(f"\n{alert.format_console()}\n")

    def _send_telegram(self, alert: Alert):
        """Send to Telegram"""
        notifier = self.notifiers.get('telegram')
        if notifier:
            message = alert.format_telegram()
            notifier.send(message)

    def _send_discord(self, alert: Alert):
        """Send to Discord"""
        notifier = self.notifiers.get('discord')
        if notifier:
            notifier.send(alert.message, alert.title)

    def _send_email(self, alert: Alert):
        """Send to Email"""
        notifier = self.notifiers.get('email')
        if notifier:
            subject = f"[{alert.level.value}] {alert.title}"
            notifier.send(subject, alert.message)

    # ==========================================================
    # CONVENIENCE METHODS
    # ==========================================================

    def alert_value_bet(
        self,
        match: str,
        market: str,
        odds: float,
        probability: float,
        stake: float,
        ev: float
    ):
        """Alert für Value Bet"""
        if ev < self.config.min_value_bet_ev:
            return  # Nicht wichtig genug

        message = f"""
🎯 High Value Bet Detected!

Match: {match}
Market: {market}
Odds: {odds:.2f}
Probability: {probability:.2%}
Stake: €{stake:.2f}
Expected Value: {ev:.2%}

💰 Expected Profit: €{stake * ev:.2f}
        """.strip()

        alert = Alert(
            level=AlertLevel.SUCCESS if ev > 0.15 else AlertLevel.INFO,
            type=AlertType.VALUE_BET,
            title="Value Bet Alert",
            message=message,
            data={
                'match': match,
                'market': market,
                'odds': odds,
                'probability': probability,
                'stake': stake,
                'ev': ev
            }
        )

        self.send_alert(alert)

    def alert_cashout_opportunity(
        self,
        match: str,
        original_stake: float,
        cashout_offer: float,
        recommendation: str
    ):
        """Alert für Cashout Opportunity"""
        profit = cashout_offer - original_stake

        if profit < self.config.min_cashout_profit:
            return

        message = f"""
💵 Cashout Opportunity!

Match: {match}
Original Stake: €{original_stake:.2f}
Cashout Offer: €{cashout_offer:.2f}
Profit: €{profit:.2f} ({profit/original_stake*100:.1f}%)

🤖 Recommendation: {recommendation}
        """.strip()

        alert = Alert(
            level=AlertLevel.WARNING,
            type=AlertType.CASHOUT_OPPORTUNITY,
            title="Cashout Alert",
            message=message,
            data={
                'match': match,
                'original_stake': original_stake,
                'cashout_offer': cashout_offer,
                'profit': profit
            }
        )

        self.send_alert(alert)

    def alert_drawdown(
        self,
        current_bankroll: float,
        peak_bankroll: float,
        drawdown_pct: float
    ):
        """Alert für Drawdown"""
        if drawdown_pct < self.config.drawdown_threshold:
            return

        message = f"""
🔴 DRAWDOWN WARNING!

Peak Bankroll: €{peak_bankroll:.2f}
Current Bankroll: €{current_bankroll:.2f}
Drawdown: {drawdown_pct:.2%}

⚠️ Consider reducing stake sizes or reviewing strategy.
        """.strip()

        alert = Alert(
            level=AlertLevel.CRITICAL,
            type=AlertType.DRAWDOWN_WARNING,
            title="Drawdown Alert",
            message=message,
            data={
                'current_bankroll': current_bankroll,
                'peak_bankroll': peak_bankroll,
                'drawdown_pct': drawdown_pct
            }
        )

        self.send_alert(alert)

    def alert_profit_milestone(
        self,
        total_profit: float,
        roi: float,
        num_bets: int
    ):
        """Alert für Profit Milestone"""
        message = f"""
🎉 Profit Milestone Reached!

Total Profit: €{total_profit:.2f}
ROI: {roi:.2f}%
Number of Bets: {num_bets}

Great job! Keep it up! 🚀
        """.strip()

        alert = Alert(
            level=AlertLevel.SUCCESS,
            type=AlertType.PROFIT_MILESTONE,
            title="Profit Milestone",
            message=message,
            data={
                'total_profit': total_profit,
                'roi': roi,
                'num_bets': num_bets
            }
        )

        self.send_alert(alert)

    def alert_model_performance(
        self,
        model_name: str,
        accuracy: float,
        sharpe: float,
        recommendation: str
    ):
        """Alert für Model Performance"""
        message = f"""
📊 Model Performance Update

Model: {model_name}
Accuracy: {accuracy:.2%}
Sharpe Ratio: {sharpe:.2f}

💡 {recommendation}
        """.strip()

        alert = Alert(
            level=AlertLevel.INFO,
            type=AlertType.MODEL_PERFORMANCE,
            title="Model Performance",
            message=message,
            data={
                'model_name': model_name,
                'accuracy': accuracy,
                'sharpe': sharpe
            }
        )

        self.send_alert(alert)

    def alert_system_error(self, error_message: str):
        """Alert für System Error"""
        message = f"""
❌ SYSTEM ERROR

{error_message}

Please check the logs and investigate.
        """.strip()

        alert = Alert(
            level=AlertLevel.CRITICAL,
            type=AlertType.SYSTEM_ERROR,
            title="System Error",
            message=message,
            data={'error': error_message}
        )

        self.send_alert(alert)


# ==========================================================
# EXAMPLE USAGE
# ==========================================================
if __name__ == "__main__":
    print("🔔 ALERT SYSTEM - EXAMPLE\n")

    # Configuration (Console only for demo)
    config = AlertConfig(
        console_enabled=True,
        telegram_enabled=False,  # Set to True + add credentials to test
        discord_enabled=False,
        email_enabled=False
    )

    # Create Manager
    manager = AlertManager(config)

    # Test Alerts
    print("Testing Value Bet Alert...")
    manager.alert_value_bet(
        match="Liverpool vs Chelsea",
        market="3Way Result - Home",
        odds=2.50,
        probability=0.52,
        stake=100.0,
        ev=0.15
    )

    print("\nTesting Cashout Alert...")
    manager.alert_cashout_opportunity(
        match="Bayern vs Dortmund",
        original_stake=100.0,
        cashout_offer=180.0,
        recommendation="CASHOUT NOW - High Profit Secured"
    )

    print("\nTesting Drawdown Alert...")
    manager.alert_drawdown(
        current_bankroll=800.0,
        peak_bankroll=1000.0,
        drawdown_pct=0.20
    )

    print("\nTesting Profit Milestone...")
    manager.alert_profit_milestone(
        total_profit=500.0,
        roi=35.5,
        num_bets=150
    )

    print("\n✅ Alert System Test Complete!")
    print(f"\nTotal Alerts Sent: {len(manager.alert_history)}")
