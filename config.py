import os

class Settings:
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me")
    BASE_URL = os.getenv("BASE_URL", "http://localhost:5000")

    # Stripe
    STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
    STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
    STRIPE_PRICE_ID_FREE = os.getenv("STRIPE_PRICE_ID_FREE", "")
    STRIPE_PRICE_ID_VIP_MONTHLY = os.getenv("STRIPE_PRICE_ID_VIP_MONTHLY", "")
    STRIPE_PRICE_ID_VIP_YEARLY = os.getenv("STRIPE_PRICE_ID_VIP_YEARLY", "")
    STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    STRIPE_PORTAL_RETURN_URL = os.getenv("STRIPE_PORTAL_RETURN_URL", "http://localhost:5012/account")

    # Mailgun
    MAILGUN_API_KEY = os.getenv("MAILGUN_API_KEY", "")
    MAILGUN_DOMAIN = os.getenv("MAILGUN_DOMAIN", "")
    MAIL_FROM = os.getenv("MAIL_FROM", f"Pick3 App <mail@{os.getenv('MAILGUN_DOMAIN','example.com')}>")
    
    # Admin
    ADMIN_EMAILS = os.getenv("ADMIN_EMAILS", "")


