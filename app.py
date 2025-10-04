import os
import time
import datetime
import stripe
import requests
from flask import Flask, jsonify, request, render_template, redirect, session, abort, url_for
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from functools import wraps
from config import Settings
from db import init_db, upsert_user, get_user, add_picks, get_picks_for_date, delete_picks_for_date

# Pick 3 System imports
try:
    from vendor.pick3_system.universal_lottery_analyzer import EnhancedLearningHybridSystem as UniversalLotteryAnalyzer
    PICK3_SYSTEM_AVAILABLE = True
except ImportError:
    PICK3_SYSTEM_AVAILABLE = False
    print("Warning: Pick 3 System not available. Copy your system files to vendor/pick3_system/")

def create_app():
    app = Flask(__name__)
    app.config.from_object(Settings)
    app.secret_key = app.config["SECRET_KEY"]

    init_db()  # ensure SQLite table exists
    stripe.api_key = app.config["STRIPE_SECRET_KEY"]
    
    # Magic link authentication
    signer = URLSafeTimedSerializer(app.secret_key)
    MAGIC_LINK_TTL = 15 * 60  # 15 minutes
    
    # Admin allowlist
    ALLOWED_ADMINS = [e.strip().lower() for e in (app.config.get("ADMIN_EMAILS", "").split(",")) if e.strip()]

    # ---------- helpers ----------
    def is_vip(email):
        u = get_user(email)
        if not u:
            return False
        if u["status"] not in ("active", "trialing"):
            return False
        return u["plan"] in ("vip_monthly", "vip_yearly")

    def require_login():
        if "user_email" not in session:
            # send them to home or a sign-in page (Phase 2 can add magic-link login)
            return redirect(url_for("index"))
        return None

    def require_vip(f):
        # decorator to guard VIP features
        from functools import wraps
        @wraps(f)
        def wrapper(*args, **kwargs):
            if "user_email" not in session:
                return redirect(url_for("index"))
            if not is_vip(session["user_email"]):
                return abort(402)  # Payment Required (or redirect to pricing)
            return f(*args, **kwargs)
        return wrapper

    def require_admin(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if "user_email" not in session: 
                return redirect(url_for("login_get"))
            if session["user_email"].lower() not in ALLOWED_ADMINS:
                return abort(403)
            return f(*args, **kwargs)
        return wrapper

    # ---------- routes ----------
    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.post("/email/test")
    def test_email():
        """Test endpoint to verify Mailgun is working"""
        data = request.get_json()
        to_email = data.get("to") if data else "test@example.com"
        
        try:
            response = send_email(
                app, 
                to=to_email,
                subject="Mailgun Test - Pick3 App",
                text="This is a test email from your Pick3 app. Mailgun is working correctly!"
            )
            return jsonify({
                "status": "success", 
                "message": "Test email sent successfully",
                "response_status": response.status_code
            })
        except Exception as e:
            return jsonify({
                "status": "error", 
                "message": f"Failed to send test email: {str(e)}"
            }), 500

    @app.get("/")
    def index():
        email = session.get("user_email")
        user = get_user(email) if email else None
        return render_template(
            "index.html",
            publishable_key=app.config["STRIPE_PUBLISHABLE_KEY"],
            user=user
        )

    # Choose plan: free | vip_monthly | vip_yearly
    @app.post("/create-checkout-session/<plan>")
    def create_checkout_session(plan):
        price_map = {
            "free": app.config.get("STRIPE_PRICE_ID_FREE"),
            "vip_monthly": app.config.get("STRIPE_PRICE_ID_VIP_MONTHLY"),
            "vip_yearly": app.config.get("STRIPE_PRICE_ID_VIP_YEARLY"),
        }
        price_id = price_map.get(plan)
        if not price_id:
            return jsonify({"error": "Invalid plan"}), 400

        try:
            session_obj = stripe.checkout.Session.create(
                mode="subscription",
                payment_method_types=["card"] if plan != "free" else ["card"],  # card still okay, but free won't charge
                line_items=[{"price": price_id, "quantity": 1}],
                success_url=f"{app.config['BASE_URL']}/success?session_id={{CHECKOUT_SESSION_ID}}",
                cancel_url=f"{app.config['BASE_URL']}/cancel",
            )
            return jsonify({"id": session_obj.id})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    # After Stripe Checkout returns
    @app.get("/success")
    def success():
        csid = request.args.get("session_id")
        if not csid:
            return "Missing session_id", 400

        try:
            cs = stripe.checkout.Session.retrieve(csid, expand=["customer", "subscription", "line_items"])
        except Exception as e:
            return f"Error retrieving checkout session: {e}", 400

        # Extract email & customer
        email = (cs.get("customer_details") or {}).get("email") or cs.get("customer_email")
        customer = cs.get("customer")
        customer_id = customer.id if hasattr(customer, 'id') else customer
        sub = cs.get("subscription")

        # Determine plan label from price used
        plan = "free"
        status = "active"
        current_period_end = None
        if sub:
            # pull live state from subscription
            sub_obj = stripe.Subscription.retrieve(sub) if isinstance(sub, str) else sub
            status = sub_obj.get("status", "active")
            current_period_end = sub_obj.get("current_period_end")
            # read first item price to map plan name
            try:
                price_id = sub_obj["items"]["data"][0]["price"]["id"]
            except Exception:
                price_id = None
            if price_id == app.config.get("STRIPE_PRICE_ID_VIP_MONTHLY"):
                plan = "vip_monthly"
            elif price_id == app.config.get("STRIPE_PRICE_ID_VIP_YEARLY"):
                plan = "vip_yearly"
            else:
                plan = "free"

        # Persist / update user record
        if email:
            upsert_user(
                email=email,
                stripe_customer_id=customer_id,
                plan=plan,
                status=status,
                current_period_end=current_period_end
            )
            # "Log them in" for this browser
            session["user_email"] = email

        return render_template("success.html")

    @app.get("/cancel")
    def cancel():
        return render_template("cancel.html")

    @app.get("/account")
    def account():
        if "user_email" not in session:
            return redirect(url_for("index"))
        user = get_user(session["user_email"])
        return render_template("account.html", user=user)

    # Stripe Customer Portal to manage/cancel
    @app.post("/create-portal-session")
    def create_portal_session():
        if "user_email" not in session:
            return jsonify({"error": "Not logged in"}), 401
        user = get_user(session["user_email"])
        if not user or not user["stripe_customer_id"]:
            return jsonify({"error": "No Stripe customer"}), 400
        return_url = app.config.get("STRIPE_PORTAL_RETURN_URL") or f"{app.config['BASE_URL']}/account"
        ps = stripe.billing_portal.Session.create(
            customer=user["stripe_customer_id"],
            return_url=return_url,
        )
        return jsonify({"url": ps.url})

    # Example VIP-only API (SmartScore, Singles)
    @app.get("/api/v1/smartscore")
    @require_vip
    def smartscore_api():
        return jsonify({"ok": True, "message": "VIP access granted", "data": {"score": 0.92}})

    # ---------- Admin routes for posting picks ----------
    @app.get("/admin/picks")
    @require_admin
    def admin_picks_get():
        today = datetime.date.today().isoformat()
        return render_template("admin_picks.html", today=today)

    @app.post("/admin/picks")
    @require_admin
    def admin_picks_post():
        # form fields we'll paste from a textarea
        play_date = (request.form.get("play_date") or "").strip()          # YYYY-MM-DD
        game = "pick3"  # fixed for now
        # We accept four buckets to keep it simple:
        free_doubles = (request.form.get("free_doubles") or "").strip()
        vip_doubles  = (request.form.get("vip_doubles") or "").strip()
        vip_singles  = (request.form.get("vip_singles") or "").strip()

        if not play_date:
            return "Missing play_date", 400

        # normalize spacing and commas
        def norm(s): 
            return ", ".join([x.strip() for x in s.replace("\n", ",").split(",") if x.strip()])

        now = int(time.time())
        if free_doubles:
            add_picks(play_date, game, "double", "free", norm(free_doubles), now)
        if vip_doubles:
            add_picks(play_date, game, "double", "vip", norm(vip_doubles), now)
        if vip_singles:
            add_picks(play_date, game, "single", "vip", norm(vip_singles), now)

        return redirect(url_for("admin_picks_get"))

    @app.get("/admin/picks/today")
    @require_admin
    def admin_picks_today():
        today = datetime.date.today().isoformat()
        data = get_picks_for_date(today)  # all tiers
        return {"date": today, "picks": data}

    @app.post("/admin/picks/delete")
    @require_admin
    def admin_picks_delete():
        play_date = (request.form.get("play_date") or "").strip()
        if not play_date:
            return "Missing play_date", 400
        delete_picks_for_date(play_date)
        return redirect(url_for("admin_picks_get"))

    # ---------- Generate picks using Pick 3 System ----------
    @app.get("/admin/generate")
    @require_admin
    def admin_generate_get():
        today = datetime.date.today().isoformat()
        return render_template("admin_generate.html", today=today, preview=None, error=None, system_available=PICK3_SYSTEM_AVAILABLE)

    @app.post("/admin/generate")
    @require_admin
    def admin_generate_post():
        if not PICK3_SYSTEM_AVAILABLE:
            return render_template("admin_generate.html", 
                                 today=datetime.date.today().isoformat(), 
                                 preview=None, 
                                 error="Pick 3 System not available. Copy your system files to vendor/pick3_system/",
                                 system_available=False)

        play_date = (request.form.get("play_date") or "").strip()
        state     = (request.form.get("state") or "Generic").strip()
        game_type = (request.form.get("game_type") or "Pick3").strip()

        try:
            # 1) init the analyzer (from your system)
            analyzer = UniversalLotteryAnalyzer(state_name=state, game_type=game_type)

            # 2) get doubles/singles using your system's methods
            # Get the last drawing for prediction
            if hasattr(analyzer, 'df') and not analyzer.df.empty:
                last_draw = analyzer.df.iloc[-1]['numbers']
            else:
                # Fallback to a sample number if no data
                last_draw = "123"
            
            # Generate predictions using your system
            if hasattr(analyzer, 'generate_enhanced_predictions'):
                # Use the main prediction method
                predictions = analyzer.generate_enhanced_predictions(last_draw, num_singles=20, num_doubles=20)
                singles = predictions.get('singles', [])[:20]
                doubles = predictions.get('doubles', [])[:20]
            else:
                # Fallback to individual methods
                if hasattr(analyzer, 'generate_doubles'):
                    doubles = analyzer.generate_doubles(last_draw, 20)
                else:
                    doubles = []
                
                if hasattr(analyzer, 'generate_singles'):
                    singles = analyzer.generate_singles(last_draw, 20)
                else:
                    singles = []

            # 3) make pretty CSV strings
            def to_csv(seq): 
                return ", ".join(map(str, seq)) if seq else ""

            preview = {
                "free_doubles": to_csv(doubles[:10]),   # show fewer for free
                "vip_doubles":  to_csv(doubles),        # full list for VIP
                "vip_singles":  to_csv(singles),
            }

            # "Save" clicked?
            if request.form.get("action") == "save":
                now = int(time.time())
                if preview["free_doubles"]:
                    add_picks(play_date, "pick3", "double", "free", preview["free_doubles"], now)
                if preview["vip_doubles"]:
                    add_picks(play_date, "pick3", "double", "vip",  preview["vip_doubles"],  now)
                if preview["vip_singles"]:
                    add_picks(play_date, "pick3", "single", "vip",  preview["vip_singles"],  now)
                return redirect(url_for("admin_picks_get"))

            return render_template("admin_generate.html", 
                                 today=play_date, 
                                 state=state, 
                                 game_type=game_type, 
                                 preview=preview, 
                                 error=None,
                                 system_available=True)

        except Exception as e:
            return render_template("admin_generate.html", 
                                 today=play_date, 
                                 state=state, 
                                 game_type=game_type, 
                                 preview=None, 
                                 error=str(e),
                                 system_available=True)

    # ---------- Display picks to users ----------
    @app.get("/picks/today")
    def picks_today():
        today = datetime.date.today().isoformat()
        email = session.get("user_email")
        user = get_user(email) if email else None
        is_vip_user = bool(user and user.get("status") in ("active","trialing") and user.get("plan") in ("vip_monthly","vip_yearly"))

        free = get_picks_for_date(today, tier="free")
        vip  = get_picks_for_date(today, tier="vip") if is_vip_user else []

        return render_template("picks_today.html", is_vip=is_vip_user, free=free, vip=vip, today=today)

    @app.get("/picks/<play_date>")
    def picks_by_date(play_date):
        email = session.get("user_email")
        user = get_user(email) if email else None
        is_vip_user = bool(user and user.get("status") in ("active","trialing")
                           and user.get("plan") in ("vip_monthly","vip_yearly"))
        free = get_picks_for_date(play_date, tier="free")
        vip  = get_picks_for_date(play_date, tier="vip") if is_vip_user else []
        return render_template("picks_today.html", is_vip=is_vip_user, free=free, vip=vip, today=play_date)

    # ---------- USER PREDICTION SYSTEM ----------
    
    @app.get("/predict")
    def predict_get():
        """User-facing prediction generation page"""
        email = session.get("user_email")
        if not email:
            return redirect(url_for("login_get"))
        
        user = get_user(email)
        is_vip_user = bool(user and user.get("status") in ("active","trialing") and user.get("plan") in ("vip_monthly","vip_yearly"))
        
        return render_template("predict.html", is_vip=is_vip_user, error=None, predictions=None)
    
    @app.post("/predict")
    def predict_post():
        """Generate predictions for the user"""
        email = session.get("user_email")
        if not email:
            return redirect(url_for("login_get"))
        
        user = get_user(email)
        is_vip_user = bool(user and user.get("status") in ("active","trialing") and user.get("plan") in ("vip_monthly","vip_yearly"))
        
        if not PICK3_SYSTEM_AVAILABLE:
            return render_template("predict.html", is_vip=is_vip_user, error="Prediction system temporarily unavailable", predictions=None)
        
        try:
            # Get form parameters
            state = (request.form.get("state") or "New Jersey").strip()
            game_type = (request.form.get("game_type") or "Pick3").strip()
            num_doubles = int(request.form.get("num_doubles", 10))
            num_singles = int(request.form.get("num_singles", 10)) if is_vip_user else 0
            
            # Initialize the analyzer
            analyzer = UniversalLotteryAnalyzer(state_name=state, game_type=game_type)
            
            # Get last drawing for context
            last_draw = analyzer.df.iloc[-1]['numbers'] if not analyzer.df.empty else "000"
            
            # Generate predictions
            predictions = {
                "state": state,
                "game_type": game_type,
                "last_draw": last_draw,
                "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "doubles": [],
                "singles": []
            }
            
            # Generate doubles (available to all users)
            if hasattr(analyzer, 'generate_enhanced_predictions'):
                doubles_result = analyzer.generate_enhanced_predictions(last_draw=last_draw, num_predictions=num_doubles)
                if doubles_result and 'doubles' in doubles_result:
                    predictions["doubles"] = doubles_result['doubles'][:num_doubles]
                else:
                    # Fallback to generate_doubles if available
                    if hasattr(analyzer, 'generate_doubles'):
                        predictions["doubles"] = analyzer.generate_doubles(num_doubles)
            elif hasattr(analyzer, 'generate_doubles'):
                predictions["doubles"] = analyzer.generate_doubles(num_doubles)
            
            # Generate singles (VIP only)
            if is_vip_user and num_singles > 0:
                if hasattr(analyzer, 'generate_enhanced_predictions'):
                    singles_result = analyzer.generate_enhanced_predictions(last_draw=last_draw, num_predictions=num_singles)
                    if singles_result and 'singles' in singles_result:
                        predictions["singles"] = singles_result['singles'][:num_singles]
                    else:
                        # Fallback to generate_singles if available
                        if hasattr(analyzer, 'generate_singles'):
                            predictions["singles"] = analyzer.generate_singles(num_singles)
                elif hasattr(analyzer, 'generate_singles'):
                    predictions["singles"] = analyzer.generate_singles(num_singles)
            
            return render_template("predict.html", is_vip=is_vip_user, error=None, predictions=predictions)
            
        except Exception as e:
            return render_template("predict.html", is_vip=is_vip_user, error=f"Error generating predictions: {str(e)}", predictions=None)

    # ---------- Magic Link Authentication ----------
    @app.get("/login")
    def login_get():
        return render_template("login.html")

    @app.post("/login")
    def login_post():
        email = (request.form.get("email") or "").strip().lower()
        if not email:
            return "Email required", 400

        # Create signed token with email
        token = signer.dumps({"email": email})
        link = f"{app.config['BASE_URL']}/auth/verify?token={token}"

        # Send email via Mailgun
        subject = "Your Pick3 App sign-in link"
        text = f"Click to sign in:\n\n{link}\n\nThis link expires in 15 minutes."
        try:
            send_email(app, to=email, subject=subject, text=text)
        except Exception as e:
            return f"Email error: {e}", 500

        return render_template("check_email.html", email=email)

    @app.get("/auth/verify")
    def auth_verify():
        token = request.args.get("token", "")
        if not token:
            return "Missing token", 400
        try:
            data = signer.loads(token, max_age=MAGIC_LINK_TTL)
        except SignatureExpired:
            return "This link has expired. Please request a new one.", 400
        except BadSignature:
            return "Invalid link.", 400

        email = (data.get("email") or "").strip().lower()
        if not email:
            return "Invalid link.", 400

        # Ensure user exists; if new, create with plan=None (or 'free' if you want)
        # You can also choose to auto-create as free plan here.
        upsert_user(email=email)

        # Log them in
        session["user_email"] = email

        # Optional: redirect VIPs to VIP page, others to pricing
        u = get_user(email)
        if u and u.get("plan") in ("vip_monthly", "vip_yearly") and u.get("status") in ("active","trialing"):
            return redirect(url_for("account"))
        return redirect(url_for("index"))

    # Temporary bypass for testing - REMOVE THIS IN PRODUCTION
    @app.get("/admin/login")
    def admin_login_bypass():
        email = "i4cprofitsllc@gmail.com"
        upsert_user(email=email)
        session["user_email"] = email
        return redirect(url_for("admin_picks_get"))
    
    # Another bypass route
    @app.get("/quick-admin")
    def quick_admin():
        email = "i4cprofitsllc@gmail.com"
        upsert_user(email=email)
        session["user_email"] = email
        return redirect(url_for("admin_picks_get"))

    @app.get("/logout")
    def logout():
        session.pop("user_email", None)
        return redirect(url_for("index"))

    # ---------- webhook: keep DB in sync ----------
    @app.post("/webhook")
    def webhook():
        endpoint_secret = app.config["STRIPE_WEBHOOK_SECRET"]
        payload = request.data
        sig_header = request.headers.get("Stripe-Signature", "")

        if not endpoint_secret:
            # In dev, you can return 200 to avoid noise; in prod use the real secret
            return "Webhook not configured", 200

        try:
            event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
        except Exception as e:
            return str(e), 400

        t = event["type"]
        obj = event["data"]["object"]

        # Map subscription events to user table
        if t in ("customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted", "invoice.payment_succeeded"):
            sub = obj
            customer_id = sub.get("customer")
            status = sub.get("status")
            current_period_end = sub.get("current_period_end")
            # Get the email from the customer object
            try:
                cust = stripe.Customer.retrieve(customer_id)
                email = cust.get("email")
            except Exception:
                email = None

            # Determine plan
            price_id = None
            try:
                price_id = sub["items"]["data"][0]["price"]["id"]
            except Exception:
                pass

            plan = None
            if price_id == app.config.get("STRIPE_PRICE_ID_VIP_MONTHLY"):
                plan = "vip_monthly"
            elif price_id == app.config.get("STRIPE_PRICE_ID_VIP_YEARLY"):
                plan = "vip_yearly"
            # If canceled, keep plan label but mark status canceled
            if email:
                upsert_user(
                    email=email,
                    stripe_customer_id=customer_id,
                    plan=plan,
                    status=status,
                    current_period_end=current_period_end
                )

        elif t == "checkout.session.completed":
            # backup: handle free plan or missing sub
            email = (obj.get("customer_details") or {}).get("email") or obj.get("customer_email")
            customer = obj.get("customer")
            customer_id = customer.id if hasattr(customer, 'id') else customer
            if email:
                upsert_user(email=email, stripe_customer_id=customer_id)

        return "ok", 200

    return app

def send_email(app, to, subject, text):
    return requests.post(
        f"https://api.mailgun.net/v3/{app.config['MAILGUN_DOMAIN']}/messages",
        auth=("api", app.config["MAILGUN_API_KEY"]),
        data={
            "from": app.config["MAIL_FROM"],
            "to": [to],
            "subject": subject,
            "text": text
        },
        timeout=20
    )

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)