import os
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from app.config import settings

SUPPORT_EMAIL = "genverse.ai@fgilservices.com"

# Path to the GenVerse logo PNG for CID embedding in emails
_LOGO_PATH = os.path.join(os.path.dirname(__file__), '..', 'static', 'logo-email.png')

# Reusable inline logo HTML — references CID-embedded image (works in Gmail, Outlook, etc.)
LOGO_HTML = '''<img src="cid:genverse_logo" alt="G" width="48" height="48" style="display:block;margin:0 auto 10px;border-radius:12px;" />'''


def _attach_logo(msg: MIMEMultipart) -> None:
    """Attach the GenVerse logo as a CID-embedded image to an email message."""
    try:
        with open(_LOGO_PATH, 'rb') as f:
            logo_data = f.read()
        logo_img = MIMEImage(logo_data, _subtype='png')
        logo_img.add_header('Content-ID', '<genverse_logo>')
        logo_img.add_header('Content-Disposition', 'inline', filename='logo.png')
        msg.attach(logo_img)
    except FileNotFoundError:
        pass  # Gracefully skip if logo file is missing


def send_verification_otp(to_email: str, otp: str) -> bool:
    """Send a 6-digit OTP to verify the user's email during signup."""
    print(f"[EmailService] send_verification_otp called for {to_email}", flush=True)

    if not settings.MAIL_USERNAME or not settings.MAIL_PASSWORD:
        print(f"[EmailService] SKIPPED: SMTP credentials not configured!", flush=True)
        return False

    html_body = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Verify your email</title>
    </head>
    <body style="margin:0;padding:0;font-family:Arial,sans-serif;background:#f4f4f5;">
        <div style="max-width:480px;margin:40px auto;background:white;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
            <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);padding:32px;text-align:center;">
                {LOGO_HTML}
                <h1 style="color:white;margin:0;font-size:24px;">{settings.MAIL_FROM_NAME}</h1>
                <p style="color:rgba(255,255,255,0.85);margin:8px 0 0;font-size:14px;">Email Verification</p>
            </div>
            <div style="padding:32px;">
                <p style="color:#333;font-size:15px;margin:0 0 8px;">Hi there,</p>
                <p style="color:#555;font-size:14px;line-height:1.6;">
                    Use the code below to verify your email address and complete your signup.
                    This code expires in <strong>10 minutes</strong>.
                </p>
                <div style="text-align:center;margin:28px 0;">
                    <div style="display:inline-block;background:#f4f4f5;border:2px dashed #6366f1;border-radius:12px;padding:16px 40px;">
                        <span style="font-size:36px;font-weight:700;letter-spacing:8px;color:#1a1a1a;">{otp}</span>
                    </div>
                </div>
                <p style="color:#999;font-size:12px;text-align:center;">
                    If you didn't request this, you can safely ignore this email.
                </p>
            </div>
            <div style="background:#f9fafb;padding:16px;text-align:center;border-top:1px solid #e5e7eb;">
                <p style="color:#999;font-size:11px;margin:0;">&copy; {settings.MAIL_FROM_NAME} &mdash; AI-powered learning platform</p>
            </div>
        </div>
    </body>
    </html>
    '''

    try:
        smtp = smtplib.SMTP(settings.MAIL_SERVER, settings.MAIL_PORT)
        smtp.starttls()
        smtp.login(settings.MAIL_USERNAME, settings.MAIL_PASSWORD)

        msg = MIMEMultipart('related')
        msg['From'] = f"{settings.MAIL_FROM_NAME} <{settings.MAIL_FROM}>"
        msg['To'] = to_email
        msg['Subject'] = f"Your verification code: {otp}"
        msg.attach(MIMEText(html_body, 'html'))
        _attach_logo(msg)

        smtp.sendmail(settings.MAIL_FROM, to_email, msg.as_string())
        smtp.quit()
        print(f"[EmailService] OTP sent to {to_email}", flush=True)
        return True
    except Exception as e:
        import traceback
        print(f"[EmailService] OTP SEND FAILED for {to_email}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return False


def send_password_reset_email(to_email: str, reset_link: str) -> bool:
    """Send a password reset link to the user."""
    print(f"[EmailService] send_password_reset_email called for {to_email}", flush=True)

    if not settings.MAIL_USERNAME or not settings.MAIL_PASSWORD:
        print(f"[EmailService] SKIPPED: SMTP credentials not configured!", flush=True)
        return False

    html_body = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reset your password</title>
    </head>
    <body style="margin:0;padding:0;font-family:Arial,sans-serif;background:#f4f4f5;">
        <div style="max-width:480px;margin:40px auto;background:white;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
            <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);padding:32px;text-align:center;">
                {LOGO_HTML}
                <h1 style="color:white;margin:0;font-size:24px;">{settings.MAIL_FROM_NAME}</h1>
                <p style="color:rgba(255,255,255,0.85);margin:8px 0 0;font-size:14px;">Password Reset</p>
            </div>
            <div style="padding:32px;">
                <p style="color:#333;font-size:15px;margin:0 0 8px;">Hi there,</p>
                <p style="color:#555;font-size:14px;line-height:1.6;">
                    We received a request to reset your password. Click the button below to set a new password.
                    This link expires in <strong>30 minutes</strong>.
                </p>
                <div style="text-align:center;margin:28px 0;">
                    <a href="{reset_link}"
                       style="display:inline-block;background:#6366f1;color:white;padding:14px 32px;
                              text-decoration:none;border-radius:8px;font-weight:600;font-size:15px;">
                        Reset Password
                    </a>
                </div>
                <p style="color:#999;font-size:12px;text-align:center;">
                    Or copy this link: <br>
                    <a href="{reset_link}" style="color:#6366f1;word-break:break-all;">{reset_link}</a>
                </p>
                <p style="color:#999;font-size:12px;text-align:center;margin-top:16px;">
                    If you didn't request this, you can safely ignore this email. Your password won't change.
                </p>
            </div>
            <div style="background:#f9fafb;padding:16px;text-align:center;border-top:1px solid #e5e7eb;">
                <p style="color:#999;font-size:11px;margin:0;">&copy; {settings.MAIL_FROM_NAME} &mdash; AI-powered learning platform</p>
            </div>
        </div>
    </body>
    </html>
    '''

    try:
        smtp = smtplib.SMTP(settings.MAIL_SERVER, settings.MAIL_PORT)
        smtp.starttls()
        smtp.login(settings.MAIL_USERNAME, settings.MAIL_PASSWORD)

        msg = MIMEMultipart('related')
        msg['From'] = f"{settings.MAIL_FROM_NAME} <{settings.MAIL_FROM}>"
        msg['To'] = to_email
        msg['Subject'] = "Reset your password"
        msg.attach(MIMEText(html_body, 'html'))
        _attach_logo(msg)

        smtp.sendmail(settings.MAIL_FROM, to_email, msg.as_string())
        smtp.quit()
        print(f"[EmailService] Reset email sent to {to_email}", flush=True)
        return True
    except Exception as e:
        import traceback
        print(f"[EmailService] RESET EMAIL FAILED for {to_email}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return False


def send_renewal_reminder(to_email: str, plan_name: str, expiry_date: str, renew_url: str) -> bool:
    """Send a subscription renewal reminder email."""
    print(f"[EmailService] send_renewal_reminder called for {to_email}", flush=True)

    if not settings.MAIL_USERNAME or not settings.MAIL_PASSWORD:
        print(f"[EmailService] SKIPPED: SMTP credentials not configured!", flush=True)
        return False

    html_body = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Subscription Renewal Reminder</title>
    </head>
    <body style="margin:0;padding:0;font-family:Arial,sans-serif;background:#f4f4f5;">
        <div style="max-width:480px;margin:40px auto;background:white;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
            <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);padding:32px;text-align:center;">
                {LOGO_HTML}
                <h1 style="color:white;margin:0;font-size:24px;">{settings.MAIL_FROM_NAME}</h1>
                <p style="color:rgba(255,255,255,0.85);margin:8px 0 0;font-size:14px;">Subscription Reminder</p>
            </div>
            <div style="padding:32px;">
                <p style="color:#333;font-size:15px;margin:0 0 8px;">Hi there,</p>
                <p style="color:#555;font-size:14px;line-height:1.6;">
                    Your <strong>{plan_name}</strong> plan is expiring on <strong>{expiry_date}</strong>.
                    Renew now to keep your points, storage, and AI features active without interruption.
                </p>
                <div style="text-align:center;margin:28px 0;">
                    <a href="{renew_url}"
                       style="display:inline-block;background:#6366f1;color:white;padding:14px 32px;
                              text-decoration:none;border-radius:8px;font-weight:600;font-size:15px;">
                        Renew Now
                    </a>
                </div>
                <p style="color:#999;font-size:12px;text-align:center;">
                    If your plan has auto-renewal enabled with Stripe, this is just a heads-up — your subscription will renew automatically.
                </p>
            </div>
            <div style="background:#f9fafb;padding:16px;text-align:center;border-top:1px solid #e5e7eb;">
                <p style="color:#999;font-size:11px;margin:0;">&copy; {settings.MAIL_FROM_NAME} &mdash; AI-powered learning platform</p>
            </div>
        </div>
    </body>
    </html>
    '''

    try:
        smtp = smtplib.SMTP(settings.MAIL_SERVER, settings.MAIL_PORT)
        smtp.starttls()
        smtp.login(settings.MAIL_USERNAME, settings.MAIL_PASSWORD)

        msg = MIMEMultipart('related')
        msg['From'] = f"{settings.MAIL_FROM_NAME} <{settings.MAIL_FROM}>"
        msg['To'] = to_email
        msg['Subject'] = f"Your {plan_name} plan expires soon — renew now"
        msg.attach(MIMEText(html_body, 'html'))
        _attach_logo(msg)

        smtp.sendmail(settings.MAIL_FROM, to_email, msg.as_string())
        smtp.quit()
        print(f"[EmailService] Renewal reminder sent to {to_email}", flush=True)
        return True
    except Exception as e:
        import traceback
        print(f"[EmailService] RENEWAL REMINDER FAILED for {to_email}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return False


async def send_assessment_invitation(
    email_token_pairs: list[tuple[str, str]],
    assessment_title: str,
    teacher_name: str = "Your Teacher",
    due_date: str | None = None,
    time_limit: int | None = None,
):
    """Send assessment invitation emails with per-recipient unique token links."""
    print(f"[EmailService] send_assessment_invitation called for {len(email_token_pairs)} recipients", flush=True)

    if not settings.ENABLE_EMAIL_NOTIFICATIONS:
        print(f"[EmailService] SKIPPED: Email notifications disabled!", flush=True)
        return

    if not settings.MAIL_USERNAME or not settings.MAIL_PASSWORD:
        print(f"[EmailService] SKIPPED: SMTP credentials not configured!", flush=True)
        return

    details_html = ""
    if due_date:
        details_html += f'<p style="margin:4px 0;color:#555;">Due: <strong>{due_date}</strong></p>'
    if time_limit:
        time_minutes = round(time_limit / 60) if time_limit > 60 else time_limit
        details_html += f'<p style="margin:4px 0;color:#555;">Time Limit: <strong>{time_minutes} minutes</strong></p>'

    subject = f"Assessment: {assessment_title}"

    # Open ONE SMTP connection and send personalized email per recipient
    try:
        print(f"[EmailService] Connecting to {settings.MAIL_SERVER}:{settings.MAIL_PORT}...", flush=True)
        smtp = smtplib.SMTP(settings.MAIL_SERVER, settings.MAIL_PORT)
        smtp.starttls()
        smtp.login(settings.MAIL_USERNAME, settings.MAIL_PASSWORD)
        print(f"[EmailService] SMTP login OK", flush=True)

        for email, token in email_token_pairs:
            try:
                assessment_link = f"{settings.FRONTEND_URL}/genverse/take-assessment?token={token}"

                html_body = f'''
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Assessment Invitation</title>
                </head>
                <body style="margin:0;padding:0;font-family:Arial,sans-serif;background:#f4f4f5;">
                    <div style="max-width:520px;margin:40px auto;background:white;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
                        <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);padding:32px;text-align:center;">
                            {LOGO_HTML}
                            <h1 style="color:white;margin:0;font-size:24px;">{settings.MAIL_FROM_NAME}</h1>
                            <p style="color:rgba(255,255,255,0.85);margin:8px 0 0;font-size:14px;">Assessment Invitation</p>
                        </div>
                        <div style="padding:32px;">
                            <p style="color:#333;font-size:15px;margin:0 0 8px;">Hi there,</p>
                            <p style="color:#555;font-size:14px;line-height:1.6;">
                                <strong>{teacher_name}</strong> has assigned you a new assessment:
                            </p>
                            <div style="background:#f9fafb;border-radius:8px;padding:16px;border:1px solid #e5e7eb;margin:16px 0;">
                                <h3 style="margin:0 0 8px;color:#1a1a1a;">{assessment_title}</h3>
                                {details_html}
                            </div>
                            <p style="color:#555;font-size:14px;">Click the button below to start your assessment. No login required.</p>
                            <div style="text-align:center;margin:24px 0;">
                                <a href="{assessment_link}"
                                   style="display:inline-block;background:#6366f1;color:white;
                                          padding:14px 32px;text-decoration:none;border-radius:8px;
                                          font-weight:600;font-size:15px;">
                                    Start Assessment
                                </a>
                            </div>
                            <p style="color:#999;font-size:12px;text-align:center;">
                                Or copy this link:<br>
                                <a href="{assessment_link}" style="color:#6366f1;word-break:break-all;">{assessment_link}</a>
                            </p>
                            <p style="color:#999;font-size:11px;text-align:center;margin-top:16px;">
                                This link is unique to you. Please do not share it with others.
                            </p>
                        </div>
                        <div style="background:#f9fafb;padding:16px;text-align:center;border-top:1px solid #e5e7eb;">
                            <p style="color:#999;font-size:11px;margin:0;">&copy; {settings.MAIL_FROM_NAME} &mdash; AI-powered learning platform</p>
                        </div>
                    </div>
                </body>
                </html>
                '''

                msg = MIMEMultipart('related')
                msg['From'] = f"{settings.MAIL_FROM_NAME} <{settings.MAIL_FROM}>"
                msg['To'] = email
                msg['Subject'] = subject
                msg.attach(MIMEText(html_body, 'html'))
                _attach_logo(msg)

                smtp.sendmail(settings.MAIL_FROM, email, msg.as_string())
                print(f"[EmailService] SENT to {email}", flush=True)
            except Exception as e:
                print(f"[EmailService] FAILED for {email}: {type(e).__name__}: {e}", flush=True)

        smtp.quit()
        print(f"[EmailService] DONE — connection closed", flush=True)
    except Exception as e:
        import traceback
        print(f"[EmailService] SMTP CONNECTION FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


def send_purchase_invoice_email(
    to_email: str,
    user_name: str,
    invoice_id: str,
    transaction_id: str,
    item_label: str,
    item_type: str,
    plan_name: str,
    amount_inr: int,
    payment_gateway: str,
    is_renewal: bool = False,
    billing_period_start: str | None = None,
    billing_period_end: str | None = None,
    promo_code: str | None = None,
    promo_discount: int = 0,
    original_amount: int | None = None,
) -> bool:
    """Send a purchase/renewal invoice email with inline HTML invoice."""
    action_word = "Auto-Renewal" if is_renewal else "Purchase"
    print(f"[EmailService] send_purchase_invoice_email ({action_word}) to {to_email}", flush=True)

    if not settings.MAIL_USERNAME or not settings.MAIL_PASSWORD:
        print(f"[EmailService] SKIPPED: SMTP credentials not configured!", flush=True)
        return False

    now_str = datetime.utcnow().strftime("%B %d, %Y %I:%M %p UTC")
    amount_str = f"₹{amount_inr:,}" if amount_inr > 0 else "Free"
    gateway_label = payment_gateway.capitalize() if payment_gateway else "N/A"

    billing_row = ""
    if billing_period_start and billing_period_end:
        billing_row = f'''
        <tr>
            <td style="padding:8px 12px;color:#666;font-size:13px;">Billing Period</td>
            <td style="padding:8px 12px;text-align:right;font-size:13px;">{billing_period_start} — {billing_period_end}</td>
        </tr>'''

    promo_row = ""
    if promo_code and promo_discount > 0:
        original_str = f"₹{original_amount:,}" if original_amount else ""
        promo_row = f'''
        <tr style="background:#fef3c7;">
            <td style="padding:8px 12px;color:#92400e;font-size:13px;">🎉 Promo Code</td>
            <td style="padding:8px 12px;text-align:right;font-size:13px;font-weight:600;color:#92400e;">
                {promo_code} &mdash; ₹{promo_discount:,} off{f" (was {original_str})" if original_str else ""}
            </td>
        </tr>'''
        if item_type == "Plan Subscription":
            promo_row += '''
        <tr>
            <td colspan="2" style="padding:6px 12px;color:#b45309;font-size:11px;font-style:italic;">
                Note: Promo discount applies to the first month only. Subsequent renewals will be at the regular price.
            </td>
        </tr>'''

    html_body = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Payment Invoice</title>
    </head>
    <body style="margin:0;padding:0;font-family:Arial,sans-serif;background:#f4f4f5;">
        <div style="max-width:560px;margin:40px auto;background:white;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
            <!-- Header -->
            <div style="background:linear-gradient(135deg,#327f6d,#2a9d8f);padding:28px 32px;text-align:center;">
                {LOGO_HTML}
                <h1 style="color:white;margin:0;font-size:22px;">{settings.MAIL_FROM_NAME}</h1>
                <p style="color:rgba(255,255,255,0.85);margin:6px 0 0;font-size:13px;">Payment {action_word} Confirmation</p>
            </div>

            <!-- Body -->
            <div style="padding:28px 32px;">
                <p style="color:#333;font-size:15px;margin:0 0 6px;">Hi {user_name},</p>
                <p style="color:#555;font-size:14px;line-height:1.6;margin:0 0 20px;">
                    {'Your subscription has been automatically renewed.' if is_renewal else 'Thank you for your purchase!'} Here are your invoice details:
                </p>

                <!-- Invoice Card -->
                <div style="border:1px solid #e5e7eb;border-radius:10px;overflow:hidden;margin-bottom:20px;">
                    <!-- Invoice header -->
                    <div style="background:#f9fafb;padding:12px 16px;border-bottom:1px solid #e5e7eb;">
                        <table width="100%" cellpadding="0" cellspacing="0">
                            <tr>
                                <td>
                                    <span style="font-weight:700;color:#327f6d;font-size:14px;">INVOICE</span>
                                </td>
                                <td style="text-align:right;">
                                    <span style="color:#666;font-size:12px;">{invoice_id}</span>
                                </td>
                            </tr>
                        </table>
                    </div>

                    <!-- Invoice details -->
                    <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
                        <tr>
                            <td style="padding:8px 12px;color:#666;font-size:13px;">Date</td>
                            <td style="padding:8px 12px;text-align:right;font-size:13px;">{now_str}</td>
                        </tr>
                        <tr style="background:#f9fafb;">
                            <td style="padding:8px 12px;color:#666;font-size:13px;">Transaction ID</td>
                            <td style="padding:8px 12px;text-align:right;font-size:12px;font-family:monospace;color:#333;">{transaction_id}</td>
                        </tr>
                        <tr>
                            <td style="padding:8px 12px;color:#666;font-size:13px;">Item</td>
                            <td style="padding:8px 12px;text-align:right;font-size:13px;font-weight:600;">{item_label}</td>
                        </tr>
                        <tr style="background:#f9fafb;">
                            <td style="padding:8px 12px;color:#666;font-size:13px;">Type</td>
                            <td style="padding:8px 12px;text-align:right;font-size:13px;">{item_type}</td>
                        </tr>
                        <tr>
                            <td style="padding:8px 12px;color:#666;font-size:13px;">Plan</td>
                            <td style="padding:8px 12px;text-align:right;font-size:13px;">{plan_name}</td>
                        </tr>
                        <tr style="background:#f9fafb;">
                            <td style="padding:8px 12px;color:#666;font-size:13px;">Payment Method</td>
                            <td style="padding:8px 12px;text-align:right;font-size:13px;">{gateway_label}</td>
                        </tr>{billing_row}{promo_row}
                    </table>

                    <!-- Total -->
                    <div style="background:#327f6d;padding:14px 16px;">
                        <table width="100%" cellpadding="0" cellspacing="0">
                            <tr>
                                <td style="color:white;font-size:15px;font-weight:700;">Total Paid</td>
                                <td style="text-align:right;color:white;font-size:18px;font-weight:700;">{amount_str}</td>
                            </tr>
                        </table>
                    </div>
                </div>

                <p style="color:#999;font-size:12px;text-align:center;margin:16px 0 0;">
                    You can download the full invoice PDF from your
                    <a href="{settings.FRONTEND_URL}/common/profile" style="color:#327f6d;text-decoration:none;font-weight:600;">Profile → Transactions</a> page.
                </p>
            </div>

            <!-- Footer -->
            <div style="background:#f9fafb;padding:16px;text-align:center;border-top:1px solid #e5e7eb;">
                <p style="color:#999;font-size:11px;margin:0;">
                    This is an auto-generated invoice. For queries, contact support@genverse.ai
                </p>
                <p style="color:#bbb;font-size:10px;margin:6px 0 0;">&copy; {settings.MAIL_FROM_NAME} — AI-powered learning platform</p>
            </div>
        </div>
    </body>
    </html>
    '''

    subject = (
        f"{'Renewal' if is_renewal else 'Purchase'} Invoice — {item_label} ({amount_str})"
    )

    try:
        smtp = smtplib.SMTP(settings.MAIL_SERVER, settings.MAIL_PORT)
        smtp.starttls()
        smtp.login(settings.MAIL_USERNAME, settings.MAIL_PASSWORD)

        msg = MIMEMultipart('related')
        msg['From'] = f"{settings.MAIL_FROM_NAME} <{settings.MAIL_FROM}>"
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(html_body, 'html'))
        _attach_logo(msg)

        smtp.sendmail(settings.MAIL_FROM, to_email, msg.as_string())
        smtp.quit()
        print(f"[EmailService] Invoice email sent to {to_email} for {item_label}", flush=True)
        return True
    except Exception as e:
        import traceback
        print(f"[EmailService] INVOICE EMAIL FAILED for {to_email}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return False


def send_organization_invitation_email(
    to_email: str,
    admin_name: str,
    organization_name: str,
    role: str,
    accept_url: str,
    is_existing_user: bool = False,
) -> bool:
    """Send an organization invitation email to a teacher/member.

    If is_existing_user is True, the email tells them they've been added directly.
    Otherwise, it includes a signup + accept-invitation link.
    """
    print(f"[EmailService] send_organization_invitation_email to {to_email}", flush=True)

    if not settings.MAIL_USERNAME or not settings.MAIL_PASSWORD:
        print(f"[EmailService] SKIPPED: SMTP credentials not configured!", flush=True)
        return False

    role_display = role.replace("_", " ").title()

    if is_existing_user:
        headline = "You've Been Added!"
        intro = f"""
            <p style="color:#555;font-size:14px;line-height:1.6;">
                Great news! <strong>{admin_name}</strong> has added you as a
                <strong>{role_display}</strong> to the <strong>{organization_name}</strong>
                workspace on {settings.MAIL_FROM_NAME}.
            </p>
            <p style="color:#555;font-size:14px;line-height:1.6;">
                Switch to the organization workspace to get started.
            </p>
        """
        button_text = "Open Workspace"
        button_url = f"{settings.FRONTEND_URL}/genverse/dashboard"
    else:
        headline = "You're Invited!"
        intro = f"""
            <p style="color:#555;font-size:14px;line-height:1.6;">
                <strong>{admin_name}</strong> has invited you to join
                <strong>{organization_name}</strong> as a <strong>{role_display}</strong>
                on {settings.MAIL_FROM_NAME}.
            </p>
            <p style="color:#555;font-size:14px;line-height:1.6;">
                Click the button below to accept your invitation. If you don't have an account yet,
                you'll be able to sign up first — your invitation will be applied automatically.
            </p>
        """
        button_text = "Accept Invitation"
        button_url = accept_url

    extra_invite_section = ""
    if not is_existing_user:
        extra_invite_section = (
            '<p style="color:#999;font-size:12px;text-align:center;">'
            "Or copy this link:<br>"
            f'<a href="{button_url}" style="color:#6366f1;word-break:break-all;">{button_url}</a>'
            "</p>"
            '<p style="color:#999;font-size:11px;text-align:center;margin-top:16px;">'
            "This invitation expires in <strong>7 days</strong>. If you did not expect this, you can safely ignore it."
            "</p>"
        )

    html_body = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Organization Invitation</title>
    </head>
    <body style="margin:0;padding:0;font-family:Arial,sans-serif;background:#f4f4f5;">
        <div style="max-width:520px;margin:40px auto;background:white;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
            <!-- Header -->
            <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);padding:32px;text-align:center;">
                {LOGO_HTML}
                <h1 style="color:white;margin:0;font-size:24px;">{settings.MAIL_FROM_NAME}</h1>
                <p style="color:rgba(255,255,255,0.85);margin:8px 0 0;font-size:14px;">{headline}</p>
            </div>

            <!-- Body -->
            <div style="padding:32px;">
                <p style="color:#333;font-size:15px;margin:0 0 8px;">Hi there,</p>
                {intro}

                <!-- Org Info Card -->
                <div style="background:#f9fafb;border-radius:10px;padding:20px;border:1px solid #e5e7eb;margin:20px 0;">
                    <table width="100%" cellpadding="0" cellspacing="0">
                        <tr>
                            <td style="padding:6px 0;color:#666;font-size:13px;">Organization</td>
                            <td style="padding:6px 0;text-align:right;font-size:13px;font-weight:600;color:#1a1a1a;">{organization_name}</td>
                        </tr>
                        <tr>
                            <td style="padding:6px 0;color:#666;font-size:13px;">Your Role</td>
                            <td style="padding:6px 0;text-align:right;">
                                <span style="display:inline-block;background:#6366f1;color:white;padding:3px 12px;
                                             border-radius:12px;font-size:12px;font-weight:600;">{role_display}</span>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding:6px 0;color:#666;font-size:13px;">Invited By</td>
                            <td style="padding:6px 0;text-align:right;font-size:13px;color:#1a1a1a;">{admin_name}</td>
                        </tr>
                    </table>
                </div>

                <!-- CTA Button -->
                <div style="text-align:center;margin:28px 0;">
                    <a href="{button_url}"
                       style="display:inline-block;background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;
                              padding:14px 36px;text-decoration:none;border-radius:8px;font-weight:600;
                              font-size:15px;box-shadow:0 4px 12px rgba(99,102,241,0.3);">
                        {button_text}
                    </a>
                </div>

                {extra_invite_section}
            </div>

            <!-- Footer -->
            <div style="background:#f9fafb;padding:16px;text-align:center;border-top:1px solid #e5e7eb;">
                <p style="color:#999;font-size:11px;margin:0;">&copy; {settings.MAIL_FROM_NAME} &mdash; AI-powered learning platform</p>
            </div>
        </div>
    </body>
    </html>
    '''

    subject = f"{admin_name} invited you to join {organization_name} on {settings.MAIL_FROM_NAME}"

    try:
        smtp = smtplib.SMTP(settings.MAIL_SERVER, settings.MAIL_PORT)
        smtp.starttls()
        smtp.login(settings.MAIL_USERNAME, settings.MAIL_PASSWORD)

        msg = MIMEMultipart('related')
        msg['From'] = f"{settings.MAIL_FROM_NAME} <{settings.MAIL_FROM}>"
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(html_body, 'html'))
        _attach_logo(msg)

        smtp.sendmail(settings.MAIL_FROM, to_email, msg.as_string())
        smtp.quit()
        print(f"[EmailService] Invitation email sent to {to_email} for org '{organization_name}'", flush=True)
        return True
    except Exception as e:
        import traceback
        print(f"[EmailService] INVITATION EMAIL FAILED for {to_email}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return False


def send_class_invitation_email(
    to_email: str,
    teacher_name: str,
    class_name: str,
    organization_name: str | None = None,
    join_code: str | None = None,
) -> bool:
    """Send an invitation email when a teacher adds a student (by email) to a class."""
    print(f"[EmailService] send_class_invitation_email to {to_email} for class '{class_name}'", flush=True)

    if not settings.MAIL_USERNAME or not settings.MAIL_PASSWORD:
        print(f"[EmailService] SKIPPED: SMTP credentials not configured!", flush=True)
        return False

    org_line = f" at <strong>{organization_name}</strong>" if organization_name else ""
    signup_url = f"{settings.FRONTEND_URL}/genverse/get-started"

    html_body = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Class Invitation</title>
    </head>
    <body style="margin:0;padding:0;font-family:Arial,sans-serif;background:#f4f4f5;">
        <div style="max-width:520px;margin:40px auto;background:white;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
            <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);padding:32px;text-align:center;">
                {LOGO_HTML}
                <h1 style="color:white;margin:0;font-size:24px;">{settings.MAIL_FROM_NAME}</h1>
                <p style="color:rgba(255,255,255,0.85);margin:8px 0 0;font-size:14px;">Class Invitation</p>
            </div>
            <div style="padding:32px;">
                <p style="color:#333;font-size:15px;margin:0 0 8px;">Hi there,</p>
                <p style="color:#555;font-size:14px;line-height:1.6;">
                    <strong>{teacher_name}</strong> has invited you to join the class
                    <strong>{class_name}</strong>{org_line} on {settings.MAIL_FROM_NAME}.
                </p>
                <div style="background:#f9fafb;border-radius:10px;padding:20px;border:1px solid #e5e7eb;margin:20px 0;">
                    <table width="100%" cellpadding="0" cellspacing="0">
                        <tr>
                            <td style="padding:6px 0;color:#666;font-size:13px;">Class</td>
                            <td style="padding:6px 0;text-align:right;font-size:13px;font-weight:600;color:#1a1a1a;">{class_name}</td>
                        </tr>
                        <tr>
                            <td style="padding:6px 0;color:#666;font-size:13px;">Teacher</td>
                            <td style="padding:6px 0;text-align:right;font-size:13px;color:#1a1a1a;">{teacher_name}</td>
                        </tr>
                        {f'<tr><td style="padding:6px 0;color:#666;font-size:13px;">Join Code</td><td style="padding:6px 0;text-align:right;font-size:16px;font-weight:700;letter-spacing:2px;color:#6366f1;">{join_code}</td></tr>' if join_code else ''}
                    </table>
                </div>
                <p style="color:#555;font-size:14px;">
                    Sign up on {settings.MAIL_FROM_NAME} with this email address (<strong>{to_email}</strong>) and you'll be automatically enrolled in the class.
                </p>
                <div style="text-align:center;margin:28px 0;">
                    <a href="{signup_url}"
                       style="display:inline-block;background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;
                              padding:14px 36px;text-decoration:none;border-radius:8px;font-weight:600;
                              font-size:15px;box-shadow:0 4px 12px rgba(99,102,241,0.3);">
                        Sign Up &amp; Join Class
                    </a>
                </div>
                <p style="color:#999;font-size:12px;text-align:center;">
                    If you already have an account, just log in — the class will appear in your dashboard.
                </p>
            </div>
            <div style="background:#f9fafb;padding:16px;text-align:center;border-top:1px solid #e5e7eb;">
                <p style="color:#999;font-size:11px;margin:0;">&copy; {settings.MAIL_FROM_NAME} &mdash; AI-powered learning platform</p>
            </div>
        </div>
    </body>
    </html>
    '''

    try:
        smtp = smtplib.SMTP(settings.MAIL_SERVER, settings.MAIL_PORT)
        smtp.starttls()
        smtp.login(settings.MAIL_USERNAME, settings.MAIL_PASSWORD)

        msg = MIMEMultipart('related')
        msg['From'] = f"{settings.MAIL_FROM_NAME} <{settings.MAIL_FROM}>"
        msg['To'] = to_email
        msg['Subject'] = f"{teacher_name} invited you to join {class_name} on {settings.MAIL_FROM_NAME}"
        msg.attach(MIMEText(html_body, 'html'))
        _attach_logo(msg)

        smtp.sendmail(settings.MAIL_FROM, to_email, msg.as_string())
        smtp.quit()
        print(f"[EmailService] Class invitation sent to {to_email} for '{class_name}'", flush=True)
        return True
    except Exception as e:
        import traceback
        print(f"[EmailService] CLASS INVITATION FAILED for {to_email}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return False


def send_feedback_notification_email(
    user_name: str,
    user_email: str,
    rating: int,
    comment: str,
    page: str,
) -> bool:
    """Send a feedback notification email to the support team."""
    print(f"[EmailService] send_feedback_notification_email from {user_email}", flush=True)

    if not settings.MAIL_USERNAME or not settings.MAIL_PASSWORD:
        print(f"[EmailService] SKIPPED: SMTP credentials not configured!", flush=True)
        return False

    stars_html = "".join(
        f'<span style="color:#f59e0b;font-size:24px;">{"&#9733;" if i < rating else "&#9734;"}</span>'
        for i in range(5)
    )
    rating_label = ["", "Poor", "Fair", "Good", "Great", "Excellent"][rating]
    now_str = datetime.now(timezone.utc).strftime("%B %d, %Y %I:%M %p UTC")
    comment_html = (
        f'<div style="background:#f9fafb;border-radius:8px;padding:16px;margin:16px 0;border-left:4px solid #6366f1;">'
        f'<p style="color:#333;font-size:14px;line-height:1.6;margin:0;white-space:pre-wrap;">{comment}</p>'
        f'</div>'
        if comment
        else '<p style="color:#999;font-size:13px;font-style:italic;">No comment provided.</p>'
    )

    html_body = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>New Feedback</title>
    </head>
    <body style="margin:0;padding:0;font-family:Arial,sans-serif;background:#f4f4f5;">
        <div style="max-width:520px;margin:40px auto;background:white;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
            <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);padding:28px 32px;text-align:center;">
                {LOGO_HTML}
                <h1 style="color:white;margin:0;font-size:22px;">{settings.MAIL_FROM_NAME}</h1>
                <p style="color:rgba(255,255,255,0.85);margin:6px 0 0;font-size:13px;">New User Feedback Received</p>
            </div>
            <div style="padding:28px 32px;">
                <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:16px;">
                    <tr>
                        <td style="color:#666;font-size:13px;padding:6px 0;">From</td>
                        <td style="font-size:13px;font-weight:600;padding:6px 0;">{user_name} ({user_email})</td>
                    </tr>
                    <tr>
                        <td style="color:#666;font-size:13px;padding:6px 0;">Date</td>
                        <td style="font-size:13px;padding:6px 0;">{now_str}</td>
                    </tr>
                    <tr>
                        <td style="color:#666;font-size:13px;padding:6px 0;">Page</td>
                        <td style="font-size:13px;padding:6px 0;">{page or 'N/A'}</td>
                    </tr>
                </table>

                <div style="text-align:center;margin:20px 0;">
                    <div>{stars_html}</div>
                    <p style="margin:4px 0 0;font-size:14px;font-weight:600;color:#333;">{rating_label} ({rating}/5)</p>
                </div>

                <p style="color:#555;font-size:13px;font-weight:600;margin:16px 0 4px;">Comment:</p>
                {comment_html}

                <div style="text-align:center;margin-top:24px;">
                    <a href="mailto:{user_email}?subject=Re:%20Your%20GenVerse%20Feedback"
                       style="display:inline-block;background:#6366f1;color:white;padding:10px 24px;
                              text-decoration:none;border-radius:8px;font-weight:600;font-size:13px;">
                        Reply to User
                    </a>
                </div>
            </div>
            <div style="background:#f9fafb;padding:14px;text-align:center;border-top:1px solid #e5e7eb;">
                <p style="color:#999;font-size:11px;margin:0;">&copy; {settings.MAIL_FROM_NAME} — Feedback Notification</p>
            </div>
        </div>
    </body>
    </html>
    '''

    try:
        smtp = smtplib.SMTP(settings.MAIL_SERVER, settings.MAIL_PORT)
        smtp.starttls()
        smtp.login(settings.MAIL_USERNAME, settings.MAIL_PASSWORD)

        msg = MIMEMultipart('related')
        msg['From'] = f"{settings.MAIL_FROM_NAME} <{settings.MAIL_FROM}>"
        msg['To'] = SUPPORT_EMAIL
        msg['Subject'] = f"[Feedback] {rating_label} ({rating}/5) from {user_name}"
        msg.attach(MIMEText(html_body, 'html'))
        _attach_logo(msg)

        smtp.sendmail(settings.MAIL_FROM, SUPPORT_EMAIL, msg.as_string())
        smtp.quit()
        print(f"[EmailService] Feedback notification sent to {SUPPORT_EMAIL}", flush=True)
        return True
    except Exception as e:
        import traceback
        print(f"[EmailService] FEEDBACK EMAIL FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return False
