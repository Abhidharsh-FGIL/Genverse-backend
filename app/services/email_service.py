import smtplib
import asyncio
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from app.config import settings


def _send_smtp(subject: str, html_body: str, to_emails: list[str]):
    """Send an HTML email via SMTP (blocking). Called from a thread."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{settings.MAIL_FROM_NAME} <{settings.MAIL_FROM}>"
    msg["To"] = ", ".join(to_emails)
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(settings.MAIL_SERVER, settings.MAIL_PORT) as server:
        server.ehlo()
        if settings.MAIL_STARTTLS:
            server.starttls()
            server.ehlo()
        server.login(settings.MAIL_USERNAME, settings.MAIL_PASSWORD)
        server.sendmail(settings.MAIL_FROM, to_emails, msg.as_string())


async def send_assessment_invitation(
    to_emails: list[str],
    assessment_title: str,
    assessment_id: str,
    teacher_name: str = "Your Teacher",
    due_date: str | None = None,
    time_limit: int | None = None,
):
    """Send assessment invitation emails to students."""
    print(f"[EmailService] send_assessment_invitation called")
    print(f"[EmailService]   to_emails={to_emails}")
    print(f"[EmailService]   assessment_title={assessment_title}")
    print(f"[EmailService]   ENABLE_EMAIL_NOTIFICATIONS={settings.ENABLE_EMAIL_NOTIFICATIONS}")
    print(f"[EmailService]   MAIL_USERNAME={settings.MAIL_USERNAME}")
    print(f"[EmailService]   MAIL_SERVER={settings.MAIL_SERVER}:{settings.MAIL_PORT}")

    if not settings.ENABLE_EMAIL_NOTIFICATIONS:
        print(f"[EmailService] SKIPPED: Email notifications disabled!")
        return

    if not settings.MAIL_USERNAME or not settings.MAIL_PASSWORD:
        print(f"[EmailService] SKIPPED: SMTP credentials not configured!")
        return

    assessment_link = f"{settings.FRONTEND_URL}/assessment/{assessment_id}"

    details_html = ""
    if due_date:
        details_html += f'<p style="margin:4px 0;color:#555;">Due: <strong>{due_date}</strong></p>'
    if time_limit:
        details_html += f'<p style="margin:4px 0;color:#555;">Time Limit: <strong>{time_limit} minutes</strong></p>'

    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:560px;margin:0 auto;padding:24px;">
      <div style="text-align:center;margin-bottom:24px;">
        <h2 style="color:#1a1a1a;margin:0;">New Assessment Assigned</h2>
      </div>
      <div style="background:#f8f9fa;border-radius:12px;padding:24px;margin-bottom:20px;">
        <p style="margin:0 0 8px;color:#666;">Hi there,</p>
        <p style="margin:0 0 16px;color:#333;">
          <strong>{teacher_name}</strong> has assigned you a new assessment:
        </p>
        <div style="background:white;border-radius:8px;padding:16px;border:1px solid #e5e7eb;">
          <h3 style="margin:0 0 8px;color:#1a1a1a;">{assessment_title}</h3>
          {details_html}
        </div>
      </div>
      <div style="text-align:center;margin-bottom:20px;">
        <a href="{assessment_link}"
           style="display:inline-block;background:#4f46e5;color:white;text-decoration:none;
                  padding:12px 32px;border-radius:8px;font-weight:600;font-size:15px;">
          Start Assessment
        </a>
      </div>
      <p style="text-align:center;color:#999;font-size:12px;margin:0;">
        Or copy this link: {assessment_link}
      </p>
      <hr style="border:none;border-top:1px solid #eee;margin:24px 0 12px;" />
      <p style="text-align:center;color:#bbb;font-size:11px;margin:0;">
        Sent by {settings.MAIL_FROM_NAME}
      </p>
    </div>
    """

    subject = f"Assessment: {assessment_title}"

    try:
        print(f"[EmailService] Sending via smtplib to {to_emails}...")
        # Run blocking SMTP in a thread so we don't block the event loop
        await asyncio.to_thread(
            _send_smtp, subject, html_body, to_emails
        )
        print(f"[EmailService] SUCCESS: Email delivered to {to_emails}")
    except Exception as e:
        import traceback
        print(f"[EmailService] FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise
