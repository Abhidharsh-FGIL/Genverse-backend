import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from app.config import settings


async def send_assessment_invitation(
    to_emails: list[str],
    assessment_title: str,
    assessment_id: str,
    teacher_name: str = "Your Teacher",
    due_date: str | None = None,
    time_limit: int | None = None,
):
    """Send assessment invitation emails to students."""
    print(f"[EmailService] send_assessment_invitation called", flush=True)
    print(f"[EmailService]   to_emails={to_emails}", flush=True)

    if not settings.ENABLE_EMAIL_NOTIFICATIONS:
        print(f"[EmailService] SKIPPED: Email notifications disabled!", flush=True)
        return

    if not settings.MAIL_USERNAME or not settings.MAIL_PASSWORD:
        print(f"[EmailService] SKIPPED: SMTP credentials not configured!", flush=True)
        return

    assessment_link = f"{settings.FRONTEND_URL}/assessment/{assessment_id}"

    details_html = ""
    if due_date:
        details_html += f'<p style="margin:4px 0;color:#555;">Due: <strong>{due_date}</strong></p>'
    if time_limit:
        details_html += f'<p style="margin:4px 0;color:#555;">Time Limit: <strong>{time_limit} minutes</strong></p>'

    html_body = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Assessment Invitation</title>
        <style>
            body, html {{
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
            }}
            .container {{
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .body-content {{
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }}
            .footer {{
                text-align: center;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="body-content">
                <p>Hi there,</p>
                <p><strong>{teacher_name}</strong> has assigned you a new assessment:</p>
                <div style="background:white;border-radius:8px;padding:16px;border:1px solid #e5e7eb;margin:16px 0;">
                    <h3 style="margin:0 0 8px;color:#1a1a1a;">{assessment_title}</h3>
                    {details_html}
                </div>
                <p>Please click on the button below to start your assessment.</p>
                <a href="{assessment_link}"
                   style="display:inline-block;background-color:#4f46e5;color:white;
                          padding:15px 20px;text-align:center;text-decoration:none;
                          border-radius:5px;font-weight:600;">
                    Start Assessment
                </a>
                <p style="color:#999;font-size:12px;margin-top:16px;">
                    Or copy this link: {assessment_link}
                </p>
            </div>
            <div class="footer">
                <p>Best Regards,<br>The {settings.MAIL_FROM_NAME} Team</p>
            </div>
        </div>
    </body>
    </html>
    '''

    subject = f"Assessment: {assessment_title}"

    # Open ONE SMTP connection and send all emails through it
    # (runs synchronously — same as GenVerse OTP pattern)
    try:
        print(f"[EmailService] Connecting to {settings.MAIL_SERVER}:{settings.MAIL_PORT}...", flush=True)
        smtp = smtplib.SMTP(settings.MAIL_SERVER, settings.MAIL_PORT)
        smtp.starttls()
        smtp.login(settings.MAIL_USERNAME, settings.MAIL_PASSWORD)
        print(f"[EmailService] SMTP login OK", flush=True)

        for email in to_emails:
            try:
                msg = MIMEMultipart()
                msg['From'] = settings.MAIL_FROM
                msg['To'] = email
                msg['Subject'] = subject
                msg.attach(MIMEText(html_body, 'html'))

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
