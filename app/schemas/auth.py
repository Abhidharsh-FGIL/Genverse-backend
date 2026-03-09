from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional


class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str = "normal_user"  # normal_user | teacher | student | guardian
    otp: str  # 6-digit OTP from email verification

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed = {"normal_user", "teacher", "student", "guardian"}
        if v not in allowed:
            raise ValueError(f"Role must be one of: {', '.join(allowed)}")
        return v


class OrgSignupRequest(BaseModel):
    org_name: str
    admin_name: str
    email: EmailStr
    password: str
    product_type: str = "genverse"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    is_new_user: bool = False


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr
    frontend_url: str = ""  # optional: frontend sends its origin so the reset link is correct


class ResetPasswordRequest(BaseModel):
    token: str
    password: str

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class GoogleLoginRequest(BaseModel):
    token: str
    auto_signup: bool = False


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class SendOtpRequest(BaseModel):
    email: EmailStr


class VerifyOtpRequest(BaseModel):
    email: EmailStr
    otp: str
