import uuid
from datetime import datetime
from pydantic import BaseModel, EmailStr
from typing import Optional, List


class OrganizationCreate(BaseModel):
    name: str
    logo_url: Optional[str] = None
    product_type: str = "genverse"
    has_genverse: bool = True
    has_evaluation: bool = False
    locked_grade: Optional[int] = None
    locked_board: Optional[str] = None
    enforce_academic_context: bool = False
    plan: str = "free"  # org plan selected during signup


class OrganizationUpdate(BaseModel):
    name: Optional[str] = None
    logo_url: Optional[str] = None
    locked_grade: Optional[int] = None
    locked_board: Optional[str] = None
    enforce_academic_context: Optional[bool] = None
    default_theme: Optional[str] = None


class OrganizationResponse(BaseModel):
    id: uuid.UUID
    name: str
    logo_url: Optional[str] = None
    product_type: str
    has_genverse: bool
    has_evaluation: bool
    locked_grade: Optional[int] = None
    locked_board: Optional[str] = None
    enforce_academic_context: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class OrgMemberResponse(BaseModel):
    id: uuid.UUID
    org_id: uuid.UUID
    user_id: uuid.UUID
    role: str
    status: str
    joined_at: datetime
    user_name: Optional[str] = None
    user_email: Optional[str] = None

    model_config = {"from_attributes": True}


class InviteMemberRequest(BaseModel):
    email: EmailStr
    role: str  # org_admin | teacher | student | guardian


class BulkInviteRequest(BaseModel):
    members: List[InviteMemberRequest]


class UpdateMemberRoleRequest(BaseModel):
    role: str


class OrgInvitationResponse(BaseModel):
    id: uuid.UUID
    org_id: uuid.UUID
    email: str
    role: str
    status: str
    token: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DirectAddMemberRequest(BaseModel):
    email: Optional[EmailStr] = None
    user_id: Optional[str] = None  # used by POST /{org_id}/members with known user_id
    role: str  # org_admin | teacher | student | guardian


class ModuleOverrideRequest(BaseModel):
    feature_key: str
    enabled: bool
    access_role: Optional[str] = None


class ModuleOverrideResponse(BaseModel):
    id: uuid.UUID
    org_id: uuid.UUID
    feature_key: str
    enabled: bool
    access_role: Optional[str] = None
    updated_at: datetime

    model_config = {"from_attributes": True}


class WorkspaceItem(BaseModel):
    id: str  # 'personal' or org_id
    name: str
    type: str  # 'personal' | 'organization'
    role: Optional[str] = None  # org role if type is 'organization'
    logo_url: Optional[str] = None
    has_genverse: bool = True
    has_evaluation: bool = False
