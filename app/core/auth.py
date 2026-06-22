from app.auth.dependencies import (
    require_authenticated_user,
    require_student,
    require_teacher_or_admin,
    require_admin,
)
from app.auth.permissions import rbac

# Backward compatibility: old imports from app.core.auth still work.
