class RBAC:

    def can_create_exam(self, role: str) -> bool:
        return role in ["teacher", "admin"]

    def can_take_exam(self, role: str) -> bool:
        return role == "student"

    def can_view_results(self, role: str) -> bool:
        return role in ["teacher", "admin"]
