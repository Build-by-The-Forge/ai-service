from pydantic import BaseModel, Field


class TreeNode(BaseModel):
    id: str
    title: str
    description: str | None = None

    level: int = Field(..., description="0=course, 1=topic, 2=subtopic, 3=concept")

    parent_id: str | None = None
    children: list["TreeNode"] = Field(default_factory=list)

    metadata: dict = Field(default_factory=dict)


TreeNode.model_rebuild()
