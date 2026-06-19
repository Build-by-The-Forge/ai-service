from pydantic import BaseModel, Field
from typing import List, Optional


class TreeNode(BaseModel):
    id: str
    title: str
    description: Optional[str] = None

    level: int = Field(..., description="0=course, 1=topic, 2=subtopic, 3=concept")

    parent_id: Optional[str] = None
    children: List["TreeNode"] = Field(default_factory=list)

    metadata: dict = Field(default_factory=dict)


TreeNode.model_rebuild()
