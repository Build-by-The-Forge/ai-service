import uuid
from typing import List
from app.schemas.tree import TreeNode


class TreeGeneratorService:
    """
    Converts raw course input into structured learning tree.
    """

    def generate_base_tree(self, course_title: str, topics: List[str]) -> TreeNode:
        root = TreeNode(
            id=str(uuid.uuid4()),
            title=course_title,
            level=0,
            children=[],
        )

        for topic in topics:
            topic_node = TreeNode(
                id=str(uuid.uuid4()),
                title=topic,
                level=1,
                parent_id=root.id,
                children=[],
            )

            root.children.append(topic_node)

        return root
