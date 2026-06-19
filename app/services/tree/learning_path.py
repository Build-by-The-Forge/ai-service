from typing import List
from app.schemas.tree import TreeNode
from app.services.tree.traversal import TreeTraversalService


class LearningPathService:

    def __init__(self):
        self.traversal = TreeTraversalService()

    def generate_path(self, root: TreeNode) -> List[TreeNode]:
        """
        Default DFS learning path (will later become AI-adaptive)
        """
        return self.traversal.flatten_tree(root)

    def next_step(self, root: TreeNode, current_id: str) -> TreeNode | None:
        node = self.traversal.find_node(root, current_id)

        if node and node.children:
            return node.children[0]

        flat = self.traversal.flatten_tree(root)

        for i, n in enumerate(flat):
            if n.id == current_id and i + 1 < len(flat):
                return flat[i + 1]

        return None
