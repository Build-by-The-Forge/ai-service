from typing import List, Optional
from app.schemas.tree import TreeNode


class TreeTraversalService:
    def find_node(self, root: TreeNode, node_id: str) -> Optional[TreeNode]:
        if root.id == node_id:
            return root

        for child in root.children:
            result = self.find_node(child, node_id)
            if result:
                return result

        return None

    def get_next_nodes(self, root: TreeNode, node_id: str) -> List[TreeNode]:
        node = self.find_node(root, node_id)

        if not node:
            return []

        return node.children

    def flatten_tree(self, root: TreeNode) -> List[TreeNode]:
        result: List[TreeNode] = []

        def dfs(node: TreeNode) -> None:
            result.append(node)
            for child in node.children:
                dfs(child)

        dfs(root)
        return result
