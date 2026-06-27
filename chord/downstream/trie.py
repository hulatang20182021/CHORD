from __future__ import annotations


class Trie:
    def __init__(self) -> None:
        self.children: dict[str, "Trie"] = {}
        self.terminal = False

    def insert(self, tokens: list[str]) -> None:
        node = self
        for token in tokens:
            node = node.children.setdefault(str(token), Trie())
        node.terminal = True

    def next_tokens(self, prefix: list[str]) -> list[str]:
        node = self
        for token in prefix:
            token = str(token)
            if token not in node.children:
                return []
            node = node.children[token]
        return sorted(node.children)
